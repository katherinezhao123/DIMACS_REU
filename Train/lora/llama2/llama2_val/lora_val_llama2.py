from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer
import torch
import os
import argparse
import re

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=8)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--per_device", type=int, default=4)
parser.add_argument("--gradient_accum", type=int, default=4)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--lr_scheduler", type=str, default="cosine")
parser.add_argument("--output_dir", type=str, default="./lora_output")
parser.add_argument("--save_dir", type=str, default="./lora")
parser.add_argument("--target_mod", type = str, nargs = '+', help = "Space-separated list of strings")
parser.add_argument("--val_samples", type=int, default=200)
#["q_proj","v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
parsed_args = parser.parse_args()


def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

class GSM8KEvalCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer, num_samples=200):
        self.eval_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # Save whether gradient checkpointing was enabled
        was_training = model.training
        gc_enabled = getattr(model, "is_gradient_checkpointing", False)

        try:
            # Disable gradient checkpointing for generation
            if gc_enabled:
                model.gradient_checkpointing_disable()

            model.eval()
            correct = 0
            total = len(self.eval_dataset)

            for example in self.eval_dataset:
                prompt = f"Question: {example['question']}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted = extract_answer(generated)
                gold = extract_answer(example["answer"])

                if predicted and gold and predicted == gold:
                    correct += 1

            accuracy = correct / total
            print(f"\nEpoch {state.epoch:.0f} | GSM8K Val Accuracy: {accuracy:.3f} ({correct}/{total})")
        finally:
            if was_training:
                model.train()
            if gc_enabled:
                model.gradient_checkpointing_enable()

            # Restore use_cache=False since gradient checkpointing needs it off
            model.config.use_cache = False



# ── Data ──────────────────────────────────────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main")
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(dataset['test'])}")


# ── Model & tokenizer ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=parsed_args.r,
    lora_alpha=parsed_args.lora_alpha,
    target_modules=parsed_args.target_mod,
    lora_dropout=parsed_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# ── Callback ──────────────────────────────────────────────────────────────────
gsm8k_callback = GSM8KEvalCallback(
    val_dataset=val_dataset,
    tokenizer=tokenizer,
    num_samples=parsed_args.val_samples,
)

# ── Training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=parsed_args.output_dir,
    per_device_train_batch_size=parsed_args.per_device,
    gradient_accumulation_steps=parsed_args.gradient_accum,
    num_train_epochs=parsed_args.num_train_epochs,
    learning_rate=parsed_args.lr,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    warmup_ratio=parsed_args.warmup_ratio,
    gradient_checkpointing=True,
    lr_scheduler_type=parsed_args.lr_scheduler,
    eval_strategy="epoch",
    )

def format_example(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"

train_text_dataset = train_dataset.map(lambda example: {"text": format_example(example)})
val_text_dataset = val_dataset.map(lambda example: {"text": format_example(example)})

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_text_dataset,
    eval_dataset=val_text_dataset,
    peft_config=lora_config,
    formatting_func=format_example,
    processing_class=tokenizer,
    args=args,
    callbacks=[gsm8k_callback],
)

trainer.train()

trainer.model.save_pretrained(parsed_args.save_dir)
tokenizer.save_pretrained(parsed_args.save_dir)
print("Training complete. Model saved to " + parsed_args.save_dir)