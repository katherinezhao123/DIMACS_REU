from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from peft import LorTaConfig, get_peft_model
from trl import SFTTrainer
import torch
import os
import argparse
import re

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--lorta_alpha", type=int, default=8)
parser.add_argument("--lorta_dropout", type=float, default=0.05)
parser.add_argument("--per_device", type=int, default=4)
parser.add_argument("--gradient_accum", type=int, default=4)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--lr_scheduler", type=str, default="cosine")
parser.add_argument("--output_dir", type=str, default="./lorta_output")
parser.add_argument("--save_dir", type=str, default="./lorta")
parser.add_argument("--val_samples", type=int, default=200,
                    help="How many val examples to actually run inference on (for speed)")
parsed_args = parser.parse_args()


def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


class GSM8KEvalCallback(TrainerCallback):
    """Runs greedy-decoded accuracy on the val split at the end of every epoch."""

    def __init__(self, val_dataset, tokenizer, num_samples=200):
        # num_samples lets you cap inference for speed; set to len(val_dataset) for full eval
        self.eval_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """
        Fires after the built-in Trainer.evaluate() call each epoch.
        We grab the live model from kwargs so it's always up to date.
        """
        if model is None:
            return

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
        # Log through the Trainer's metric system so it appears in wandb/tensorboard
        state.log_history.append({
            "epoch": state.epoch,
            "gsm8k_val_accuracy": accuracy,
            "gsm8k_val_correct": correct,
            "gsm8k_val_total": total,
        })
        print(f"\nEpoch {state.epoch:.0f} | GSM8K Val Accuracy: {accuracy:.3f} ({correct}/{total})")
        model.train()


# ── Data ──────────────────────────────────────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main")

# Carve a validation split from the training data (80/20)
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]   # this is our held-out validation set

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(dataset['test'])}")

# ── Model & tokenizer ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-8B-A1B-Base")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-8B-A1B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ── Callback ──────────────────────────────────────────────────────────────────
gsm8k_callback = GSM8KEvalCallback(
    val_dataset=val_dataset,
    tokenizer=tokenizer,
    num_samples=parsed_args.val_samples,
)

# ── LoRTA config ──────────────────────────────────────────────────────────────
lorta_config = LorTaConfig(
    r=parsed_args.r,
    lora_alpha=parsed_args.lorta_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=parsed_args.lorta_dropout,
    bias="none",
    task_type="CAUSAL_LM", 
)
lorta_config.qkvo_mapping = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "output_proj"
}

# ── Training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=parsed_args.output_dir,
    per_device_train_batch_size=parsed_args.per_device,
    gradient_accumulation_steps=parsed_args.gradient_accum,
    num_train_epochs=parsed_args.num_train_epochs,
    learning_rate=parsed_args.lr,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=parsed_args.warmup_ratio,
    gradient_checkpointing=True,
    lr_scheduler_type=parsed_args.lr_scheduler,
    # Trigger evaluate() at the end of every epoch so on_evaluate fires
    eval_strategy="epoch",
)

def format_example(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,          # 80% of original train
    eval_dataset=val_dataset,             # 20% held-out val — Trainer uses this for loss
    peft_config=lorta_config,
    formatting_func=format_example,
    callbacks=[gsm8k_callback],
    processing_class=tokenizer,
    args=args,
)

trainer.train()

trainer.model.save_pretrained(parsed_args.save_dir)
tokenizer.save_pretrained(parsed_args.save_dir)
print("Training complete. Model saved to " + parsed_args.save_dir)