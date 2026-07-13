from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from peft import LorTaPaddedConfig, get_peft_model
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
parser.add_argument("--target_mod", type = str, nargs = '+', help = "Space-separated list of strings")
parsed_args = parser.parse_args()


def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None



# ── Data ──────────────────────────────────────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main")


# ── Model & tokenizer ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# ── LoRTA config ──────────────────────────────────────────────────────────────
lorta_padded_config = LorTaPaddedConfig(
    r=parsed_args.r,
    lora_alpha=parsed_args.lorta_alpha,
    target_modules=parsed_args.target_mod,
    lora_dropout=parsed_args.lorta_dropout,
    bias="none",
    task_type="CAUSAL_LM", 
    init_lora_weights=True,
)

model.enable_input_require_grads()
model.config.use_cache = False 
model = get_peft_model(model, lorta_padded_config)  
model.print_trainable_parameters()

# ── Training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=parsed_args.output_dir,
    per_device_train_batch_size=parsed_args.per_device,
    gradient_accumulation_steps=parsed_args.gradient_accum,
    num_train_epochs=parsed_args.num_train_epochs,
    learning_rate=parsed_args.lr,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    warmup_ratio=parsed_args.warmup_ratio,
    gradient_checkpointing=False,
    lr_scheduler_type=parsed_args.lr_scheduler,
    dataloader_pin_memory=False,
)

def format_example(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"



# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],             
    formatting_func=format_example,
    processing_class=tokenizer,
    args=args,
)

trainer.train()

trainer.model.save_pretrained(parsed_args.save_dir)
tokenizer.save_pretrained(parsed_args.save_dir)
print("Training complete. Model saved to " + parsed_args.save_dir)