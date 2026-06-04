from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import SFTTrainer
import torch
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

dataset = load_dataset("openai/gsm8k", "main")
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Base")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

args = TrainingArguments(
    output_dir="./output_full",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,        
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    save_total_limit=1,        
)

def format_example(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    formatting_func = format_example,  
    args=args
)

trainer.train()
trainer.save_model("./fpft")
tokenizer.save_pretrained("./fpft")
print("Training complete. Model saved to ./fpft")