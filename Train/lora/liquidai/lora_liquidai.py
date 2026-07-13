from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer
import torch
from torch.profiler import profile, ProfilerActivity
from huggingface_hub import login 
import os 
import argparse 

login(token = os.environ["HF_TOKEN"] )

parser = argparse.ArgumentParser()
parser.add_argument("--r", type = int, default =8)
parser.add_argument("--lora_alpha", type =int, default = 8)
parser.add_argument("--lora_dropout", type= float, default = 0.05)
parser.add_argument("--per_device", type = int, default = 4)
parser.add_argument("--gradient_accum", type = int, default =4)
parser.add_argument("--num_train_epochs", type = int, default = 3)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--lr_scheduler", type=str, default="cosine")
parser.add_argument("--output_dir", type=str, default="./lora_output") # checkpoints 
parser.add_argument("--save_dir", type=str, default="./lora") #actual model
parsed_args = parser.parse_args()

class ProfilerCallback (TrainerCallback):
    def __init__(self, prof):
        self.prof = prof
    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()



dataset = load_dataset("openai/gsm8k", "main") 
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-8B-A1B-Base") 
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-8B-A1B-Base", 
    torch_dtype = torch.bfloat16,
    device_map = "auto"
    )
lora_config = LoraConfig(
    r=parsed_args.r,                       
    lora_alpha=parsed_args.lora_alpha,               
    target_modules=["q_proj", "v_proj"],  #attention only, focus on different words choice and least ram used could use all linear layers, similar to full fine tuning 
    lora_dropout=parsed_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

args = TrainingArguments(
    output_dir=parsed_args.output_dir, #"./something"
    per_device_train_batch_size =parsed_args.per_device,
    gradient_accumulation_steps=parsed_args.gradient_accum,
    num_train_epochs=parsed_args.num_train_epochs,
    learning_rate=parsed_args.lr,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=parsed_args.warmup_ratio,
    gradient_checkpointing=True,
    lr_scheduler_type=parsed_args.lr_scheduler
)

def format_example(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    formatting_func= format_example,
    processing_class=tokenizer,
    args=args, 
)
trainer.train()


trainer.model.save_pretrained(parsed_args.save_dir)
tokenizer.save_pretrained(parsed_args.save_dir)

print("Training complete. Model saved to "+ parsed_args.save_dir)