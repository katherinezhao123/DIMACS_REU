from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer
import torch
from torch.profiler import profile, ProfilerActivity
from huggingface_hub import login 
import os 


login(token = os.environ["HF_TOKEN"] )

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
    r=8,                       
    lora_alpha=8,               
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  #attention only, focus on different words choice and least ram used could use all linear layers, similar to full fine tuning 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size =4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine"
)

def format_example(example): 
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

dataset = dataset.map(format_example)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    dataset_text_field="text", 
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=args, 
    packing = True
)

prof = None
torch.cuda.memory._record_memory_history(max_entries = 100000)
try:
    with profile( 
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat = 0), #repeat loops indefinitely
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        trainer.add_callback(ProfilerCallback(prof))
        trainer.train()
        trainer.save_model("./lora-adapter")
finally: 
    if prof is not None:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    torch.cuda.memory._dump_snapshot("profile.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)