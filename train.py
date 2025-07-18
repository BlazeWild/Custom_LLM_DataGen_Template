from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import os
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("json", data_files="final_dataset/filtered.json", split="train")
print(Fore.GREEN + str(dataset[2]) + Fore.RESET)

def format_chat_template(batch, tokenizer):
    system_prompt="""You are a helpful, honest and harmless assistant designed to help about your domain. Think through each question logically and provide an answer. Don't make up things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""
    
    #Chat template for llama 3.2, Use other chat template for other models
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    samples =[]
    questions = batch["question"]
    answers = batch["answer"]
    for i in range(len(questions)):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append(text)
        
    return {
        "instruction": questions,
        "response" : answers,
        "text": samples
    }

auth_token = os.getenv("HF_TOKEN")
base_model = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    base_model, 
    trust_remote_code=True,
    token=auth_token,
    )

train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, 
                            batched=True,
                            batch_size=128,)
print(Fore.YELLOW + str(train_dataset[0]) + Fore.RESET)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=quant_config,
    token=auth_token,
    cache_dir="./workspace",
)

print(Fore.CYAN + str(model) + Fore.RESET)
print(Fore.LIGHTYELLOW_EX + str(next(model.parameters())) + Fore.RESET)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config= LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="meta-llama/Llama-3.2-3b-finetuned",
        num_train_epochs=10,  # Reduced for faster training
        save_steps=500,
        logging_steps=10,
        per_device_train_batch_size=12,  # Optimized for L40S 48GB VRAM
        gradient_accumulation_steps=2,   # Effective batch size = 24
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,  # Use bfloat16 for L40S efficiency
        optim="adamw_torch",  # Adam optimizer
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    ),
    peft_config=peft_config,
)

trainer.train()

trainer.save_model('complete_checkpoint')
trainer.model.save_pretrained('final_model')
