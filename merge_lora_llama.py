#!/usr/bin/env python3
"""
Merge LoRA adapter with base Llama-3.2-3B-Instruct model and save to merged_travel directory.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

auth_token= os.getenv("HF_TOKEN", "your_huggingface_token_here")  # Ensure you have your Hugging Face token set

def merge_lora_with_base():
    """
    Merge the LoRA adapter from final_model_v4 with the base Llama-3.2-3B-Instruct model
    and save the merged model to merged_travel directory.
    """
    
    # Paths
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = "final_model_v4"
    output_path = "merged_travel"
    
    print(f"Loading base model: {base_model_name}")
    
    # Load base model with memory optimization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=auth_token,  # Use your Hugging Face token for authentication
        # low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load the model with LoRA adapter
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float16
    )
    
    print("Merging LoRA weights with base model...")
    
    # Merge LoRA weights into the base model
    merged_model = model_with_lora.merge_and_unload()
    
    # Clean up intermediate models to free memory
    del model_with_lora, base_model
    import gc
    gc.collect()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    
    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Successfully merged and saved model to {output_path}")
    print(f"📁 Model files:")
    for file in os.listdir(output_path):
        print(f"   - {file}")
    
    # Cleanup memory
    del merged_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


if __name__ == "__main__":
    print("🚀 Starting LoRA merge process...")
    
    # Check if LoRA adapter exists
    if not os.path.exists("final_model_v4"):
        print("❌ Error: final_model_v4 directory not found!")
        print("   Please ensure the LoRA adapter is available.")
        exit(1)
    
    # Check if adapter config exists
    if not os.path.exists("final_model_v4/adapter_config.json"):
        print("❌ Error: adapter_config.json not found in final_model_v4!")
        exit(1)
    
    try:
        # Perform the merge
        output_path = merge_lora_with_base()
        
        # Skip test for now - just merge
        # test_merged_model(output_path)
        
        print(f"\n🎉 Merge complete! Your Nepal trekking model is ready at: {output_path}")
        print(f"💡 You can now use this model for inference or deploy it to Ollama.")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
