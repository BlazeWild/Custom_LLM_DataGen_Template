from colorama import Fore
from pydantic import BaseModel
from prompts import generation_prompt_template
import json
import re
import glob
import os
import time
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

class Record(BaseModel):
    question: str
    answer: str
    
class Response(BaseModel):
    records: list[Record]

def load_existing_progress():
    """Load existing progress if any"""
    progress_file = "dataset/generation_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"{Fore.CYAN}Resuming from chunk {progress['last_processed_chunk'] + 1}{Fore.RESET}")
        return progress
    return {"last_processed_chunk": -1, "total_generated": 0}

def save_progress(chunk_idx, total_generated):
    """Save current progress"""
    progress = {
        "last_processed_chunk": chunk_idx,
        "total_generated": total_generated
    }
    with open("dataset/generation_progress.json", 'w') as f:
        json.dump(progress, f, indent=2)

def clean_json_breaking_characters(text):
    """Remove only characters that break JSON parsing while preserving markdown"""
    # Remove control characters and non-printable characters except newlines
    # Keep markdown characters: *, #, -, <, >, br
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Don't escape quotes - let the LLM provide proper JSON
    # The code fence removal and JSON parsing should handle it
    
    return cleaned

def llm_call(data: str) -> dict:
    """
    Calls Google Gemini to generate 8-15 Q&A pairs and returns the parsed JSON.
    """
    prompt = generation_prompt_template(data)  # Use the template from prompts.py
    
    response = model.generate_content(prompt)
    data_text = response.text
    
    print(f"{Fore.LIGHTGREEN_EX}LLM Response received{Fore.RESET}")
    
    # Clean special characters that break JSON, but preserve markdown formatting
    data_text = clean_json_breaking_characters(data_text)
    
    # Remove any leading/trailing ``` or ```json fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", data_text, flags=re.MULTILINE | re.DOTALL).strip()
    
    try:
        parsed_data = json.loads(cleaned)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}JSON parsing failed: {e}{Fore.RESET}")
        print(f"Raw response: {cleaned[:500]}...")
        return []


if __name__ == "__main__":
    # Create dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    
    # Check if chunks exist
    if not os.path.exists("chunks"):
        print(f"{Fore.RED}Error: chunks folder not found. Please run chunk_generation.py first.{Fore.RESET}")
        exit(1)
    
    # Load chunk metadata
    metadata_path = os.path.join("chunks", "chunks_metadata.json")
    chunk_files = glob.glob(os.path.join("chunks", "*_chunk_*.json"))
    
    if not chunk_files:
        print(f"{Fore.RED}Error: No chunk JSON files found in chunks folder.{Fore.RESET}")
        exit(1)
    
    total_chunks = len(chunk_files)
    print(f"{Fore.CYAN}Found {total_chunks} JSON chunk files to process{Fore.RESET}")
    
    # Load existing progress
    progress = load_existing_progress()
    start_chunk = progress["last_processed_chunk"] + 1
    
    # Load existing dataset if it exists
    dataset_path = "dataset/raw.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"{Fore.CYAN}Loaded existing dataset with {len(dataset)} entries{Fore.RESET}")
    else:
        dataset = {}
    
    # Process chunks starting from where we left off
    last_api_time = 0
    
    for i in range(start_chunk, total_chunks):
        chunk_path = chunk_files[i]
        
        print(f"{Fore.CYAN}\n=== Processing Chunk {i+1}/{total_chunks}: {os.path.basename(chunk_path)} ==={Fore.RESET}")
        
        # Load JSON chunk content
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        chunk_content = chunk_data['contextualized_text']
        source_info = f"Source: {chunk_data['source_file']}, Chunk: {chunk_data['chunk_index']}"
        
        print(f"{Fore.YELLOW}Chunk preview: {chunk_content[:100]}...{Fore.RESET}")
        print(f"{Fore.YELLOW}{source_info}{Fore.RESET}")
        
        try:
            # Ensure 4 seconds have passed since last API call
            if i > 0:  # Skip wait for first chunk
                time_since_last_api = time.time() - last_api_time
                if time_since_last_api < 4:
                    wait_time = 4 - time_since_last_api
                    print(f"{Fore.MAGENTA}Waiting {wait_time:.1f} seconds before next API call...{Fore.RESET}")
                    time.sleep(wait_time)
            
            print(f"{Fore.BLUE}Calling LLM...{Fore.RESET}")
            last_api_time = time.time()  # Record when API call was made
            data = llm_call(chunk_content)  # Generate 5-10 Q&A pairs per chunk
            
            # Store the generated data
            dataset[str(i)] = {
                "generated": data, 
                "context": chunk_content[:500] + "...",  # Store preview of context
                "chunk_file": os.path.basename(chunk_path),
                "source_info": source_info
            }
            
            print(f"{Fore.GREEN}✓ Chunk {i+1} processed successfully - Generated {len(data)} Q&A pairs{Fore.RESET}")
            
            # Save progress after each chunk
            save_progress(i, progress["total_generated"] + len(data))
            progress["total_generated"] += len(data)
            
            # Save dataset after each chunk (in case of interruption)
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing chunk {i+1}: {e}{Fore.RESET}")
            # Store error info instead of skipping
            dataset[str(i)] = {
                "error": str(e), 
                "context": chunk_content[:500] + "...",
                "chunk_file": os.path.basename(chunk_path)
            }
            # Save progress even on error
            save_progress(i, progress["total_generated"])
            continue
    
    print(f"\n{Fore.GREEN}✓ Processing complete!{Fore.RESET}")
    print(f"Total entries in dataset: {len(dataset)}")
    print(f"Total Q&A pairs generated: {progress['total_generated']}")
    print(f"Dataset saved to: {dataset_path}")
    
    # Final save
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
