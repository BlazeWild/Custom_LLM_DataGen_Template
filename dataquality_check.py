import json
from pydantic import BaseModel
# from litellm import completion
from colorama import Fore
import os
from dotenv import load_dotenv
import re
import time
import google.generativeai as genai
from prompts import quality_check_prompt_template
load_dotenv()

class Score(BaseModel):
    score: int
    explanation: str
    
class Rank(BaseModel):
    accuracy: Score
    style: Score

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

BATCH_SIZE = 5   # Send 5 Q&A pairs per API request
REQUEST_DELAY = 2  # 2 seconds between requests

# Domain configuration - customize for your specific use case
DOMAIN_CONFIG = {
    "domain_name": "your domain",
    "domain_description": "domain-specific topics and information",
    "positive_example": "domain-related question + good domain answer",
    "negative_example": "non-domain question + proper rejection response",
    "rejection_template": "I'm sorry, I can only provide information related to [your domain]"
}

def llm_call_batch(records_batch, domain_config=DOMAIN_CONFIG):
    """Process 5 Q&A pairs in one API call"""
    try:
        prompt = quality_check_prompt_template(records_batch, domain_config)
        
        response = model.generate_content(prompt)
        data = response.text
        
        # Debug: Print first 200 chars of response
        print(f"{Fore.MAGENTA}LLM Response Preview: {data[:200]}...{Fore.RESET}")
        
        # Remove code blocks if present
        if data.startswith("```"):
            data = data.split("```", 2)[1]
            if data.startswith("json"):
                data = data[4:]
        
        # Clean the data more thoroughly
        data = data.strip()
        
        # Try to parse JSON with better error handling
        try:
            parsed_data = json.loads(data)
            print(f"{Fore.MAGENTA}Successfully parsed {len(parsed_data)} objects{Fore.RESET}")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}JSON parsing failed: {e}{Fore.RESET}")
            print(f"{Fore.RED}Error at position {e.pos}: '{data[max(0, e.pos-20):e.pos+20]}'{Fore.RESET}")
            print(f"{Fore.YELLOW}Full raw LLM response:{Fore.RESET}")
            print(response.text)
            print(f"{Fore.YELLOW}End of raw response{Fore.RESET}")
            
            # Fix mixed quotes issue - replace single quotes with double quotes in string values
            try:
                # More comprehensive fix for single quotes
                fixed_data = data
                
                # Fix single quotes around values: ': 'value' -> ": "value"
                fixed_data = re.sub(r": '([^']*)'", r': "\1"', fixed_data)
                
                # Fix single quotes around keys: 'key': -> "key":
                fixed_data = re.sub(r"'([^']*)':", r'"\1":', fixed_data)
                
                # Fix any remaining single quotes that might break JSON
                fixed_data = fixed_data.replace("'", '"')
                
                parsed_data = json.loads(fixed_data)
                print(f"{Fore.GREEN}Fixed quotes and parsed {len(parsed_data)} objects{Fore.RESET}")
                return parsed_data
            except Exception as fix_error:
                print(f"{Fore.RED}Fix attempt failed: {fix_error}{Fore.RESET}")
                # Return empty list instead of fallback
                return []
            
    except Exception as e:
        # Return default scores for all records in batch
        return [{"question": record.get("question", ""), "answer": record.get("answer", ""), 
                "quality": {"accuracy": {"score": 1, "explanation": f"Error: {str(e)}"}, 
                           "style": {"score": 1, "explanation": f"Error: {str(e)}"}}} 
               for record in records_batch]

def load_existing_results():
    """Load existing results if they exist"""
    quality = []
    processed_batches = 0
    
    try:
        if os.path.exists('dataset/quality_results.json'):
            with open('dataset/quality_results.json', 'r') as f:
                quality = json.load(f)
        
        if os.path.exists('dataset/checkpoint.json'):
            with open('dataset/checkpoint.json', 'r') as f:
                checkpoint = json.load(f)
                processed_batches = checkpoint.get('processed_batches', 0)
        
        print(f"{Fore.CYAN}Resuming from batch {processed_batches + 1}, found {len(quality)} existing quality records{Fore.RESET}")
    except:
        print(f"{Fore.YELLOW}Starting fresh{Fore.RESET}")
    
    return quality, processed_batches

def save_checkpoint(quality, batch_idx):
    """Save checkpoint and current results"""
    # Don't overwrite the input file, just save progress
    with open('dataset/quality_results.json', 'w') as f:
        json.dump(quality, f, indent=2)
    
    with open('dataset/checkpoint.json', 'w') as f:
        json.dump({'processed_batches': batch_idx + 1}, f)

def main():
    """Main processing function"""
    print(f"{Fore.CYAN}Starting quality evaluation{Fore.RESET}")
    
    # Load existing results and checkpoint
    quality, start_batch = load_existing_results()
    
    # Initialize instructions list for filtered results
    instructions = []
    
    # Load data
    try:
        with open('dataset/unfiltered.json', 'r') as f:
            data = json.load(f)
        print(f"{Fore.GREEN}Loaded {len(data)} records{Fore.RESET}")
    except FileNotFoundError:
        print(f"{Fore.RED}Error: dataset/unfiltered.json not found{Fore.RESET}")
        return
    
    # Process data in batches
    start_time = time.time()
    
    # Create batches of 5
    batches = [data[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    
    # Skip already processed batches
    for batch_idx in range(start_batch, len(batches)):
        batch = batches[batch_idx]
        print(f"\n{Fore.YELLOW}Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} records){Fore.RESET}")
        
        # Process batch - get 5 results for 5 Q&A pairs
        results = llm_call_batch(batch)
        print(f"{Fore.BLUE}LLM returned {len(results)} results{Fore.RESET}")
        
        # Process the batch results
        batch_passed = 0
        batch_failed = 0
        
        for record, result in zip(batch, results):
            try:
                quality_data = result.get('quality', {})
                accuracy_score = quality_data.get('accuracy', {}).get('score', 1)
                style_score = quality_data.get('style', {}).get('score', 1)
            except:
                accuracy_score = 1
                style_score = 1
            
            if accuracy_score > 6 and style_score > 6:
                instructions.append(record)
                quality.append({**record, 'quality': result})
                batch_passed += 1
            else:
                batch_failed += 1
        
        # Print batch statistics
        total_processed = len(instructions)
        total_records = (batch_idx + 1) * BATCH_SIZE
        overall_pass_rate = (total_processed / total_records) * 100
        
        print(f"{Fore.GREEN}✓ {batch_passed} passed{Fore.RESET}, {Fore.RED}✗ {batch_failed} failed{Fore.RESET}")
        print(f"{Fore.CYAN}Overall: {total_processed}/{total_records} passed ({overall_pass_rate:.1f}%){Fore.RESET}")
        
        # Save checkpoint every batch
        save_checkpoint(quality, batch_idx)
        
        # Wait 4 seconds between API calls
        if batch_idx < len(batches) - 1:
            time.sleep(REQUEST_DELAY)
    
    end_time = time.time()
    
    # Save final results
    with open('final_dataset/filtered.json', 'w') as f:
        json.dump(instructions, f, indent=2)
    
    with open('dataset/qualityresults.json', 'w') as f:
        json.dump(quality, f, indent=2)
    
    # Print final statistics
    print(f"\n{Fore.CYAN}Final Results:{Fore.RESET}")
    print(f"Total records processed: {len(data)}")
    print(f"Records that passed quality check: {len(instructions)}")
    print(f"Pass rate: {len(instructions)/len(data)*100:.1f}%")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Quality data saved to: final_dataset/filtered.json")

if __name__ == "__main__":
    main()
