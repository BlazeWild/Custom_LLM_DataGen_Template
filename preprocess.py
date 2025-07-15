import json
from colorama import Fore
import os

os.makedirs('data', exist_ok=True)

instructions = []
with open('dataset/raw.json', 'r') as f: 
    data = json.load(f)
    
    # First, let's examine the structure
    print(f"Data keys: {list(data.keys())[:5]}")  # Show first 5 keys
    first_key = list(data.keys())[0]
    print(f"First chunk structure: {data[first_key].keys()}")
    print(f"Sample chunk: {data[first_key]}")
    
    for key, chunk in data.items(): 
        # Check if 'generated' key exists, if not, look for the actual structure
        if 'generated' in chunk:
            pairs_data = chunk['generated']
        elif 'records' in chunk:
            pairs_data = chunk['records']
        else:
            # If it's a direct list of Q&A pairs
            print(f"Chunk {key} structure: {chunk.keys()}")
            continue
            
        for pairs in pairs_data: 
            question, answer = pairs['question'], pairs['answer'] 
            context_pair = {
                'question': f"{pairs['question']}", 
                'answer': pairs['answer']
            }
            instructions.append(context_pair) 
        
        print(f"Processed chunk {key} with {len(pairs_data)} pairs")

print(f"\nTotal instructions created: {len(instructions)}")

with open('dataset/unfiltered.json', 'w') as f:
    json.dump(instructions, f, indent=2)
    
print(f"Saved {len(instructions)} instructions to dataset/unfiltered.json")