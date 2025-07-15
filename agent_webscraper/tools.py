from langchain_core.tools import tool
from bs4 import BeautifulSoup
import json
import re
import os
import requests
from serpapi import GoogleSearch
from dotenv import load_dotenv
from agent_webscraper.prompt import inspection_prompt, extract_chunk_count_and_topic_prompt
import PyPDF2
import docx
import io
load_dotenv()

chunk_counter = {"count": 0}
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
llm_instance = None

def set_llm_instance(llm):
    global llm_instance
    llm_instance = llm

@tool
def extract_topic_and_chunk(user_request: str) -> dict:
    """Extract topic and chunk count from user request using LLM."""
    extraction_prompt = extract_chunk_count_and_topic_prompt(user_request)
    response = llm_instance.invoke(extraction_prompt)
    response_text = response.content.strip()
    
    chunks = None
    topic = None
    
    for line in response_text.split('\n'):
        if line.startswith('CHUNKS:'):
            chunks = int(line.split('CHUNKS:')[1].strip())
        elif line.startswith('TOPIC:'):
            topic = line.split('TOPIC:')[1].strip()
    
    return {"topic": topic, "target_chunks": chunks}

@tool
def search_urls(topic: str, max_results: int = 5) -> list:
    """Search for URLs using SerpAPI with topic from LLM."""
    try:
        params = {"q": topic, 
                  "api_key": SERPAPI_KEY, 
                  "num": max_results,
                  "engine": "google",  # Specify the search engine
                    "google_domain": "google.com",
                    "hl": "en",
                    "gl": "us",
                    "location_requested": "Austin, Texas, United States",
                    "device": "desktop",  # Optional: specify device type
                  }
        search = GoogleSearch(params)
        results = search.get_dict()
        urls = [r["link"] for r in results.get("organic_results", []) if "link" in r][:max_results]
        print(f"Found {len(urls)} URLs for topic: {topic}")
        return urls
    except Exception as e:
        print(f"Search error: {e}")
        return []

@tool
def extract_text_from_url(url: str) -> str:
    """Extract text from URL (HTML, PDF, DOCX)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        if url.lower().endswith('.pdf') or 'pdf' in content_type:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            return text.strip()
        elif url.lower().endswith('.docx') or 'openxmlformats' in content_type:
            doc = docx.Document(io.BytesIO(response.content))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text.strip()
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return re.sub(r"\n{2,}", "\n", text) if len(text.strip()) > 50 else ""
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def chunk_text(text: str, max_chars: int = 2000) -> list:
    """Split text into chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def _is_chunk_relevant(chunk: str, user_request: str) -> dict:
    """Check if chunk is relevant using LLM."""
    if not llm_instance:
        return {"is_relevant": True, "reason": "No LLM available"}
    
    try:
        response = llm_instance.invoke(inspection_prompt(user_request, chunk))
        response_text = response.content.strip()
        is_relevant = "RELEVANT: YES" in response_text.upper()
        reason = response_text.split("REASON:")[1].strip() if "REASON:" in response_text else "No reason"
        
        print(f"    {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT'}: {reason[:80]}...")
        return {"is_relevant": is_relevant, "reason": reason}
    except Exception as e:
        print(f"Quality check failed: {e}")
        return {"is_relevant": False, "reason": f"Failed: {str(e)}"}

@tool
def save_chunks(chunks: list, source_url: str, user_request: str, target_count: int) -> list:
    """Save relevant chunks to files."""
    os.makedirs("./chunks", exist_ok=True)
    saved = []
    
    for chunk in chunks:
        if chunk_counter["count"] >= target_count or len(chunk.strip()) < 100:
            break
        
        # Check relevance
        check = _is_chunk_relevant(chunk, user_request)
        if not check["is_relevant"]:
            print(f"  - Skipping chunk {chunk_counter['count']}: {check['reason'][:50]}...")
            continue
        
        # Save chunk
        chunk_data = {
            "source_file": source_url,
            "chunk_index": chunk_counter["count"],
            "raw_text": chunk,
            "contextualized_text": chunk,
            "metadata": {
                "chunk_size": len(chunk),
                "contextualized_size": len(chunk)
            }
        }
        
        filename = f"chunks/chunk_{chunk_counter['count']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        saved.append(chunk_data)
        chunk_counter["count"] += 1
        print(f"  - ✅ Saved chunk {chunk_counter['count']-1}")
    
    return saved

@tool
def check_target_reached(target: int) -> bool:
    """Check if target chunk count reached."""
    return chunk_counter["count"] >= target

@tool
def reset_counter():
    """Reset chunk counter."""
    chunk_counter["count"] = 0