import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from agent_webscraper.tools import (
    search_urls, extract_text_from_url, chunk_text, save_chunks,
    check_target_reached, reset_counter, set_llm_instance, chunk_counter,
    extract_topic_and_chunk
)

load_dotenv()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
        max_output_tokens=1200,
    )
    print("✅ LLM initialized")
except Exception as e:
    print(f"❌ LLM error: {e}")
    llm = None

class AgentState(TypedDict):
    user_request: str
    topic: str
    target_chunks: int
    urls: List[str]
    completed: bool

class WebScrapingAgent:
    def __init__(self, model):
        self.model = model
        set_llm_instance(model)

    def generate_urls(self, state: AgentState):
        """Extract topic and chunk count, then generate URLs."""
        user_request = state['user_request']
        
        # Extract topic and chunk count using LLM
        result = extract_topic_and_chunk.invoke({"user_request": user_request})
        
        # Search for URLs using the extracted topic (max 10 URLs)
        urls = search_urls.invoke({"topic": result["topic"], "max_results": 10})
        
        reset_counter.invoke({})
        
        return {
            "topic": result["topic"],
            "target_chunks": result["target_chunks"],
            "urls": urls,
            "completed": False
        }

    def scrape_and_save(self, state: AgentState):
        """Scrape URLs and save chunks (max 10 chunks per URL)."""
        urls = state["urls"]
        user_request = state["user_request"]
        target = state.get('target_chunks', 100)  # max 100 chunks (10 URLs x 10 chunks)
        
        print(f"📊 Target: {target} chunks from {len(urls)} URLs")
        
        for url in urls:
            if check_target_reached.invoke({"target": target}):
                break
            
            print(f"  - Processing: {url}")
            text = extract_text_from_url.invoke({"url": url})
            
            if "Error" in str(text) or len(text.strip()) < 100:
                print(f"  - ⚠️ Failed to extract text")
                continue
            
            chunks = chunk_text.invoke({"text": text})
            if not chunks:
                print(f"  - ⚠️ No chunks generated")
                continue
            
            # Save max 10 chunks per URL
            url_chunk_count = 0
            for chunk in chunks:
                if url_chunk_count >= 10 or check_target_reached.invoke({"target": target}):
                    break
                
                saved = save_chunks.invoke({
                    "chunks": [chunk],  # Pass single chunk
                    "source_url": url,
                    "user_request": user_request,
                    "target_count": target
                })
                
                if saved:  # If chunk was saved (relevant)
                    url_chunk_count += 1
            
            print(f"  - ✅ Saved {url_chunk_count} chunks from this URL")
        
        completed = check_target_reached.invoke({"target": target})
        print(f"\n✅ Complete: {chunk_counter['count']}/{target} chunks")
        
        return {"completed": completed}

# Build graph
agent = WebScrapingAgent(model=llm)

graph = StateGraph(AgentState)
graph.add_node("generate_urls", agent.generate_urls)
graph.add_node("scrape_and_save", agent.scrape_and_save)

graph.set_entry_point("generate_urls")
graph.add_edge("generate_urls", "scrape_and_save")
graph.add_edge("scrape_and_save", END)

web_agent = graph.compile()
