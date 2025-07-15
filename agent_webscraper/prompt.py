import re

def extract_chunk_count_and_topic_prompt(user_message: str) -> str:
    """Generate a prompt for LLM to extract chunk count and topic from user message."""
    return f"""Extract the number of chunks requested and the topic from this user message:

USER MESSAGE: {user_message}

INSTRUCTIONS:
- If the user specifies a number of chunks (like "make 25 chunks", "generate 30 chunks"), extract that number
- If no number is specified, use 50 as default
- Extract the main topic they want chunks about
- The topic should be concise, specific and detailed - not just a general subject

RESPONSE FORMAT:
CHUNKS: [number]
TOPIC: [topic]

Examples:
User: "make 25 chunks about machine learning algorithms"
CHUNKS: 25
TOPIC: machine learning algorithms

User: "generate content about sustainable gardening practices"
CHUNKS: 50
TOPIC: sustainable gardening practices

User: "create 15 chunks for cybersecurity best practices"
CHUNKS: 15
TOPIC: cybersecurity best practices"""


def inspection_prompt(user_prompt: str, chunk_text: str):
    """Generate a prompt for LLM-based chunk quality inspection."""
    return f"""You are a strict content quality inspector. Your job is to determine if this text chunk contains USEFUL, SPECIFIC information related to the user's request.

USER'S REQUEST: {user_prompt}

TEXT CHUNK TO EVALUATE:
{chunk_text}

STRICT EVALUATION CRITERIA:
✅ ACCEPT ONLY IF the chunk contains:
- Specific facts, data, or detailed information about the requested topic
- Practical tips, guides, or actionable advice
- Technical details or explanations



❌ REJECT IF the chunk contains:
- Generic website navigation, menus, or headers
- Outdated inforamtion 
- Copyright notices, disclaimers, or legal text  
- Advertisements or promotional content
- Social media sharing buttons or widgets
- "Contact us" or "About us" boilerplate text
- Completely unrelated topics
- Vague or generic statements without substance
- Less than 2 sentences of actual content

BE STRICT. When in doubt, choose NO.

RESPONSE FORMAT (use exactly this format):
RELEVANT: YES
REASON: Brief explanation

OR

RELEVANT: NO  
REASON: Brief explanation"""