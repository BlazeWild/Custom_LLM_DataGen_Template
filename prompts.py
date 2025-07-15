"""
Centralized prompts for the Custom LLM Dataset Generation Template
Contains all prompts used for data generation and quality checking
"""

def generation_prompt_template(data: str, num_records: int = 15, domain: str = "your domain", domain_description: str = "domain-specific information"):
    """
    Generate Q&A pairs from data chunks for any domain
    """
    return f"""You are an expert data curator creating a high-quality instruction tuning dataset for a {domain} assistant.

Generate {num_records} Q&A pairs from this data chunk. The exact number should depend on the richness of the content - generate more pairs for information-dense chunks and fewer for sparse content.

**DISTRIBUTION:**
- **85% Domain-specific Q&A pairs** - Detailed answers about {domain_description}, concepts, procedures, best practices, etc.
- **10% Related but tangential** - Adjacent topics, general concepts, or broader context related to {domain}
- **5% Negative examples** - Refuse to answer completely unrelated topics (cooking, sports, entertainment, etc. - topics outside your domain)

**ANSWER FORMATTING REQUIREMENTS FOR POSITIVE ANSWERS:**
- **MUST use extensive markdown formatting**:
  - **Bold text** for emphasis on important terms, costs, names, key concepts
  - ## Main headings for major sections
  - ### Sub-headings for subsections
  - - Bullet points for lists and key points
  - **Bold bullet points** like - **Important Point**: details
  - Use <br> for line breaks (NEVER use \\n or actual newlines)
- **Make answers very detailed and comprehensive**:
  - Minimum 5-8 sentences for domain-specific answers
  - Include specific details from the provided data
  - Add context and background information
  - Provide practical advice and actionable insights
  - Use proper structure with headings and bullet points

**ANSWER FORMATTING FOR NEGATIVE ANSWERS:**
- Keep simple and brief
- Politely refuse and redirect to {domain} topics
- No markdown formatting needed for refusals

**OUTPUT FORMAT:**
Return ONLY a JSON array:
[
  {{
    "question": "What are the key principles of [domain concept]?",
    "answer": "The **key principles** of [domain concept] include several important aspects:<br><br>## **Core Principles**<br><br>- **Principle 1**: Detailed explanation with specific examples from the data<br>- **Principle 2**: Comprehensive coverage with practical applications<br>- **Principle 3**: Important considerations and best practices<br><br>**Additional considerations** include proper implementation, common pitfalls to avoid, and recommended approaches for success. These principles form the foundation for effective [domain-specific activity] and should be carefully considered in any implementation."
  }},
  {{
    "question": "How does [domain process] work in practice?",
    "answer": "**[Domain process]** works through a systematic approach with several key stages:<br><br>### **Process Overview**<br><br>- **Stage 1**: Initial assessment and preparation<br>- **Stage 2**: Implementation with specific methodologies<br>- **Stage 3**: Monitoring and optimization<br><br>The process typically takes **[timeframe]** and requires **[resources/tools]**. Success depends on **proper planning**, **adequate resources**, and **continuous monitoring** throughout the implementation phase."
  }},
  {{
    "question": "What's the best recipe for chocolate cake?",
    "answer": "I'm sorry, I can only provide information related to {domain}. I'd be happy to help with questions about {domain_description}, concepts, procedures, or best practices in this field."
  }}
]

---

Data: {data}
"""





def quality_check_prompt_template(records_batch, domain_config):
    """
    Evaluate Q&A pairs quality for any domain
    """
    batch_text = "\n\n".join([f"Record {i+1}: {record}" for i, record in enumerate(records_batch)])
    
    return f"""Rate these {len(records_batch)} Q&A records for a {domain_config['domain_name']} dataset on accuracy (1-10) and style (1-10).

**SCORING RULES:**

**HIGH SCORES (7-10) for ACCURACY:**
- {domain_config['positive_example']} = 7-10
- Non-domain question + Proper rejection response ("{domain_config['rejection_template']}") = 7-10

**LOW SCORES (1-6) for ACCURACY:**
- Domain question + Rejection response = 1-6
- Non-domain question + Actual answer (not rejection) = 1-6

**STYLE SCORING (1-10):**
- 1-2: Harmful, inappropriate, or very poorly written
- 3-4: Poor formatting, unclear language
- 5-6: Basic but adequate
- 7-10: Well-written, clear, and professional

**IMPORTANT: Use ONLY double quotes in JSON. Never use single quotes.**

**EXAMPLE OUTPUT FORMAT:**
[
  {{
    "question": "What is the weather like in [unrelated topic]?",
    "answer": "I'm sorry, I can only provide information related to {domain_config['domain_name']}. I'd be happy to help with questions about {domain_config['domain_description']}.",
    "quality": {{
      "accuracy": {{
        "score": 9,
        "explanation": "The question is about [unrelated topic], not {domain_config['domain_name']}. The answer is a proper rejection response, stating the dataset's focus on {domain_config['domain_name']}. This demonstrates good understanding and adherence to the dataset's scope."
      }},
      "style": {{
        "score": 8,
        "explanation": "The response is well-written, clear, and professional. It politely declines to answer the question and offers alternative assistance within the dataset's domain."
      }}
    }}
  }},
  {{
    "question": "What are the key concepts in {domain_config['domain_name']}?",
    "answer": "The key concepts in {domain_config['domain_name']} include various important principles and practices...",
    "quality": {{
      "accuracy": {{
        "score": 9,
        "explanation": "The question is directly relevant to {domain_config['domain_name']}. The answer provides comprehensive and accurate information about key concepts in the domain. Therefore, it deserves a high accuracy score."
      }},
      "style": {{
        "score": 8,
        "explanation": "The answer is well-written, clear, and professional. The language is appropriate for the context, and the information is presented in a logical and easy-to-understand manner."
      }}
    }}
  }}
]

{batch_text}

Return ONLY the JSON array in the exact format shown above, with {len(records_batch)} objects. Use ONLY double quotes, never single quotes."""