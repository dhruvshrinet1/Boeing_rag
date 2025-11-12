
from langchain_core.prompts import ChatPromptTemplate

# Prompt for answering questions based on retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are reading from a Boeing 737 training manual for study purposes. Answer the question using ONLY the manual content below. Write directly as if telling another pilot - no phrases like "based on the manual".

Manual Content:
{context}

Question: {input}

Instructions:
- Look up the specific value or procedure in the manual content
- State it directly (example: "The Climb Limit Weight is 52,200 kg")
- For procedures, give clear steps (example: "Call GEAR UP")
- Always cite the page(s) where you found the MAIN answer: (Page 43)
- Only cite multiple pages if the core answer actually spans across them
- Don't cite pages with just supporting or related information
- Be concise and factual"""),
])

# Prompt for query expansion with aviation terms
query_expansion_prompt = ChatPromptTemplate.from_template("""
You are an aviation technical expert. Given a user query about Boeing 737 operations,
expand it to include relevant technical terms, synonyms, and related concepts that might
appear in a technical manual.

Original query: {query}

Generate 2-3 expanded versions of this query that:
1. Include technical aviation terminology
2. Use synonyms for key concepts
3. Rephrase to match manual language

Return ONLY the expanded queries, one per line, without numbering or explanation.
Keep each expansion concise (max 20 words).
""")

# Prompt for describing images from technical manual
image_description_prompt = """Describe this image from a Boeing 737 technical manual.
Focus on technical details, numbers, labels, and any text visible in the image.
Be concise and specific. If it's a diagram, describe what it shows."""

PROMPT_REGISTRY = {
    "context_qa": qa_prompt,
    "query_expansion": query_expansion_prompt,
    "image_description": image_description_prompt,
}