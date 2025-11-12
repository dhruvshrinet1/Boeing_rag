
from langchain_core.prompts import ChatPromptTemplate

# Prompt for answering questions based on retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ("human", """Use the manual content below to answer the question. Write your answer as if you're telling another pilot directly - no phrases like "based on the manual" or "according to". Just state the facts, values, and procedures clearly and concisely.

Manual Content:
{context}

Question: {input}

Answer format:
- State specific values directly (e.g., "The Climb Limit Weight is 52,200 kg")
- Give clear actions (e.g., "Call GEAR UP")
- Be concise - pilot-to-pilot style
- Always cite pages: (Page 43) or (Pages 39 and 51)"""),
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