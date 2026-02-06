from .hf_client import generate_with_llm
import re

def clean_sentence_boundary(text: str, max_len=700) -> str:
    text = text.strip()

    if len(text) <= max_len:
        return text

    truncated = text[:max_len]

    # Find last sentence-ending punctuation
    match = re.search(r'[.!?](?!.*[.!?])', truncated)
    if match:
        return truncated[:match.end()]

    return truncated

def clean_artifacts(text: str) -> str:
    # remove standalone numbers like "32." or "27."
    text = re.sub(r'\n?\s*\d+\.\s*', ' ', text)

    # fix spaced hyphens
    text = re.sub(r'\s-\s', '-', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
def fallback_answer(retrieved_chunks):
    texts = []

    for c in retrieved_chunks:
        content = c.get("content")

        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, dict):
            # common nested case
            if "text" in content:
                texts.append(content["text"])
        # else: ignore safely

    if not texts:
        return "No relevant information found in the document."

    text = " ".join(texts)
    raw_text = text.replace("\n", " ")
    raw_text = clean_artifacts(raw_text)
    polished = clean_sentence_boundary(raw_text, max_len=700)

    return (
        "Based on the uploaded document, the key points are:\n\n"
        + polished[:900] + "..."
    )



def generate_answer(question: str, retrieved_chunks: list):
    print(">>> generate_answer() CALLED")

    if not retrieved_chunks:
        return "No relevant information found in the document."

    context = "\n\n".join(
        f"- {c['content']}" for c in retrieved_chunks if c.get("content")
    )

    prompt = f"""
Answer the question strictly using the context below.

Context:
{context}

Question:
{question}

Instructions:
- Be concise
- Do not add external knowledge
"""

    print(">>> Calling HF LLM now")
    answer = generate_with_llm(prompt)

    if answer:
        print(">>> HF ANSWER RECEIVED")
        return answer

    print(">>> HF FAILED, USING FALLBACK")
    return fallback_answer(retrieved_chunks)
