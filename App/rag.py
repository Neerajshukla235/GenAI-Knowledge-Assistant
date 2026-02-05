from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
index = None

CANONICAL_SECTIONS = {
    "introduction": ["introduction", "overview", "preface"],
    "part_a": ["part a", "part - a"],
    "part_b": ["part b", "part - b"],
    "direct_taxes": ["direct tax", "direct taxes"],
    "indirect_taxes": ["indirect tax", "indirect taxes"]
}
def normalize_section(section_name: str) -> str:
    s = section_name.lower()

    # IMPORTANT: indirect before direct
    if "indirect tax" in s:
        return "indirect_taxes"

    if "direct tax" in s:
        return "direct_taxes"

    if "introduction" in s:
        return "introduction"

    if "part" in s and "a" in s:
        return "part_a"

    if "part" in s and "b" in s:
        return "part_b"

    return "general"



def extract_toc_sections(text):
    sections = []
    for line in text.split("\n"):
        line = line.strip()

        # Detect PART headings
        if re.match(r"PART\s*[-–]\s*[A-Z]", line):
            sections.append(line)

        # Detect numbered or titled sections
        elif (
            line.lower().startswith(("first", "second", "third"))
            or "tax" in line.lower()
            or "commission" in line.lower()
        ):
            sections.append(line)

    return sections


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunk_using_toc(text, chunk_size=500):
    sections = extract_toc_sections(text)

    chunks = []
    current_section = "Introduction"
    buffer = ""

    for line in text.split("\n"):
        clean = line.strip()

        # Switch section if TOC match found
        for sec in sections:
            if clean.startswith(sec):
                if buffer:
                    chunks.append({
                        "section": normalize_section(current_section),
                        "content": buffer.strip()
                    })
                    buffer = ""
                current_section = sec

        buffer += clean + " "

        if len(buffer) >= chunk_size:
            chunks.append({
                "section": current_section,
                "content": buffer.strip()
            })
            buffer = ""

    if buffer:
        chunks.append({
            "section": current_section,
            "content": buffer.strip()
        })

    return chunks


def normalize_chunk(c, idx=None):
    return {
        "source_id": c.get("source_id", idx),
        "content": c.get("content"),
        "section": c.get("section", "document")
    }
def index_documents(text):
    global documents, index

    documents = chunk_using_toc(text)

    texts = [d["content"] for d in documents]
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

def retrieve(query, top_k=4):
    if index is None:
        return []

    q = query.lower()
    section_hint = None

    if "part a" in q:
        section_hint = "PART - A"
    elif "part b" in q:
        section_hint = "PART - B"
    elif "direct tax" in q:
        section_hint = "Direct taxes"
    elif "indirect tax" in q:
        section_hint = "Indirect Taxes"

    q_emb = model.encode([query])
    _, idx = index.search(q_emb, top_k * 2)

    results = []
    for i in idx[0]:
        i = int(i)
        chunk = documents[i]

        if section_hint and section_hint.lower() not in chunk["section"].lower():
            continue

        results.append({
            "source_id": i,
            "content": documents[i].get("content") or documents[i].get("text") or "",
            "section": documents[i].get("section", "document")
        })

        if len(results) == top_k:
            break

    return results

def get_all_documents():
    return documents

def get_representative_chunks(n=6):
    if not documents:
        return []

    step = max(1, len(documents) // n)
    reps = []

    for i in range(0, len(documents), step):
        doc = documents[i]

        # 🔑 extract text safely
        text = ""
        if isinstance(doc, dict):
            text = doc.get("content") or doc.get("text") or ""

        reps.append({
            "source_id": i,
            "content": text,
            "section": doc.get("section", "document")
        })

        if len(reps) == n:
            break

    return reps




