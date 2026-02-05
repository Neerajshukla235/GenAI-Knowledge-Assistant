from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from .llm import generate_answer
from PyPDF2 import PdfReader
from .rag import retrieve, get_representative_chunks,index_documents,CANONICAL_SECTIONS,get_all_documents


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for now
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = ""

    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            content += page.extract_text()
    else:
        content = (await file.read()).decode("utf-8")

    index_documents(content)
    return {"message": "Document indexed successfully"}


def is_document_level_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "context of document",
        "context of the document",
        "summarize the document",
        "summary of document",
        "overview",
        "what is this document about"
    ]
    return any(k in q for k in keywords)


SUMMARY_KEYWORDS = [
    "summary", "context", "overview",
    "what is this document about",
    "high level", "gist"
]

def is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in SUMMARY_KEYWORDS)

def detect_section_query(question: str):
    q = question.lower()

    if "indirect tax" in q:
        return "indirect_taxes"

    if "direct tax" in q:
        return "direct_taxes"

    if "introduction" in q:
        return "introduction"

    if "part a" in q:
        return "part_a"

    if "part b" in q:
        return "part_b"

    return None
@app.post("/ask")
def ask(req: QuestionRequest):

    question = req.question.strip()

    # -------- 1. Decide retrieval strategy --------

    if is_document_level_question(question):
        # Document-level summary → representative chunks
        retrieved = get_representative_chunks()

    else:
        section_key = detect_section_query(question)

        if section_key:
            # Section-level deterministic retrieval
            all_chunks = get_all_documents()
            retrieved = []

            for i, c in enumerate(all_chunks):
                if c.get("section") == section_key:
                    retrieved.append({
                        "source_id": i,
                        "content": c.get("content"),
                        "section": c.get("section")
                    })

                if len(retrieved) == 5:
                    break
        else:
            # Semantic retrieval (FAISS)
            retrieved = retrieve(question)

    # -------- 2. Normalize retrieved chunks --------

    normalized_chunks = []

    for i, r in enumerate(retrieved):
        content = r.get("content", "")

        # Handle nested content
        if isinstance(content, dict):
            content = content.get("text", "")

        if not isinstance(content, str):
            content = ""

        normalized_chunks.append({
            "source_id": r.get("source_id", i),
            "content": content,
            "section": r.get("section", "document")
        })

    if not normalized_chunks:
        return {
            "question": question,
            "answer": "No relevant information found in the document.",
            "sources": []
        }

    # -------- 3. Generate answer (LLM + fallback) --------

    answer = generate_answer(question, normalized_chunks)

    # -------- 4. Build sources safely --------

    sources = []

    for c in normalized_chunks:
        excerpt = c["content"][:200].replace("\n", " ")

        sources.append({
            "section": c["section"],
            "excerpt": excerpt
        })

    # -------- 5. Final response --------

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
