import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
import fitz  # PyMuPDF

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai optional for compile
    OpenAI = None

# ---------- Paths ----------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------- OpenAI/Qwen Client ----------
def get_client() -> "OpenAI":
    """Return an OpenAI-compatible client for Qwen models."""
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=base_url)


# ---------- Text Extraction ----------
def extract_text(path: Path) -> str:
    """Extract UTF-8 text from a file. Supports PDF and plain text."""
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    # fallback: assume UTF-8 text
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------- Chunking ----------
def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Simple text chunking with overlap."""
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ---------- Embeddings ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_client()
    resp = client.embeddings.create(
        model="text-embedding-v4", input=texts, dimensions=2048, encoding_format="float"
    )
    return [item.embedding for item in resp.data]


# ---------- Chroma helpers ----------
def get_collection(user: str):
    """Get or create a Chroma collection for the user."""
    path = DATA_DIR / user / "index"
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    coll = client.get_or_create_collection(
        name=f"kb_{user}", metadata={"hnsw:space": "cosine"}
    )
    return coll


# ---------- Indexing ----------
def index_file(user: str, file_path: Path) -> str:
    """Parse, chunk, embed and store a document. Returns doc_id."""
    data = file_path.read_bytes()
    doc_id = hashlib.sha256(data).hexdigest()
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return doc_id
    embeddings = embed_texts(chunks)
    ids = [f"{doc_id}#c{i}" for i in range(len(chunks))]
    metas = [{"doc_id": doc_id, "filename": file_path.name, "chunk_index": i} for i in range(len(chunks))]
    coll = get_collection(user)
    coll.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    return doc_id


# ---------- Query ----------
def search(user: str, query: str, top_k: int = 4) -> List[Tuple[str, dict]]:
    coll = get_collection(user)
    qvec = embed_texts([query])[0]
    res = coll.query(
        query_embeddings=[qvec], n_results=top_k, include=["documents", "metadatas"]
    )
    documents = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(documents, metas))


# ---------- Chat ----------
def chat_with_docs(user: str, query: str) -> str:
    """Retrieve docs and ask Qwen for an answer."""
    client = get_client()
    hits = search(user, query)
    context = "\n\n".join(
        f"[{h[1].get('filename')}] {h[0][:400]}" for h in hits
    )
    system_prompt = (
        "你是企业内部知识库助手。你必须仅根据提供的证据块回答；若证据不足，请说明。"
    )
    user_prompt = f"问题：{query}\n\n证据：\n{context}" if context else query
    resp = client.chat.completions.create(
        model="qwen-turbo", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    return resp.choices[0].message.content
