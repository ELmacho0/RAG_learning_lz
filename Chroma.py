import os
from typing import List
from openai import OpenAI
import chromadb
from chromadb import PersistentClient

# ---------- 配置 ----------
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = "text-embedding-v4"  # 通义千问最新推荐文本向量模型
DIM = 2048  # 可改：2048/1536/.../64，务必在同一集合内保持一致
DATA_DIR = r"./chroma_data"  # 你自己的落盘目录
COLL = "qwen_v4_demo"

if not API_KEY:
    raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")

# OpenAI 兼容客户端（指向 DashScope 兼容端点）
oai = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def embed_texts(texts: List[str], *, model: str = MODEL, dim: int = DIM) -> List[List[float]]:
    """
    调用通义千问 text-embedding-v4 生成 dense 向量（float），可批量。
    """
    # openai>=1.x 接口：dimensions 指定维度，encoding_format="float"
    resp = oai.embeddings.create(
        model=model,
        input=texts,
        dimensions=dim,
        encoding_format="float"
    )
    # 保持与输入顺序一致
    return [item.embedding for item in resp.data]


# ---------- 初始化 Chroma（本地持久化） ----------
client: PersistentClient = chromadb.PersistentClient(path=DATA_DIR)
collection = client.get_or_create_collection(
    name=COLL,
    metadata={"hnsw:space": "cosine", "model": MODEL, "dim": str(DIM)}
)

# ---------- 写入示例 ----------
docs = [
    "向量数据库适合做相似度检索与RAG召回。",
    "我更希望通过通义千问的最新embedding模型来向量化文本。",
    "Chroma 在 Windows 上用 PersistentClient 可以直接落盘。"
]
ids = ["d1", "d2", "d3"]
metas = [{"lang": "zh"}, {"lang": "zh"}, {"lang": "zh"}]

embs = embed_texts(docs)  # 先生成向量
collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

# ---------- 查询示例（用向量查询，避免依赖内置embedding函数） ----------
query = "怎么用千问做文本向量化并存储到Chroma？"
qvec = embed_texts([query])[0]
res = collection.query(
    query_embeddings=[qvec],
    n_results=3,
    include=["ids", "documents", "metadatas", "distances"]
)

print(res)
# 提示：cosine 距离越小越相似
