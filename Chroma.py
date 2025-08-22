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
    "中医的某些治愈案例当前的现代医学还无法理解。",
    "AIagent是人工智能下一个阶段的发展方向",
    "如何调用通义千问的API接口使用其embidding模型然后将转化后的向量存入本地向量数据库",
    "可以通过OpenAI提供的通用接口使用千问的嵌入模型，使用Python创建本地chroma数据库并导入转化后的向量"
]
ids = ["d1", "d2", "d3", "d4"]
metas = [{"lang": "zh"}, {"lang": "zh"}, {"lang": "zh"}, {"lang": "zh"}]

embs = embed_texts(docs)  # 先生成向量
collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

# ---------- 查询示例（用向量查询，避免依赖内置embedding函数） ----------
query = "怎么用千问做文本向量化并存储到Chroma？"
qvec = embed_texts([query])[0]
res = collection.query(
    query_embeddings=[qvec],      # 你用的是自算向量，就保持用 query_embeddings
    n_results=8,
    include=["documents", "metadatas", "distances"]
)

print(res.keys())        # 看看有哪些键：通常含 'ids'、以及你 include 的字段
print(res["ids"])        # 直接取 id
print(res["documents"])  # 取文档
print(res["distances"])  # 距离：越小越相似

print(res)
# 提示：cosine 距离越小越相似
