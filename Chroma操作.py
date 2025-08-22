import chromadb
from chromadb import PersistentClient

DATA_DIR = r"./chroma_data"
client = PersistentClient(path=DATA_DIR)

# 2.1 列出现有集合
for c in client.list_collections():  # 按创建时间排序
    print(c.name, c.count())

# 2.2 重新拿到同名集合（元数据不必完全相同也能拿到）
coll = client.get_or_create_collection(
    name="qwen_v4_demo",
)
print("count =", coll.count())

# 2.3 读取与分页
page1 = coll.get(limit=2, offset=0, include=["documents","metadatas"])
page2 = coll.get(limit=2, offset=2, include=["documents","metadatas"])

# 2.4 需要时把 embeddings 一起取回（默认不会返回）
full = coll.get(include=["embeddings","documents","metadatas"])
for i, (id_val, doc) in enumerate(zip(full['ids'], full['documents'])):
    print(f"{i+1}. ID: {id_val}")
    print(f"   Document: {doc}")
