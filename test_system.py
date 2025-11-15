from sentence_transformers import SentenceTransformer
import chromadb

# 1. تحميل النموذج
print("🔄 تحميل النموذج...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 2. الاتصال بقاعدة البيانات
print("🔄 الاتصال بقاعدة البيانات...")
client = chromadb.PersistentClient(path="data/database/chroma_db")
collection = client.get_collection("islamic_books_e5")

# 3. اختبار البحث
query = "من هو الشريف المرتضى؟"
print(f"\n🔍 السؤال: {query}\n")

# تحويل السؤال لـ vector
query_embedding = model.encode(f"query: {query}")

# البحث
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

# عرض النتائج
print("📋 النتائج:\n")
for i, (id, metadata, doc) in enumerate(zip(
    results['ids'][0], 
    results['metadatas'][0],
    results['documents'][0]
), 1):
    print(f"{i}. {metadata['type']}: {id}")
    print(f"   {doc[:200]}...")
    print()