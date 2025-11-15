import chromadb

# الاتصال بقاعدة البيانات
client = chromadb.PersistentClient(path="data/database/chroma_db")

# عرض جميع Collections
collections = client.list_collections()
print(f"عدد Collections: {len(collections)}")
for col in collections:
    print(f"  - {col.name} (عدد العناصر: {col.count()})")
