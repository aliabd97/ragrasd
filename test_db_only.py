"""
🔍 اختبار قاعدة البيانات فقط (بدون النموذج)
==========================================
"""

import chromadb
import json

def main():
    try:
        # الاتصال بقاعدة البيانات
        print("🔄 الاتصال بقاعدة البيانات...")
        client = chromadb.PersistentClient(path="data/database/chroma_db")

        # عرض Collections المتاحة
        collections = client.list_collections()
        print(f"\n✅ عدد Collections المتاحة: {len(collections)}\n")

        for col in collections:
            count = col.count()
            print(f"📦 Collection: {col.name}")
            print(f"   عدد العناصر: {count}")
            print()

            # عرض عينة من البيانات
            if count > 0:
                sample = col.get(limit=3)
                print("   🔍 عينة من البيانات:")
                for i, (id, metadata, doc) in enumerate(zip(
                    sample['ids'],
                    sample['metadatas'],
                    sample['documents']
                ), 1):
                    print(f"\n   {i}. ID: {id}")
                    print(f"      Type: {metadata.get('type', 'N/A')}")
                    if 'title' in metadata:
                        print(f"      Title: {metadata['title']}")
                    print(f"      Text: {doc[:100]}...")

        # قراءة الإحصائيات
        print(f"\n{'='*60}")
        print("📊 إحصائيات قاعدة البيانات:")
        print(f"{'='*60}")

        try:
            with open('data/database/embeddings_stats.json', 'r', encoding='utf-8') as f:
                stats = json.load(f)
                print(f"\nالنموذج: {stats.get('model', 'N/A')}")
                print(f"أبعاد Embedding: {stats.get('embedding_dimension', 'N/A')}")
                print(f"إجمالي العناصر: {stats.get('total_items', 'N/A')}")
                print(f"الأداء: {stats.get('performance', 'N/A')}")
                print(f"الوقت: {stats.get('timestamp', 'N/A')}")
        except FileNotFoundError:
            print("⚠️  ملف الإحصائيات غير موجود")

        print(f"\n{'='*60}")
        print("✅ قاعدة البيانات تعمل بنجاح!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ خطأ: {type(e).__name__}")
        print(f"   {str(e)}\n")

if __name__ == "__main__":
    main()
