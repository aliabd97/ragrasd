"""
🔍 اختبار نظام RAG - نسخة محسّنة
================================
"""

from sentence_transformers import SentenceTransformer
import chromadb

def main():
    try:
        # 1. تحميل النموذج
        print("🔄 تحميل النموذج...")
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("✅ تم تحميل النموذج بنجاح")

        # 2. الاتصال بقاعدة البيانات
        print("\n🔄 الاتصال بقاعدة البيانات...")
        client = chromadb.PersistentClient(path="data/database/chroma_db")

        # عرض Collections المتاحة
        collections = client.list_collections()
        print(f"✅ عدد Collections المتاحة: {len(collections)}")

        for col in collections:
            print(f"   - {col.name} ({col.count()} عنصر)")

        # الاتصال بـ Collection
        collection_name = "islamic_books_e5"
        collection = client.get_collection(collection_name)
        print(f"✅ تم الاتصال بـ Collection: {collection_name}")

        # 3. اختبار البحث
        test_queries = [
            "من هو الشريف المرتضى؟",
            "ما هو موضوع كتاب الشافي في الإمامة؟",
            "اشرح مفهوم الإمامة"
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"🔍 السؤال: {query}")
            print(f"{'='*60}\n")

            # تحويل السؤال لـ vector (مع بادئة query: حسب متطلبات E5)
            query_embedding = model.encode(f"query: {query}")

            # البحث
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3  # أفضل 3 نتائج فقط
            )

            # عرض النتائج
            if results['ids'][0]:
                print("📋 النتائج:\n")
                for i, (id, metadata, doc, distance) in enumerate(zip(
                    results['ids'][0],
                    results['metadatas'][0],
                    results['documents'][0],
                    results['distances'][0]
                ), 1):
                    print(f"{i}. [{metadata['type']}] {id}")
                    print(f"   📊 Distance: {distance:.4f}")
                    if 'title' in metadata:
                        print(f"   📖 الكتاب: {metadata['title']}")
                    print(f"   📝 {doc[:150]}...")
                    print()
            else:
                print("⚠️  لم يتم العثور على نتائج")

        print(f"\n{'='*60}")
        print("✅ اكتمل الاختبار بنجاح!")
        print(f"{'='*60}")

    except chromadb.errors.NotFoundError as e:
        print(f"\n❌ خطأ: Collection غير موجودة")
        print(f"   {str(e)}")
        print("\n💡 الحل:")
        print("   1. تأكد من تشغيل step3_embeddings_E5.py أولاً")
        print("   2. أو تحقق من اسم Collection في السكريبت")

    except ImportError as e:
        print(f"\n❌ خطأ: مكتبة غير مثبتة")
        print(f"   {str(e)}")
        print("\n💡 الحل:")
        print("   pip install -r requirements.txt")

    except Exception as e:
        print(f"\n❌ خطأ غير متوقع:")
        print(f"   {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    main()
