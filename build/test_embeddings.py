#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Embeddings and ChromaDB
=============================

اختبار شامل لقاعدة البيانات والبحث
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import yaml


# تحميل الإعدادات
def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()

CHROMA_DB_PATH = Path(config['paths']['chroma_db'])
MODEL_NAME = config['embeddings']['model']


# =============================================================================
# الاختبارات
# =============================================================================

def test_database_exists():
    """اختبار 1: هل قاعدة البيانات موجودة؟"""
    print("🧪 اختبار 1: وجود قاعدة البيانات")
    assert CHROMA_DB_PATH.exists(), f"❌ قاعدة البيانات غير موجودة: {CHROMA_DB_PATH}"
    print("✅ قاعدة البيانات موجودة")


def test_collection_exists():
    """اختبار 2: هل collection موجود؟"""
    print("\n🧪 اختبار 2: وجود collection")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection("islamic_books")
    assert collection is not None, "❌ Collection غير موجود"
    print("✅ Collection موجود")
    return collection


def test_items_count(collection):
    """اختبار 3: عدد العناصر"""
    print("\n🧪 اختبار 3: عدد العناصر")
    count = collection.count()
    print(f"📊 إجمالي العناصر: {count}")
    
    # يجب أن يكون حوالي 753 (4 + 315 + 434)
    assert 700 <= count <= 800, f"❌ عدد العناصر غير متوقع: {count}"
    print("✅ عدد العناصر صحيح")
    return count


def test_metadata_types(collection):
    """اختبار 4: أنواع البيانات"""
    print("\n🧪 اختبار 4: أنواع البيانات")
    
    all_metadata = collection.get()['metadatas']
    
    docs = [m for m in all_metadata if m.get('type') == 'document']
    secs = [m for m in all_metadata if m.get('type') == 'section']
    paras = [m for m in all_metadata if m.get('type') == 'paragraph']
    
    print(f"📊 Documents: {len(docs)}")
    print(f"📊 Sections: {len(secs)}")
    print(f"📊 Paragraphs: {len(paras)}")
    
    assert len(docs) == 4, f"❌ عدد Documents خطأ: {len(docs)}"
    assert 300 <= len(secs) <= 350, f"❌ عدد Sections خطأ: {len(secs)}"
    assert 400 <= len(paras) <= 500, f"❌ عدد Paragraphs خطأ: {len(paras)}"
    
    print("✅ أنواع البيانات صحيحة")


def test_search_functionality(collection):
    """اختبار 5: وظيفة البحث"""
    print("\n🧪 اختبار 5: وظيفة البحث")
    
    # تحميل النموذج
    model = SentenceTransformer(MODEL_NAME)
    
    # اختبارات بحث مختلفة
    test_queries = [
        "الإمامة",
        "الشريف المرتضى",
        "النص على الإمام"
    ]
    
    for query in test_queries:
        print(f"\n🔍 البحث عن: '{query}'")
        
        # توليد embedding للاستعلام
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # البحث
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )
        
        assert len(results['ids'][0]) > 0, f"❌ لا توجد نتائج للاستعلام: {query}"
        
        print(f"   ✅ وجد {len(results['ids'][0])} نتائج")
        
        # عرض أول نتيجة
        top_id = results['ids'][0][0]
        top_metadata = results['metadatas'][0][0]
        print(f"   🏆 أفضل نتيجة: {top_id} ({top_metadata['type']})")
    
    print("\n✅ البحث يعمل بشكل صحيح")


def test_multilevel_search(collection):
    """اختبار 6: البحث متعدد المستويات"""
    print("\n🧪 اختبار 6: البحث متعدد المستويات")
    
    model = SentenceTransformer(MODEL_NAME)
    query = "ما هي الإمامة؟"
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    # البحث في كل مستوى على حدة
    for level in ['document', 'section', 'paragraph']:
        print(f"\n   🔍 البحث في {level}s...")
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            where={"type": level}
        )
        
        count = len(results['ids'][0])
        print(f"      ✅ وجد {count} نتائج")
        
        assert count > 0, f"❌ لا توجد نتائج في {level}s"
    
    print("\n✅ البحث متعدد المستويات يعمل")


def test_embeddings_quality(collection):
    """اختبار 7: جودة Embeddings"""
    print("\n🧪 اختبار 7: جودة Embeddings")
    
    model = SentenceTransformer(MODEL_NAME)
    
    # استعلامات متشابهة يجب أن تعطي نتائج متشابهة
    similar_queries = [
        "الإمامة بالنص",
        "النص على الإمام"
    ]
    
    results_1 = collection.query(
        query_embeddings=[model.encode([similar_queries[0]], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()],
        n_results=5
    )
    
    results_2 = collection.query(
        query_embeddings=[model.encode([similar_queries[1]], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()],
        n_results=5
    )
    
    # يجب أن يكون هناك تداخل في النتائج
    ids_1 = set(results_1['ids'][0])
    ids_2 = set(results_2['ids'][0])
    overlap = len(ids_1 & ids_2)
    
    print(f"   📊 التداخل في النتائج: {overlap}/5")
    assert overlap >= 1, "❌ لا يوجد تداخل في النتائج المتشابهة"
    
    print("✅ جودة Embeddings جيدة")


# =============================================================================
# التشغيل
# =============================================================================

def main():
    """تشغيل كل الاختبارات"""
    
    print("=" * 70)
    print("🧪 اختبارات Embeddings و ChromaDB")
    print("=" * 70)
    
    try:
        # الاختبارات الأساسية
        test_database_exists()
        collection = test_collection_exists()
        test_items_count(collection)
        test_metadata_types(collection)
        
        # اختبارات البحث
        test_search_functionality(collection)
        test_multilevel_search(collection)
        test_embeddings_quality(collection)
        
        # النتيجة النهائية
        print("\n" + "=" * 70)
        print("✅ نجحت جميع الاختبارات!")
        print("=" * 70)
        print("\n🎉 قاعدة البيانات جاهزة للاستخدام!")
        
    except AssertionError as e:
        print(f"\n❌ فشل الاختبار: {e}")
        return False
    except Exception as e:
        print(f"\n❌ خطأ: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
