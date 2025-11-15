#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Generate Embeddings and Build Vector Database (E5 Version)
===================================================================

المهمة:
- تحميل البيانات من Step 2
- توليد embeddings باستخدام multilingual-e5-large (أقوى نموذج للعربية)
- بناء ChromaDB vector database
- اختبار البحث

النموذج: intfloat/multilingual-e5-large
- الحجم: 560M parameters
- الأبعاد: 1024 (أكبر من paraphrase)
- اللغات: 100 لغة
- الجودة: state-of-the-art للعربية
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml


# =============================================================================
# تحميل الإعدادات
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """تحميل ملف الإعدادات"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()


# =============================================================================
# الإعدادات من config.yaml
# =============================================================================

# Paths
PROCESSED_DIR = Path(config['paths']['processed_data'])
DATABASE_DIR = Path(config['paths']['database'])
CHROMA_DB_PATH = Path(config['paths']['chroma_db'])

# Embeddings - E5 Model
MODEL_NAME = config['embeddings']['model']  # intfloat/multilingual-e5-large
EMBEDDING_DIM = config['embeddings']['dimension']  # 1024
BATCH_SIZE = config['embeddings']['batch_size']
DEVICE = config['embeddings']['device']
SHOW_PROGRESS = config['embeddings']['show_progress']
CACHE_FOLDER = Path(config['embeddings']['cache_folder'])

# Files
DOCUMENTS_FILE = PROCESSED_DIR / "documents.json"
SECTIONS_FILE = PROCESSED_DIR / "sections.json"
PARAGRAPHS_FILE = PROCESSED_DIR / "paragraphs.json"
STATS_FILE = DATABASE_DIR / "embeddings_stats.json"


# =============================================================================
# Class: EmbeddingsGenerator (E5 Version)
# =============================================================================

class EmbeddingsGenerator:
    """
    مولد Embeddings باستخدام multilingual-e5-large
    
    ملاحظة: E5 يحتاج prefix للنصوص:
    - "query: " للاستعلامات
    - "passage: " للنصوص المخزنة
    """
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """
        التهيئة
        
        Args:
            model_name: اسم النموذج (intfloat/multilingual-e5-large)
            device: الجهاز (cpu أو cuda)
        """
        print(f"📥 تحميل نموذج Embeddings: {model_name}")
        print(f"⚙️ الجهاز: {device}")
        print(f"ℹ️ هذا أقوى نموذج للعربية - Microsoft E5")
        print(f"ℹ️ الأبعاد: 1024 (أكبر من paraphrase-768)")
        
        # إنشاء مجلد cache
        CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
        
        # تحميل النموذج
        start_time = time.time()
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(CACHE_FOLDER)
        )
        load_time = time.time() - start_time
        
        print(f"✅ تم التحميل في {load_time:.2f} ثانية")
        print(f"📊 الأبعاد الفعلية: {self.model.get_sentence_embedding_dimension()}")
        print()
    
    def encode_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True,
        prefix: str = "passage"
    ) -> List[List[float]]:
        """
        تحويل نصوص إلى embeddings
        
        Args:
            texts: قائمة النصوص
            show_progress: إظهار progress bar
            prefix: البادئة ("passage" أو "query")
        
        Returns:
            قائمة embeddings
        """
        # إضافة prefix حسب نوع النص
        # للنصوص المخزنة نستخدم "passage: "
        prefixed_texts = [f"{prefix}: {text}" for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        return embeddings.tolist()


# =============================================================================
# Class: ChromaDBBuilder
# =============================================================================

class ChromaDBBuilder:
    """بناء ChromaDB vector database"""
    
    def __init__(self, db_path: Path = CHROMA_DB_PATH):
        """
        التهيئة
        
        Args:
            db_path: مسار قاعدة البيانات
        """
        self.db_path = db_path
        
        # إنشاء المجلد
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # إنشاء/فتح قاعدة البيانات
        print(f"📂 فتح ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collection name
        self.collection_name = "islamic_books_e5"
        
        # حذف collection القديم إن وجد
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ تم حذف collection القديم")
        except:
            pass
        
        # إنشاء collection جديد
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "Multi-level RAG for Islamic books using E5",
                "model": "intfloat/multilingual-e5-large",
                "dimension": 1024
            }
        )
        
        print(f"✅ تم إنشاء collection: {self.collection_name}")
    
    def add_items(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        إضافة عناصر إلى ChromaDB
        
        Args:
            ids: معرفات فريدة
            embeddings: embeddings vectors
            documents: النصوص الأصلية
            metadatas: metadata لكل عنصر
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def get_stats(self) -> Dict:
        """الحصول على إحصائيات قاعدة البيانات"""
        count = self.collection.count()
        
        # عد كل نوع
        all_data = self.collection.get()
        docs_count = len([m for m in all_data['metadatas'] if m.get('type') == 'document'])
        secs_count = len([m for m in all_data['metadatas'] if m.get('type') == 'section'])
        paras_count = len([m for m in all_data['metadatas'] if m.get('type') == 'paragraph'])
        
        return {
            "total_items": count,
            "documents": docs_count,
            "sections": secs_count,
            "paragraphs": paras_count
        }


# =============================================================================
# دوال المساعدة
# =============================================================================

def load_json(file_path: Path) -> List[Dict]:
    """تحميل ملف JSON"""
    print(f"📂 تحميل: {file_path.name}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ تم تحميل {len(data)} عنصر")
    return data


def prepare_items(items: List[Dict], item_type: str) -> tuple:
    """
    تجهيز العناصر للإضافة إلى ChromaDB
    
    Args:
        items: قائمة العناصر
        item_type: نوع العنصر (document, section, paragraph)
    
    Returns:
        (ids, texts, metadatas)
    """
    ids = []
    texts = []
    metadatas = []
    
    for item in items:
        # ID
        if item_type == 'document':
            item_id = item['doc_id']
        elif item_type == 'section':
            item_id = item['section_id']
        else:  # paragraph
            item_id = item['para_id']
        
        ids.append(item_id)
        
        # Text - استخدام summary للـ documents
        if item_type == 'document':
            texts.append(item.get('summary', item.get('text', '')))
        else:
            texts.append(item.get('text', ''))
        
        # Metadata
        metadata = {
            'type': item_type,
            'word_count': item.get('stats', {}).get('word_count', 0)
        }
        
        # إضافة معلومات حسب النوع
        if item_type == 'document':
            metadata.update({
                'book': item.get('book', ''),
                'volume': item.get('volume', 0),
                'author': item.get('author', '')
            })
        elif item_type == 'section':
            metadata.update({
                'parent_doc': item.get('parent_doc', ''),
                'title': item.get('title', ''),
                'main_topic': item.get('main_topic', '')
            })
        else:  # paragraph
            metadata.update({
                'parent_section': item.get('parent_section', ''),
                'parent_doc': item.get('parent_doc', ''),
                'page': str(item.get('stats', {}).get('page', ''))
            })
        
        metadatas.append(metadata)
    
    return ids, texts, metadatas


# =============================================================================
# الدالة الرئيسية
# =============================================================================

def main():
    """الدالة الرئيسية"""
    
    print("=" * 70)
    print("🚀 Step 3: Embeddings with multilingual-e5-large")
    print("=" * 70)
    print("ℹ️ استخدام أقوى نموذج للعربية - Microsoft E5")
    print("ℹ️ الأبعاد: 1024 (بدلاً من 768)")
    print()
    
    start_time = time.time()
    
    # =============================================================================
    # 1. تحميل البيانات
    # =============================================================================
    
    print("📂 المرحلة 1: تحميل البيانات")
    print("-" * 70)
    
    documents = load_json(DOCUMENTS_FILE)
    sections = load_json(SECTIONS_FILE)
    paragraphs = load_json(PARAGRAPHS_FILE)
    
    total_items = len(documents) + len(sections) + len(paragraphs)
    print(f"\n📊 الإجمالي: {total_items} عنصر")
    print()
    
    # =============================================================================
    # 2. تهيئة Embeddings Generator
    # =============================================================================
    
    print("📂 المرحلة 2: تهيئة E5 Embeddings Generator")
    print("-" * 70)
    
    generator = EmbeddingsGenerator()
    
    # =============================================================================
    # 3. تهيئة ChromaDB
    # =============================================================================
    
    print("📂 المرحلة 3: تهيئة ChromaDB")
    print("-" * 70)
    
    db = ChromaDBBuilder()
    print()
    
    # =============================================================================
    # 4. معالجة Documents
    # =============================================================================
    
    print("📂 المرحلة 4: معالجة Documents")
    print("-" * 70)
    
    doc_ids, doc_texts, doc_metadatas = prepare_items(documents, 'document')
    
    print(f"🔢 توليد E5 embeddings لـ {len(doc_texts)} document...")
    doc_embeddings = generator.encode_batch(
        doc_texts, 
        show_progress=SHOW_PROGRESS,
        prefix="passage"
    )
    
    print(f"💾 إضافة Documents إلى ChromaDB...")
    db.add_items(doc_ids, doc_embeddings, doc_texts, doc_metadatas)
    print("✅ تم")
    print()
    
    # =============================================================================
    # 5. معالجة Sections
    # =============================================================================
    
    print("📂 المرحلة 5: معالجة Sections")
    print("-" * 70)
    
    sec_ids, sec_texts, sec_metadatas = prepare_items(sections, 'section')
    
    print(f"🔢 توليد E5 embeddings لـ {len(sec_texts)} section...")
    sec_embeddings = generator.encode_batch(
        sec_texts, 
        show_progress=SHOW_PROGRESS,
        prefix="passage"
    )
    
    print(f"💾 إضافة Sections إلى ChromaDB...")
    db.add_items(sec_ids, sec_embeddings, sec_texts, sec_metadatas)
    print("✅ تم")
    print()
    
    # =============================================================================
    # 6. معالجة Paragraphs
    # =============================================================================
    
    print("📂 المرحلة 6: معالجة Paragraphs")
    print("-" * 70)
    
    para_ids, para_texts, para_metadatas = prepare_items(paragraphs, 'paragraph')
    
    print(f"🔢 توليد E5 embeddings لـ {len(para_texts)} paragraph...")
    para_embeddings = generator.encode_batch(
        para_texts, 
        show_progress=SHOW_PROGRESS,
        prefix="passage"
    )
    
    print(f"💾 إضافة Paragraphs إلى ChromaDB...")
    db.add_items(para_ids, para_embeddings, para_texts, para_metadatas)
    print("✅ تم")
    print()
    
    # =============================================================================
    # 7. الإحصائيات النهائية
    # =============================================================================
    
    print("📂 المرحلة 7: الإحصائيات النهائية")
    print("-" * 70)
    
    db_stats = db.get_stats()
    
    total_time = time.time() - start_time
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "model_type": "Microsoft E5 (state-of-the-art for Arabic)",
        "embedding_dimension": EMBEDDING_DIM,
        "device": DEVICE,
        
        "data": {
            "documents": len(documents),
            "sections": len(sections),
            "paragraphs": len(paragraphs),
            "total": total_items
        },
        
        "database": db_stats,
        
        "performance": {
            "total_time_seconds": round(total_time, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "items_per_second": round(total_items / total_time, 2)
        },
        
        "model_info": {
            "advantages": [
                "أقوى نموذج للعربية",
                "1024 أبعاد (أكبر من paraphrase-768)",
                "100 لغة",
                "state-of-the-art performance"
            ]
        }
    }
    
    # حفظ الإحصائيات
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # طباعة الإحصائيات
    print(f"📊 العناصر المعالجة:")
    print(f"   - Documents: {stats['data']['documents']}")
    print(f"   - Sections: {stats['data']['sections']}")
    print(f"   - Paragraphs: {stats['data']['paragraphs']}")
    print(f"   - الإجمالي: {stats['data']['total']}")
    print()
    print(f"💾 قاعدة البيانات:")
    print(f"   - العناصر في ChromaDB: {db_stats['total_items']}")
    print(f"   - النموذج: {MODEL_NAME}")
    print(f"   - الأبعاد: {EMBEDDING_DIM}")
    print(f"   - المسار: {CHROMA_DB_PATH}")
    print()
    print(f"⏱️ الأداء:")
    print(f"   - الوقت الإجمالي: {stats['performance']['total_time_minutes']:.2f} دقيقة")
    print(f"   - السرعة: {stats['performance']['items_per_second']:.2f} عنصر/ثانية")
    print()
    print(f"💾 الإحصائيات محفوظة في: {STATS_FILE}")
    print()
    
    # =============================================================================
    # 8. اختبار البحث
    # =============================================================================
    
    print("📂 المرحلة 8: اختبار البحث (E5)")
    print("-" * 70)
    
    # تجربة بحث بسيطة
    test_query = "الإمامة"
    print(f"🔍 اختبار البحث عن: '{test_query}'")
    print(f"ℹ️ ملاحظة: E5 يحتاج prefix 'query:' للاستعلامات")
    
    # مهم: للاستعلامات نستخدم prefix "query: "
    query_embedding = generator.encode_batch(
        [test_query], 
        show_progress=False,
        prefix="query"  # للاستعلامات
    )[0]
    
    results = db.collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"\n📋 النتائج (أول 3):")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0]), 1):
        print(f"\n{i}. ID: {doc_id}")
        print(f"   النوع: {metadata['type']}")
        if metadata['type'] == 'section':
            print(f"   العنوان: {metadata['title']}")
        elif metadata['type'] == 'document':
            print(f"   الكتاب: {metadata['book']}")
        print(f"   عدد الكلمات: {metadata['word_count']}")
    
    print()
    
    # =============================================================================
    # النهاية
    # =============================================================================
    
    print("=" * 70)
    print("✅ تم إكمال Step 3 بنجاح باستخدام E5!")
    print("=" * 70)
    print()
    print("🎉 الآن لديك أقوى embeddings للعربية!")
    print(f"   - النموذج: {MODEL_NAME}")
    print(f"   - الأبعاد: {EMBEDDING_DIM}")
    print(f"   - الجودة: state-of-the-art")
    print()
    print("📦 الملفات الناتجة:")
    print(f"   - {CHROMA_DB_PATH}")
    print(f"   - {STATS_FILE}")
    print()
    print("🎯 الخطوة التالية: Step 4 - Query Analyzer")
    print()


# =============================================================================
# التشغيل
# =============================================================================

if __name__ == "__main__":
    main()