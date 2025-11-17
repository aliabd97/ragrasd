#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5: Complete RAG System with Enhanced Answer Generation
===========================================================

المهمة:
- نظام RAG كامل
- توليد إجابات مطولة ومفصلة (مقالة كاملة)
- مصادر واضحة بالاسم الحقيقي (مثل: الكافي 1/34)
- دعم OpenAI Embeddings
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from datetime import datetime

# ChromaDB
import chromadb
from chromadb.config import Settings

# Query Analyzer
from step4_query_analyzer import QueryAnalyzer


# =============================================================================
# تحميل الإعدادات
# =============================================================================

def load_config(config_path: str = "../config.yaml") -> Dict:
    """تحميل ملف الإعدادات"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# Embeddings Manager - يدعم OpenAI و Sentence Transformers
# =============================================================================

class EmbeddingsManager:
    """
    مدير Embeddings - يدعم:
    1. OpenAI (text-embedding-3-small, text-embedding-3-large)
    2. Sentence Transformers (multilingual-e5-large)
    """

    def __init__(self, provider: str = "openai", model: str = None):
        """
        التهيئة

        Args:
            provider: "openai" أو "sentence_transformers"
            model: اسم النموذج
        """
        self.provider = provider

        if provider == "openai":
            self._init_openai(model or "text-embedding-3-small")
        else:
            self._init_sentence_transformers(model or "intfloat/multilingual-e5-large")

    def _init_openai(self, model: str):
        """تهيئة OpenAI"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("يرجى تثبيت openai: pip install openai")

        # التأكد من وجود API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("يرجى تعيين OPENAI_API_KEY في ملف .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model

        print(f"✅ تم تهيئة OpenAI Embeddings: {model}")

    def _init_sentence_transformers(self, model: str):
        """تهيئة Sentence Transformers"""
        from sentence_transformers import SentenceTransformer

        self.model_obj = SentenceTransformer(model)
        self.model = model

        print(f"✅ تم تهيئة Sentence Transformers: {model}")

    def encode(self, text: str, prefix: str = "query") -> List[float]:
        """
        تحويل نص إلى embedding

        Args:
            text: النص
            prefix: البادئة (للـ E5: "query" أو "passage")

        Returns:
            embedding vector
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        else:
            # Sentence Transformers (E5)
            prefixed_text = f"{prefix}: {text}"
            embedding = self.model_obj.encode(
                prefixed_text,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embedding.tolist()


# =============================================================================
# RAG System
# =============================================================================

class RAGSystem:
    """نظام RAG كامل مع توليد إجابات محسّنة"""

    def __init__(
        self,
        db_path: str = "../data/database/chroma_db",
        collection_name: str = "islamic_books_e5",
        embeddings_provider: str = "openai",
        embeddings_model: str = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        التهيئة

        Args:
            db_path: مسار قاعدة البيانات
            collection_name: اسم collection
            embeddings_provider: "openai" أو "sentence_transformers"
            embeddings_model: اسم نموذج embeddings
            llm_provider: "openai" (للإجابات)
            llm_model: نموذج LLM
        """
        # Query Analyzer
        self.query_analyzer = QueryAnalyzer()

        # Embeddings
        self.embeddings = EmbeddingsManager(
            provider=embeddings_provider,
            model=embeddings_model
        )

        # ChromaDB
        print(f"📂 فتح ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ تم الاتصال بـ collection: {collection_name}")
        except Exception as e:
            print(f"❌ خطأ: لم يتم العثور على collection: {collection_name}")
            raise e

        # LLM للإجابات
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        if llm_provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("يرجى تثبيت openai: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("يرجى تعيين OPENAI_API_KEY في ملف .env")

            self.llm_client = OpenAI(api_key=api_key)
            print(f"✅ تم تهيئة OpenAI LLM: {llm_model}")

    def search(
        self,
        query: str,
        n_results: int = 10,
        include_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        البحث في قاعدة البيانات

        Args:
            query: الاستعلام
            n_results: عدد النتائج
            include_types: أنواع العناصر المطلوبة

        Returns:
            نتائج البحث
        """
        # تحليل الاستعلام
        query_info = self.query_analyzer.analyze(query)

        # توليد embedding
        query_embedding = self.embeddings.encode(query, prefix="query")

        # البحث
        where_filter = None
        if include_types:
            where_filter = {"type": {"$in": include_types}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )

        return {
            'query_info': query_info,
            'results': results
        }

    def build_sources_list(self, metadatas: List[Dict]) -> str:
        """
        بناء قائمة المصادر بالأسماء الحقيقية

        Args:
            metadatas: metadata من نتائج البحث

        Returns:
            قائمة المصادر المنسقة
        """
        sources_text = "### المصادر المتاحة:\n\n"

        for i, meta in enumerate(metadatas, 1):
            # استخراج معلومات المصدر
            source_type = meta.get('type', '')

            if source_type == 'paragraph':
                book = meta.get('parent_doc', '').split('_vol')[0] if meta.get('parent_doc') else 'كتاب غير محدد'
                page = meta.get('page', '')
                source_name = f"{book} (ص {page})" if page else book

            elif source_type == 'section':
                parent = meta.get('parent_doc', '')
                book = parent.split('_vol')[0] if parent else 'كتاب غير محدد'
                title = meta.get('title', '')
                source_name = f"{book} - {title}" if title else book

            elif source_type == 'document':
                book = meta.get('book', 'كتاب غير محدد')
                volume = meta.get('volume', '')
                source_name = f"{book} ({volume})" if volume else book

            else:
                source_name = "مصدر غير محدد"

            sources_text += f"{i}. {source_name}\n"

        return sources_text

    def build_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """
        بناء السياق من النتائج

        Args:
            documents: نصوص النتائج
            metadatas: metadata النتائج

        Returns:
            السياق المنسق
        """
        context = "### النصوص المرجعية:\n\n"

        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            # معلومات المصدر
            source_type = meta.get('type', '')

            if source_type == 'paragraph':
                book = meta.get('parent_doc', '').split('_vol')[0] if meta.get('parent_doc') else 'كتاب غير محدد'
                page = meta.get('page', '')
                source_info = f"{book} (ص {page})" if page else book

            elif source_type == 'section':
                parent = meta.get('parent_doc', '')
                book = parent.split('_vol')[0] if parent else 'كتاب غير محدد'
                title = meta.get('title', '')
                source_info = f"{book} - {title}" if title else book

            elif source_type == 'document':
                book = meta.get('book', 'كتاب غير محدد')
                volume = meta.get('volume', '')
                source_info = f"{book} ({volume})" if volume else book

            else:
                source_info = "مصدر غير محدد"

            context += f"**[{i}] من {source_info}:**\n{doc}\n\n"

        return context

    def generate_answer(
        self,
        query: str,
        context: str,
        sources_list: str,
        query_info: Dict
    ) -> str:
        """
        توليد إجابة مفصلة باستخدام LLM

        Args:
            query: السؤال
            context: السياق
            sources_list: قائمة المصادر
            query_info: معلومات تحليل السؤال

        Returns:
            الإجابة المولدة
        """
        # تحديد طول الإجابة المطلوب
        requires_detailed = query_info.get('requires_detailed_answer', True)

        # بناء prompt محسّن
        system_prompt = """أنت عالم ديني متخصص في الإجابة على الأسئلة الدينية بطريقة علمية ودقيقة.

**مهمتك:**
1. اكتب إجابة شاملة ومفصلة على السؤال (مثل مقالة علمية كاملة)
2. استخدم المصادر المتاحة بالأسماء الحقيقية
3. اذكر المصدر بالشكل الصحيح مثل: "الكافي (1/34)" أو "نهج البلاغة (ص 156)"

**تعليمات مهمة:**
✅ استخدم أسماء الكتب الحقيقية من قائمة المصادر
✅ اذكر المصدر داخل النص مثل: "ورد في الكافي (1/34) أن..."
✅ اكتب إجابة طويلة ومفصلة (على الأقل 5-10 فقرات)
✅ قسّم الإجابة إلى أقسام واضحة
✅ في النهاية، أضف قائمة بكل المصادر المستخدمة

❌ لا تستخدم [المصدر 1] أو [المصدر 2]
❌ لا تكتب إجابات قصيرة
❌ لا تخترع معلومات غير موجودة في النصوص

**الهيكل المطلوب:**
- مقدمة
- شرح تفصيلي (عدة فقرات)
- أمثلة وشواهد من المصادر
- خاتمة
- قائمة المصادر المستخدمة"""

        user_prompt = f"""
{sources_list}

{context}

---

**السؤال:** {query}

**المطلوب:** اكتب مقالة علمية شاملة تجيب على هذا السؤال، مع الاستشهاد بالمصادر بأسمائها الحقيقية (مثل: الكافي 1/34، نهج البلاغة ص 156).

اكتب إجابة مفصلة لا تقل عن 5 فقرات، واذكر المصادر داخل النص بالشكل الصحيح.
"""

        # استدعاء LLM
        if self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=3000  # لإجابات طويلة
            )

            return response.choices[0].message.content

        return "خطأ: LLM provider غير مدعوم"

    def query(
        self,
        question: str,
        n_results: int = 10,
        include_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        الاستعلام الكامل مع توليد الإجابة

        Args:
            question: السؤال
            n_results: عدد نتائج البحث
            include_types: أنواع العناصر المطلوبة

        Returns:
            الإجابة الكاملة مع المصادر
        """
        print(f"\n🔍 السؤال: {question}\n")

        # 1. البحث
        print("📊 البحث في قاعدة البيانات...")
        search_results = self.search(question, n_results, include_types)

        query_info = search_results['query_info']
        results = search_results['results']

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        print(f"✅ تم العثور على {len(documents)} نتيجة\n")

        # 2. بناء قائمة المصادر
        print("📚 بناء قائمة المصادر...")
        sources_list = self.build_sources_list(metadatas)

        # 3. بناء السياق
        print("📝 بناء السياق...")
        context = self.build_context(documents, metadatas)

        # 4. توليد الإجابة
        print("🤖 توليد الإجابة...\n")
        answer = self.generate_answer(question, context, sources_list, query_info)

        return {
            'question': question,
            'answer': answer,
            'query_info': query_info,
            'num_sources': len(documents),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# الدالة الرئيسية - للاختبار
# =============================================================================

def main():
    """اختبار النظام"""

    print("=" * 70)
    print("🚀 Multi-Level RAG System - Enhanced Version")
    print("=" * 70)
    print()

    # تحميل من .env إذا موجود
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # تهيئة النظام
    print("⚙️ تهيئة نظام RAG...\n")

    # يمكنك تغيير provider هنا:
    # - embeddings_provider="openai" للاستخدام OpenAI
    # - embeddings_provider="sentence_transformers" للاستخدام E5

    rag = RAGSystem(
        embeddings_provider="openai",  # أو "sentence_transformers"
        embeddings_model="text-embedding-3-small",  # أو "intfloat/multilingual-e5-large"
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )

    # اختبار
    test_question = "ما هو مفهوم الإمامة في المذهب الشيعي؟"

    result = rag.query(test_question, n_results=10)

    # عرض النتيجة
    print("=" * 70)
    print("📝 الإجابة:")
    print("=" * 70)
    print()
    print(result['answer'])
    print()
    print("=" * 70)
    print(f"✅ تم استخدام {result['num_sources']} مصدر")
    print("=" * 70)


if __name__ == "__main__":
    main()
