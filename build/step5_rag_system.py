"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Step 5: نظام RAG المتكامل
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
نظام Retrieval-Augmented Generation كامل للمحتوى الديني الإسلامي

المهام:
1. تحليل السؤال باستخدام Query Analyzer
2. البحث في قاعدة البيانات ChromaDB
3. تصفية وترتيب النتائج
4. تقديم إجابات ذكية

الإصدار: 1.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# إضافة build إلى المسار
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
import chromadb

from step4_query_analyzer import QueryAnalyzer, QueryAnalysis


@dataclass
class SearchResult:
    """نتيجة بحث واحدة"""
    id: str
    type: str  # document, section, paragraph
    content: str
    metadata: Dict
    distance: float  # المسافة (أقل = أفضل)
    score: float  # النقاط (أعلى = أفضل)
    rank: int  # الترتيب


@dataclass
class RAGResponse:
    """استجابة نظام RAG"""
    query: str
    query_analysis: QueryAnalysis
    results: List[SearchResult]
    total_results: int
    search_time: float
    timestamp: str


class RAGSystem:
    """نظام RAG المتكامل"""

    def __init__(
        self,
        db_path: str = "data/database/chroma_db",
        collection_name: str = "islamic_books_e5",
        model_name: str = "intfloat/multilingual-e5-large"
    ):
        """
        تهيئة نظام RAG

        Args:
            db_path: مسار قاعدة البيانات
            collection_name: اسم Collection
            model_name: اسم نموذج Embeddings
        """
        print("🔄 تهيئة نظام RAG...")

        # تهيئة Query Analyzer
        print("   📊 تحميل Query Analyzer...")
        self.analyzer = QueryAnalyzer()

        # تحميل نموذج Embeddings
        print(f"   🤖 تحميل نموذج Embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # الاتصال بقاعدة البيانات
        print(f"   💾 الاتصال بقاعدة البيانات: {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

        print("✅ تم تهيئة نظام RAG بنجاح!\n")

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_by_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> RAGResponse:
        """
        البحث الذكي في قاعدة البيانات

        Args:
            query: السؤال
            n_results: عدد النتائج (None = استخدم Query Analyzer)
            filter_by_type: تصفية حسب النوع (document/section/paragraph)
            min_score: الحد الأدنى للنقاط

        Returns:
            RAGResponse: الاستجابة الكاملة
        """
        start_time = datetime.now()

        # 1. تحليل السؤال
        print(f"🔍 تحليل السؤال: {query}")
        analysis = self.analyzer.analyze(query)

        # 2. تحديد عدد النتائج
        if n_results is None:
            n_results = analysis.search_strategy['n_results']

        print(f"   📊 نوع السؤال: {analysis.query_type}")
        print(f"   🌐 اللغة: {analysis.language}")
        print(f"   📏 مستوى التفصيل: {analysis.detail_level}")
        print(f"   🎯 عدد النتائج المطلوبة: {n_results}\n")

        # 3. توليد Embedding للسؤال
        print("🔢 توليد embedding للسؤال...")
        query_text = f"query: {query}"  # بادئة E5
        query_embedding = self.model.encode(query_text)

        # 4. البحث في قاعدة البيانات
        print("💾 البحث في قاعدة البيانات...")

        # إنشاء where filter إذا لزم الأمر
        where_filter = None
        if filter_by_type:
            where_filter = {"type": filter_by_type}

        db_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 2,  # جلب ضعف العدد للتصفية
            where=where_filter
        )

        # 5. معالجة وترتيب النتائج
        print("📊 معالجة النتائج...")
        results = self._process_results(
            db_results,
            analysis,
            min_score,
            n_results
        )

        # 6. حساب الوقت
        search_time = (datetime.now() - start_time).total_seconds()

        print(f"✅ تم العثور على {len(results)} نتيجة في {search_time:.2f} ثانية\n")

        return RAGResponse(
            query=query,
            query_analysis=analysis,
            results=results,
            total_results=len(results),
            search_time=search_time,
            timestamp=datetime.now().isoformat()
        )

    def _process_results(
        self,
        db_results: Dict,
        analysis: QueryAnalysis,
        min_score: float,
        n_results: int
    ) -> List[SearchResult]:
        """معالجة وترتيب النتائج"""

        results = []

        # استخراج النتائج
        for i, (id, metadata, content, distance) in enumerate(zip(
            db_results['ids'][0],
            db_results['metadatas'][0],
            db_results['documents'][0],
            db_results['distances'][0]
        )):
            # تحويل المسافة إلى نقاط (1 - distance)
            # ChromaDB يستخدم L2 distance، أقل = أفضل
            score = max(0, 1 - distance)

            # تطبيق الحد الأدنى للنقاط
            if score < min_score:
                continue

            # ترتيب حسب أولوية المستويات
            level_priority = analysis.search_strategy['level_priority']
            doc_type = metadata.get('type', 'unknown')

            # إضافة بونص للمستوى المفضل
            priority_bonus = 0
            if doc_type in level_priority:
                # الأولوية الأولى تحصل على 0.1، الثانية 0.05، إلخ
                position = level_priority.index(doc_type)
                priority_bonus = 0.1 / (position + 1)

            final_score = score + priority_bonus

            results.append(SearchResult(
                id=id,
                type=doc_type,
                content=content,
                metadata=metadata,
                distance=distance,
                score=final_score,
                rank=i + 1
            ))

        # ترتيب حسب النقاط النهائية
        results.sort(key=lambda x: x.score, reverse=True)

        # تحديث الترتيب
        for i, result in enumerate(results, 1):
            result.rank = i

        # إرجاع العدد المطلوب فقط
        return results[:n_results]

    def print_response(self, response: RAGResponse, verbose: bool = True):
        """طباعة الاستجابة بشكل منسق"""

        print("\n" + "="*70)
        print("📋 نتائج البحث")
        print("="*70)

        # تحليل السؤال
        if verbose:
            print(f"\n📝 السؤال: {response.query}")
            print(f"🌐 اللغة: {response.query_analysis.language}")
            print(f"📊 نوع السؤال: {response.query_analysis.query_type}")
            print(f"📏 مستوى التفصيل: {response.query_analysis.detail_level}")

        # الكلمات المفتاحية
        if verbose and response.query_analysis.keywords:
            print(f"\n🔑 الكلمات المفتاحية: {', '.join(response.query_analysis.keywords[:5])}")

        # النتائج
        print(f"\n📊 عدد النتائج: {response.total_results}")
        print(f"⏱️  الوقت: {response.search_time:.2f} ثانية")

        print("\n" + "-"*70)
        print("🎯 أفضل النتائج:")
        print("-"*70)

        for i, result in enumerate(response.results, 1):
            print(f"\n{i}. [{result.type.upper()}] {result.id}")
            print(f"   📊 النقاط: {result.score:.4f} | المسافة: {result.distance:.4f}")

            # العنوان إذا وجد
            if 'title' in result.metadata:
                print(f"   📖 الكتاب: {result.metadata['title']}")

            if 'author' in result.metadata:
                print(f"   ✍️  المؤلف: {result.metadata['author']}")

            # المحتوى
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"   📝 {content_preview}")

            # معلومات إضافية
            if verbose:
                if 'word_count' in result.metadata:
                    print(f"   📊 عدد الكلمات: {result.metadata['word_count']}")

        print("\n" + "="*70 + "\n")

    def ask(self, query: str, **kwargs) -> RAGResponse:
        """
        واجهة بسيطة للسؤال

        Args:
            query: السؤال
            **kwargs: معاملات إضافية للبحث

        Returns:
            RAGResponse
        """
        response = self.search(query, **kwargs)
        self.print_response(response)
        return response


def main():
    """تجربة نظام RAG"""

    print("\n" + "="*70)
    print("🚀 Step 5: نظام RAG المتكامل")
    print("="*70 + "\n")

    # تهيئة النظام
    rag = RAGSystem()

    # أمثلة متنوعة
    test_queries = [
        "من هو الشريف المرتضى؟",
        "ما هو تعريف الإمامة في الفكر الشيعي؟",
        "اشرح بالتفصيل مفهوم العصمة",
        "ما الفرق بين الإمامة والخلافة؟",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'━'*70}")
        print(f"سؤال {i}/{len(test_queries)}")
        print(f"{'━'*70}\n")

        response = rag.ask(query)

        # فاصل بين الأسئلة
        if i < len(test_queries):
            input("\nاضغط Enter للسؤال التالي...")

    print("\n" + "="*70)
    print("✅ انتهى الاختبار!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
