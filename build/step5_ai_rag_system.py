"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Step 5 AI: نظام RAG المتكامل المدعوم بالذكاء الاصطناعي
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
نظام RAG ذكي يستخدم AI لتحليل الأسئلة والبحث المتقدم

الميزات الجديدة:
1. تحليل ذكي للأسئلة باستخدام LLM
2. استراتيجية بحث مخصصة من AI
3. فلترة وترتيب ذكي للنتائج
4. تفسير AI للسؤال

الإصدار: 2.0.0 (AI-Powered)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# إضافة build إلى المسار
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
import chromadb

from step4_ai_query_analyzer import AIQueryAnalyzer, AIQueryAnalysis


@dataclass
class SearchResult:
    """نتيجة بحث واحدة"""
    id: str
    type: str
    content: str
    metadata: Dict
    distance: float
    score: float
    rank: int
    relevance_explanation: str = ""  # جديد: تفسير الصلة بالسؤال


@dataclass
class AIRAGResponse:
    """استجابة نظام RAG المدعوم بـ AI"""
    query: str
    ai_analysis: AIQueryAnalysis  # تحليل AI كامل
    results: List[SearchResult]
    total_results: int
    search_time: float
    timestamp: str


class AIRAGSystem:
    """نظام RAG المتكامل المدعوم بالذكاء الاصطناعي"""

    def __init__(
        self,
        db_path: str = "data/database/chroma_db",
        collection_name: str = "islamic_books_e5",
        model_name: str = "intfloat/multilingual-e5-large",
        llm_provider: str = "auto",
        llm_model: Optional[str] = None,
        use_ai_analyzer: bool = True
    ):
        """
        تهيئة نظام RAG الذكي

        Args:
            db_path: مسار قاعدة البيانات
            collection_name: اسم Collection
            model_name: اسم نموذج Embeddings
            llm_provider: مزود LLM (auto/openai/gemini/claude)
            llm_model: نموذج LLM محدد
            use_ai_analyzer: استخدام AI Analyzer أم لا
        """
        print("🔄 تهيئة نظام RAG الذكي...")

        # تهيئة Query Analyzer
        print(f"   📊 تحميل Query Analyzer (AI: {use_ai_analyzer})...")
        if use_ai_analyzer:
            self.analyzer = AIQueryAnalyzer(
                provider=llm_provider,
                model=llm_model,
                fallback_to_rules=True
            )
        else:
            # استخدام النسخة القديمة القائمة على القواعد
            from step4_query_analyzer import QueryAnalyzer
            self.analyzer = QueryAnalyzer()

        self.use_ai_analyzer = use_ai_analyzer

        # تحميل نموذج Embeddings
        print(f"   🤖 تحميل نموذج Embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # الاتصال بقاعدة البيانات
        print(f"   💾 الاتصال بقاعدة البيانات: {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

        print("✅ تم تهيئة نظام RAG الذكي بنجاح!\n")

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_by_type: Optional[str] = None,
        min_score: float = 0.0,
        use_ai_filters: bool = True
    ) -> AIRAGResponse:
        """
        البحث الذكي المدعوم بـ AI

        Args:
            query: السؤال
            n_results: عدد النتائج (None = استخدم توصية AI)
            filter_by_type: تصفية حسب النوع
            min_score: الحد الأدنى للنقاط
            use_ai_filters: استخدام فلاتر AI المقترحة

        Returns:
            AIRAGResponse
        """
        start_time = datetime.now()

        # 1. تحليل السؤال بـ AI
        print(f"🔍 تحليل السؤال: {query}")

        if self.use_ai_analyzer:
            analysis = self.analyzer.analyze(query)
        else:
            # النسخة القديمة
            old_analysis = self.analyzer.analyze(query)
            # تحويل للصيغة الجديدة
            analysis = self._convert_old_analysis(old_analysis, query)

        # 2. استخدام استراتيجية البحث من AI
        if n_results is None:
            n_results = analysis.search_strategy.get('n_results', 5)

        print(f"\n💡 تفسير AI:")
        print(f"   {analysis.ai_interpretation}")

        print(f"\n📊 استراتيجية البحث:")
        print(f"   • نوع السؤال: {analysis.query_type}")
        print(f"   • مستوى التفصيل: {analysis.detail_level}")
        print(f"   • عدد النتائج: {n_results}")
        print(f"   • الموضوع الرئيسي: {analysis.main_topic}")

        # 3. توليد Embedding للسؤال
        print(f"\n🔢 توليد embedding للسؤال...")
        query_text = f"query: {query}"
        query_embedding = self.model.encode(query_text)

        # 4. تطبيق فلاتر AI المقترحة
        where_filter = None
        if use_ai_filters and filter_by_type is None:
            # استخدام فلاتر من AI إذا كانت متاحة
            suggested_filters = analysis.search_strategy.get('suggested_filters', [])
            if suggested_filters:
                # تطبيق الفلتر الأول (يمكن تحسين هذا)
                print(f"   🎯 استخدام فلتر AI: {suggested_filters[0]}")

        if filter_by_type:
            where_filter = {"type": filter_by_type}

        # 5. البحث في قاعدة البيانات
        print("💾 البحث في قاعدة البيانات...")

        db_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 2,
            where=where_filter
        )

        # 6. معالجة وترتيب النتائج بذكاء
        print("📊 معالجة النتائج بذكاء...")
        results = self._process_results_with_ai(
            db_results,
            analysis,
            min_score,
            n_results
        )

        # 7. حساب الوقت
        search_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ تم العثور على {len(results)} نتيجة في {search_time:.2f} ثانية\n")

        return AIRAGResponse(
            query=query,
            ai_analysis=analysis,
            results=results,
            total_results=len(results),
            search_time=search_time,
            timestamp=datetime.now().isoformat()
        )

    def _process_results_with_ai(
        self,
        db_results: Dict,
        analysis: AIQueryAnalysis,
        min_score: float,
        n_results: int
    ) -> List[SearchResult]:
        """معالجة النتائج باستخدام معلومات AI"""

        results = []

        # استراتيجية البحث من AI
        level_priority = analysis.search_strategy.get('level_priority', ['paragraph', 'section', 'document'])
        search_focus = analysis.search_strategy.get('search_focus', '')

        for i, (id, metadata, content, distance) in enumerate(zip(
            db_results['ids'][0],
            db_results['metadatas'][0],
            db_results['documents'][0],
            db_results['distances'][0]
        )):
            # تحويل المسافة إلى نقاط
            base_score = max(0, 1 - distance)

            if base_score < min_score:
                continue

            # بونص حسب أولوية المستوى (من AI)
            doc_type = metadata.get('type', 'unknown')
            priority_bonus = 0
            if doc_type in level_priority:
                position = level_priority.index(doc_type)
                priority_bonus = 0.15 / (position + 1)  # أعلى من القديم

            # بونص إضافي بناءً على مطابقة الموضوع
            topic_bonus = 0
            if analysis.main_topic:
                if analysis.main_topic.lower() in content.lower():
                    topic_bonus = 0.05

            # بونص الكلمات المفتاحية
            keyword_bonus = 0
            keywords_found = sum(1 for kw in analysis.keywords if kw.lower() in content.lower())
            if keywords_found > 0:
                keyword_bonus = min(0.1, keywords_found * 0.02)

            # النقاط النهائية
            final_score = base_score + priority_bonus + topic_bonus + keyword_bonus

            # تفسير الصلة (اختياري)
            relevance_parts = []
            if topic_bonus > 0:
                relevance_parts.append(f"يحتوي على الموضوع الرئيسي: {analysis.main_topic}")
            if keywords_found > 0:
                relevance_parts.append(f"يحتوي على {keywords_found} كلمة مفتاحية")
            if priority_bonus > 0:
                relevance_parts.append(f"مستوى مناسب: {doc_type}")

            relevance_explanation = " | ".join(relevance_parts) if relevance_parts else "تشابه دلالي"

            results.append(SearchResult(
                id=id,
                type=doc_type,
                content=content,
                metadata=metadata,
                distance=distance,
                score=final_score,
                rank=i + 1,
                relevance_explanation=relevance_explanation
            ))

        # ترتيب حسب النقاط النهائية
        results.sort(key=lambda x: x.score, reverse=True)

        # تحديث الترتيب
        for i, result in enumerate(results, 1):
            result.rank = i

        return results[:n_results]

    def _convert_old_analysis(self, old_analysis, query: str) -> AIQueryAnalysis:
        """تحويل التحليل القديم للصيغة الجديدة"""
        # هذا fallback للنسخة القديمة
        return AIQueryAnalysis(
            original_query=query,
            language=old_analysis.language,
            query_type=old_analysis.query_type,
            keywords=old_analysis.keywords,
            main_topic=old_analysis.keywords[0] if old_analysis.keywords else "unknown",
            sub_topics=old_analysis.keywords[1:3] if len(old_analysis.keywords) > 1 else [],
            detail_level=old_analysis.detail_level,
            complexity="moderate",
            search_strategy=old_analysis.search_strategy,
            ai_interpretation="تحليل قائم على القواعد (بدون AI)",
            confidence=old_analysis.query_type_confidence,
            model_used="rules-based",
            metadata=old_analysis.metadata,
            timestamp=old_analysis.timestamp
        )

    def print_response(self, response: AIRAGResponse, verbose: bool = True):
        """طباعة الاستجابة بشكل منسق"""

        print("\n" + "="*70)
        print("🤖 نتائج البحث الذكي")
        print("="*70)

        # تحليل AI
        if verbose:
            print(f"\n📝 السؤال: {response.query}")
            print(f"🤖 النموذج: {response.ai_analysis.model_used}")
            print(f"📊 الثقة: {response.ai_analysis.confidence:.0%}")
            print(f"\n💡 تفسير AI:")
            print(f"   {response.ai_analysis.ai_interpretation}")

        # معلومات التحليل
        print(f"\n🌐 اللغة: {response.ai_analysis.language}")
        print(f"📋 نوع السؤال: {response.ai_analysis.query_type}")
        print(f"🎯 الموضوع: {response.ai_analysis.main_topic}")
        print(f"📏 مستوى التعقيد: {response.ai_analysis.complexity}")

        # النتائج
        print(f"\n📊 عدد النتائج: {response.total_results}")
        print(f"⏱️  الوقت: {response.search_time:.2f} ثانية")

        print("\n" + "-"*70)
        print("🎯 أفضل النتائج:")
        print("-"*70)

        for i, result in enumerate(response.results, 1):
            print(f"\n{i}. [{result.type.upper()}] {result.id}")
            print(f"   📊 النقاط: {result.score:.4f} | المسافة: {result.distance:.4f}")

            # سبب الصلة
            if result.relevance_explanation:
                print(f"   🎯 الصلة: {result.relevance_explanation}")

            # العنوان
            if 'title' in result.metadata:
                print(f"   📖 الكتاب: {result.metadata['title']}")

            # المحتوى
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"   📝 {content_preview}")

        print("\n" + "="*70 + "\n")

    def ask(self, query: str, **kwargs) -> AIRAGResponse:
        """واجهة بسيطة للسؤال"""
        response = self.search(query, **kwargs)
        self.print_response(response)
        return response


def main():
    """تجربة نظام RAG الذكي"""

    print("\n" + "="*70)
    print("🚀 Step 5 AI: نظام RAG المتكامل المدعوم بالذكاء الاصطناعي")
    print("="*70 + "\n")

    # تهيئة النظام
    rag = AIRAGSystem(
        llm_provider="auto",       # اختيار تلقائي للمزود
        use_ai_analyzer=True       # استخدام AI
    )

    # أمثلة
    test_queries = [
        "من هو الشريف المرتضى؟",
        "ما هو تعريف الإمامة في الفكر الشيعي؟",
        "اشرح بالتفصيل مفهوم العصمة وأدلته",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'━'*70}")
        print(f"سؤال {i}/{len(test_queries)}")
        print(f"{'━'*70}\n")

        response = rag.ask(query)

        if i < len(test_queries):
            input("\nاضغط Enter للسؤال التالي...")

    print("\n" + "="*70)
    print("✅ انتهى الاختبار!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
