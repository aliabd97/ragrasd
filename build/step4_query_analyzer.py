"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Step 4: Query Analyzer - محلل الأسئلة الذكي
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
نظام تحليل الأسئلة للمحتوى الديني الإسلامي

المهام:
1. كشف لغة السؤال (عربي/إنجليزي)
2. تصنيف نوع السؤال
3. استخراج الكلمات المفتاحية
4. تحديد مستوى التفصيل المطلوب
5. اقتراح استراتيجية البحث

الإصدار: 1.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Literal, Optional
from datetime import datetime
import json


@dataclass
class QueryAnalysis:
    """نتيجة تحليل السؤال"""

    # السؤال الأصلي
    original_query: str

    # اللغة المكتشفة
    language: Literal["arabic", "english", "mixed"]
    language_confidence: float  # 0-1

    # نوع السؤال
    query_type: Literal[
        "factual",        # سؤال حقيقي: من؟ ماذا؟ متى؟
        "definition",     # تعريف: ما هو؟ ما معنى؟
        "explanation",    # شرح: اشرح، وضح، كيف؟
        "comparison",     # مقارنة: الفرق بين، قارن
        "opinion",        # رأي: ما رأيك؟ هل تعتقد؟
        "list",           # قائمة: اذكر، عدد
        "procedural"      # إجرائي: كيف أفعل؟ خطوات؟
    ]
    query_type_confidence: float

    # الكلمات المفتاحية
    keywords: List[str]

    # كلمات السؤال (من، ماذا، كيف، etc.)
    question_words: List[str]

    # مستوى التفصيل المطلوب
    detail_level: Literal["brief", "moderate", "detailed"]

    # استراتيجية البحث المقترحة
    search_strategy: Dict[str, any]

    # معلومات إضافية
    metadata: Dict[str, any]

    # وقت التحليل
    timestamp: str


class QueryAnalyzer:
    """محلل الأسئلة الذكي"""

    def __init__(self):
        # أنماط كلمات السؤال
        self.arabic_question_words = {
            'من': 'who',
            'ماذا': 'what',
            'متى': 'when',
            'أين': 'where',
            'كيف': 'how',
            'لماذا': 'why',
            'هل': 'yes/no',
            'ما': 'what',
            'أي': 'which',
            'كم': 'how_many'
        }

        self.english_question_words = {
            'who', 'what', 'when', 'where', 'how', 'why',
            'which', 'whose', 'whom', 'is', 'are', 'was', 'were',
            'do', 'does', 'did', 'can', 'could', 'will', 'would'
        }

        # كلمات تحديد نوع السؤال
        self.type_indicators = {
            'definition': {
                'ar': ['ما هو', 'ما هي', 'تعريف', 'معنى', 'المقصود'],
                'en': ['what is', 'what are', 'define', 'meaning of', 'definition']
            },
            'explanation': {
                'ar': ['اشرح', 'وضح', 'فسر', 'بين', 'كيف'],
                'en': ['explain', 'describe', 'clarify', 'elaborate', 'how']
            },
            'comparison': {
                'ar': ['الفرق بين', 'قارن', 'مقارنة', 'الاختلاف', 'التشابه'],
                'en': ['difference between', 'compare', 'comparison', 'versus', 'vs']
            },
            'list': {
                'ar': ['اذكر', 'عدد', 'أمثلة', 'قائمة', 'أنواع'],
                'en': ['list', 'enumerate', 'examples', 'types of', 'mention']
            }
        }

        # كلمات توحي بمستوى التفصيل
        self.detail_indicators = {
            'brief': {
                'ar': ['بإيجاز', 'باختصار', 'ملخص', 'سريعا'],
                'en': ['briefly', 'summary', 'quick', 'short']
            },
            'detailed': {
                'ar': ['بالتفصيل', 'موسع', 'شامل', 'كامل', 'جميع'],
                'en': ['detailed', 'comprehensive', 'complete', 'thorough', 'all']
            }
        }

        # كلمات توقف عربية
        self.arabic_stopwords = {
            'في', 'على', 'إلى', 'من', 'عن', 'مع', 'هذا', 'ذلك',
            'التي', 'الذي', 'هذه', 'تلك', 'ال', 'و', 'أو', 'لكن'
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        تحليل السؤال بالكامل

        Args:
            query: السؤال المراد تحليله

        Returns:
            QueryAnalysis: نتيجة التحليل الشاملة
        """
        # تنظيف السؤال
        cleaned_query = self._clean_query(query)

        # كشف اللغة
        language, lang_confidence = self._detect_language(cleaned_query)

        # استخراج كلمات السؤال
        question_words = self._extract_question_words(cleaned_query, language)

        # تصنيف نوع السؤال
        query_type, type_confidence = self._classify_query_type(cleaned_query, language)

        # استخراج الكلمات المفتاحية
        keywords = self._extract_keywords(cleaned_query, language)

        # تحديد مستوى التفصيل
        detail_level = self._determine_detail_level(cleaned_query, language)

        # بناء استراتيجية البحث
        search_strategy = self._build_search_strategy(
            query_type, detail_level, keywords, language
        )

        # معلومات إضافية
        metadata = {
            'query_length': len(cleaned_query),
            'word_count': len(cleaned_query.split()),
            'has_question_mark': '?' in query or '؟' in query,
            'is_complex': len(keywords) > 3
        }

        return QueryAnalysis(
            original_query=query,
            language=language,
            language_confidence=lang_confidence,
            query_type=query_type,
            query_type_confidence=type_confidence,
            keywords=keywords,
            question_words=question_words,
            detail_level=detail_level,
            search_strategy=search_strategy,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    def _clean_query(self, query: str) -> str:
        """تنظيف السؤال من المسافات الزائدة والرموز غير الضرورية"""
        # إزالة المسافات الزائدة
        query = ' '.join(query.split())
        # إزالة الرموز المكررة
        query = re.sub(r'([?.!؟])+', r'\1', query)
        return query.strip()

    def _detect_language(self, query: str) -> tuple[str, float]:
        """
        كشف لغة السؤال

        Returns:
            (language, confidence)
        """
        # عد الحروف العربية والإنجليزية
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))

        total_chars = arabic_chars + english_chars

        if total_chars == 0:
            return "arabic", 0.5  # افتراضي

        arabic_ratio = arabic_chars / total_chars
        english_ratio = english_chars / total_chars

        # تحديد اللغة
        if arabic_ratio > 0.7:
            return "arabic", arabic_ratio
        elif english_ratio > 0.7:
            return "english", english_ratio
        else:
            return "mixed", max(arabic_ratio, english_ratio)

    def _extract_question_words(self, query: str, language: str) -> List[str]:
        """استخراج كلمات السؤال"""
        question_words = []
        query_lower = query.lower()

        if language in ["arabic", "mixed"]:
            for ar_word in self.arabic_question_words.keys():
                if ar_word in query_lower:
                    question_words.append(ar_word)

        if language in ["english", "mixed"]:
            words = query_lower.split()
            for en_word in self.english_question_words:
                if en_word in words:
                    question_words.append(en_word)

        return list(set(question_words))

    def _classify_query_type(self, query: str, language: str) -> tuple[str, float]:
        """
        تصنيف نوع السؤال

        Returns:
            (query_type, confidence)
        """
        query_lower = query.lower()
        scores = {
            'factual': 0.0,
            'definition': 0.0,
            'explanation': 0.0,
            'comparison': 0.0,
            'opinion': 0.0,
            'list': 0.0,
            'procedural': 0.0
        }

        # فحص مؤشرات الأنواع
        for qtype, indicators in self.type_indicators.items():
            lang_key = 'ar' if language in ['arabic', 'mixed'] else 'en'
            for indicator in indicators.get(lang_key, []):
                if indicator in query_lower:
                    scores[qtype] += 1.0

        # مؤشرات إضافية
        # تعريف
        if any(word in query_lower for word in ['ما هو', 'ما هي', 'what is', 'what are']):
            scores['definition'] += 1.5

        # شرح
        if any(word in query_lower for word in ['اشرح', 'وضح', 'explain', 'how']):
            scores['explanation'] += 1.5

        # مقارنة - أولوية عالية
        if any(word in query_lower for word in ['الفرق', 'قارن', 'difference', 'compare']):
            scores['comparison'] += 3.0  # أولوية أعلى للمقارنة

        # قائمة
        if any(word in query_lower for word in ['اذكر', 'عدد', 'list', 'enumerate']):
            scores['list'] += 1.5

        # إذا لم يتم اكتشاف نوع محدد، اعتبره سؤال حقيقي
        if max(scores.values()) == 0:
            if any(word in query_lower for word in ['من', 'who', 'متى', 'when', 'أين', 'where']):
                scores['factual'] = 1.0
            else:
                scores['definition'] = 0.5  # افتراضي

        # اختيار الأعلى
        query_type = max(scores.keys(), key=lambda k: scores[k])
        max_score = scores[query_type]

        # حساب الثقة
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5

        return query_type, min(confidence, 1.0)

    def _extract_keywords(self, query: str, language: str) -> List[str]:
        """استخراج الكلمات المفتاحية"""
        # إزالة علامات الترقيم
        cleaned = re.sub(r'[?.!؟،,;:]', ' ', query)
        words = cleaned.split()

        keywords = []

        for word in words:
            word_lower = word.lower()

            # تجاهل كلمات السؤال
            if word_lower in self.arabic_question_words:
                continue
            if word_lower in self.english_question_words:
                continue

            # تجاهل كلمات التوقف
            if word_lower in self.arabic_stopwords:
                continue

            # تجاهل الكلمات القصيرة جداً
            if len(word) < 2:
                continue

            keywords.append(word)

        return keywords[:10]  # أقصى 10 كلمات مفتاحية

    def _determine_detail_level(self, query: str, language: str) -> str:
        """تحديد مستوى التفصيل المطلوب"""
        query_lower = query.lower()

        lang_key = 'ar' if language in ['arabic', 'mixed'] else 'en'

        # فحص مؤشرات الإيجاز
        if any(word in query_lower for word in self.detail_indicators['brief'].get(lang_key, [])):
            return "brief"

        # فحص مؤشرات التفصيل
        if any(word in query_lower for word in self.detail_indicators['detailed'].get(lang_key, [])):
            return "detailed"

        # افتراضي: متوسط
        return "moderate"

    def _build_search_strategy(
        self,
        query_type: str,
        detail_level: str,
        keywords: List[str],
        language: str
    ) -> Dict[str, any]:
        """بناء استراتيجية البحث المناسبة"""

        # عدد النتائج المقترح
        n_results_map = {
            'brief': 3,
            'moderate': 5,
            'detailed': 10
        }

        # أولوية مستويات البيانات
        level_priority = {
            'factual': ['paragraph', 'section', 'document'],
            'definition': ['paragraph', 'section', 'document'],
            'explanation': ['section', 'paragraph', 'document'],
            'comparison': ['section', 'document', 'paragraph'],
            'opinion': ['document', 'section', 'paragraph'],
            'list': ['section', 'paragraph', 'document'],
            'procedural': ['section', 'paragraph', 'document']
        }

        return {
            'n_results': n_results_map.get(detail_level, 5),
            'level_priority': level_priority.get(query_type, ['paragraph', 'section', 'document']),
            'use_reranking': detail_level == 'detailed',
            'expand_query': len(keywords) < 3,
            'language': language,
            'search_modes': self._suggest_search_modes(query_type)
        }

    def _suggest_search_modes(self, query_type: str) -> List[str]:
        """اقتراح أوضاع البحث المناسبة"""
        modes = ['semantic']  # دائماً استخدم البحث الدلالي

        # أنواع معينة تستفيد من البحث بالكلمات المفتاحية
        if query_type in ['factual', 'list']:
            modes.append('keyword')

        # للمقارنة، قد نحتاج بحث متقدم
        if query_type == 'comparison':
            modes.append('multi_query')

        return modes

    def print_analysis(self, analysis: QueryAnalysis, verbose: bool = False):
        """طباعة نتيجة التحليل بشكل منسق"""
        print("\n" + "="*70)
        print("📊 تحليل السؤال")
        print("="*70)

        print(f"\n📝 السؤال: {analysis.original_query}")
        print(f"\n🌐 اللغة: {analysis.language} ({analysis.language_confidence:.0%})")
        print(f"📋 نوع السؤال: {analysis.query_type} ({analysis.query_type_confidence:.0%})")
        print(f"📏 مستوى التفصيل: {analysis.detail_level}")

        if analysis.question_words:
            print(f"\n❓ كلمات السؤال: {', '.join(analysis.question_words)}")

        if analysis.keywords:
            print(f"\n🔑 الكلمات المفتاحية:")
            for i, kw in enumerate(analysis.keywords, 1):
                print(f"   {i}. {kw}")

        print(f"\n🎯 استراتيجية البحث:")
        print(f"   • عدد النتائج: {analysis.search_strategy['n_results']}")
        print(f"   • أولوية المستويات: {' → '.join(analysis.search_strategy['level_priority'])}")
        print(f"   • أوضاع البحث: {', '.join(analysis.search_strategy['search_modes'])}")

        if verbose:
            print(f"\n📊 معلومات إضافية:")
            for key, value in analysis.metadata.items():
                print(f"   • {key}: {value}")

        print("\n" + "="*70)

    def to_json(self, analysis: QueryAnalysis) -> str:
        """تحويل التحليل إلى JSON"""
        return json.dumps(asdict(analysis), ensure_ascii=False, indent=2)


def main():
    """أمثلة على استخدام QueryAnalyzer"""

    analyzer = QueryAnalyzer()

    # أمثلة متنوعة
    test_queries = [
        "من هو الشريف المرتضى؟",
        "ما هو تعريف الإمامة في الفكر الشيعي؟",
        "اشرح بالتفصيل مفهوم العصمة",
        "ما الفرق بين الإمامة والخلافة؟",
        "اذكر أنواع الأدلة على الإمامة",
        "What is Imamah in Islamic theology?",
        "كيف يُثبت وجوب الإمامة؟"
    ]

    print("\n" + "="*70)
    print("🚀 Step 4: Query Analyzer - اختبار محلل الأسئلة")
    print("="*70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'─'*70}")
        print(f"اختبار {i}/{len(test_queries)}")
        print(f"{'─'*70}")

        # تحليل السؤال
        analysis = analyzer.analyze(query)

        # طباعة النتيجة
        analyzer.print_analysis(analysis)

    print("\n\n" + "="*70)
    print("✅ اكتمل الاختبار بنجاح!")
    print("="*70)


if __name__ == "__main__":
    main()
