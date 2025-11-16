"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Step 4 AI: Query Analyzer المدعوم بالذكاء الاصطناعي
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
محلل أسئلة ذكي يستخدم LLMs (GPT-4, Gemini, Claude)

المهام:
1. تحليل الأسئلة باستخدام AI
2. استخراج معلومات دقيقة من السياق
3. تحديد استراتيجية بحث ذكية
4. دعم متعدد لنماذج LLM

الإصدار: 2.0.0 (AI-Powered)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# استيراد مكتبات LLM
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class AIQueryAnalysis:
    """نتيجة تحليل السؤال بواسطة AI"""

    # السؤال الأصلي
    original_query: str

    # اللغة المكتشفة
    language: Literal["arabic", "english", "mixed"]

    # نوع السؤال
    query_type: Literal[
        "factual",        # سؤال حقيقي
        "definition",     # تعريف
        "explanation",    # شرح
        "comparison",     # مقارنة
        "opinion",        # رأي
        "list",           # قائمة
        "procedural"      # إجرائي
    ]

    # الكلمات المفتاحية
    keywords: list[str]

    # الموضوع الرئيسي
    main_topic: str

    # المواضيع الفرعية
    sub_topics: list[str]

    # مستوى التفصيل
    detail_level: Literal["brief", "moderate", "detailed"]

    # مستوى التعقيد
    complexity: Literal["simple", "moderate", "complex"]

    # استراتيجية البحث المقترحة (من AI)
    search_strategy: Dict[str, Any]

    # تفسير AI للسؤال
    ai_interpretation: str

    # الثقة في التحليل
    confidence: float

    # النموذج المستخدم
    model_used: str

    # معلومات إضافية
    metadata: Dict[str, Any]

    # وقت التحليل
    timestamp: str


class AIQueryAnalyzer:
    """محلل أسئلة ذكي مدعوم بـ LLMs"""

    def __init__(
        self,
        provider: Literal["openai", "gemini", "claude", "auto"] = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_to_rules: bool = True
    ):
        """
        تهيئة محلل الأسئلة بالذكاء الاصطناعي

        Args:
            provider: مزود الخدمة (openai/gemini/claude/auto)
            model: اسم النموذج (اختياري، سيستخدم الافتراضي)
            api_key: مفتاح API (اختياري، سيستخدم من البيئة)
            fallback_to_rules: الرجوع للتحليل القائم على القواعد في حالة الفشل
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.fallback_to_rules = fallback_to_rules

        # تهيئة المزود المناسب
        self._initialize_provider()

        # Prompt template للتحليل
        self.analysis_prompt = self._create_analysis_prompt()

    def _initialize_provider(self):
        """تهيئة مزود LLM"""

        if self.provider == "auto":
            # اختيار تلقائي بناءً على المتاح
            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            elif ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "claude"
            else:
                print("⚠️  لم يتم العثور على API key لأي مزود LLM")
                print("   سيتم استخدام التحليل القائم على القواعد")
                self.provider = "rules"
                return

        # تهيئة OpenAI
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Run: pip install openai")

            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            openai.api_key = api_key
            self.model = self.model or "gpt-4-turbo-preview"
            print(f"✅ تم تهيئة OpenAI ({self.model})")

        # تهيئة Gemini
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google AI library not installed. Run: pip install google-generativeai")

            api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            genai.configure(api_key=api_key)
            # استخدام النموذج الجديد gemini-1.5-flash-latest (مجاني وسريع)
            self.model = self.model or "gemini-1.5-flash-latest"
            self.gemini_model = genai.GenerativeModel(self.model)
            print(f"✅ تم تهيئة Google Gemini ({self.model})")

        # تهيئة Claude
        elif self.provider == "claude":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")

            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            self.claude_client = anthropic.Anthropic(api_key=api_key)
            # استخدام أحدث نموذج Claude Sonnet 4.5
            self.model = self.model or "claude-sonnet-4-5-20250929"
            print(f"✅ تم تهيئة Anthropic Claude ({self.model})")

    def _create_analysis_prompt(self) -> str:
        """إنشاء prompt للتحليل"""
        return """أنت محلل أسئلة خبير متخصص في الأسئلة الدينية الإسلامية.

مهمتك: تحليل السؤال التالي وتقديم معلومات دقيقة عنه.

السؤال: {query}

قم بتحليل السؤال وأعطني النتيجة بصيغة JSON التالية:

{{
    "language": "arabic أو english أو mixed",
    "query_type": "factual أو definition أو explanation أو comparison أو opinion أو list أو procedural",
    "keywords": ["كلمة1", "كلمة2", ...],
    "main_topic": "الموضوع الرئيسي للسؤال",
    "sub_topics": ["موضوع فرعي 1", "موضوع فرعي 2", ...],
    "detail_level": "brief أو moderate أو detailed",
    "complexity": "simple أو moderate أو complex",
    "ai_interpretation": "تفسيرك للسؤال وما يبحث عنه السائل",
    "confidence": 0.95,
    "search_strategy": {{
        "n_results": 5,
        "level_priority": ["paragraph", "section", "document"],
        "search_focus": "وصف ما يجب التركيز عليه في البحث",
        "suggested_filters": ["فلتر1", "فلتر2", ...]
    }}
}}

ملاحظات مهمة:
1. الكلمات المفتاحية يجب أن تكون الكلمات الجوهرية فقط (بدون كلمات السؤال مثل: من، ماذا، ما)
2. query_type يحدد نوع السؤال بدقة
3. detail_level يعتمد على وجود كلمات مثل "بالتفصيل" أو "باختصار"
4. search_strategy يجب أن يكون ذكياً ويناسب نوع السؤال
5. confidence هو مدى ثقتك في التحليل (0-1)

أعطني فقط JSON بدون أي نص إضافي."""

    def analyze(self, query: str) -> AIQueryAnalysis:
        """
        تحليل السؤال باستخدام AI

        Args:
            query: السؤال المراد تحليله

        Returns:
            AIQueryAnalysis: نتيجة التحليل
        """
        print(f"🤖 تحليل السؤال باستخدام AI ({self.provider})...")

        try:
            # استدعاء LLM المناسب
            if self.provider == "openai":
                result = self._analyze_with_openai(query)
            elif self.provider == "gemini":
                result = self._analyze_with_gemini(query)
            elif self.provider == "claude":
                result = self._analyze_with_claude(query)
            else:
                if self.fallback_to_rules:
                    print("⚠️  استخدام التحليل القائم على القواعد...")
                    return self._analyze_with_rules(query)
                else:
                    raise ValueError(f"Provider غير مدعوم: {self.provider}")

            # معالجة النتيجة
            analysis_data = self._parse_llm_response(result, query)

            # إنشاء AIQueryAnalysis
            return AIQueryAnalysis(
                original_query=query,
                language=analysis_data['language'],
                query_type=analysis_data['query_type'],
                keywords=analysis_data['keywords'],
                main_topic=analysis_data['main_topic'],
                sub_topics=analysis_data['sub_topics'],
                detail_level=analysis_data['detail_level'],
                complexity=analysis_data['complexity'],
                search_strategy=analysis_data['search_strategy'],
                ai_interpretation=analysis_data['ai_interpretation'],
                confidence=analysis_data['confidence'],
                model_used=f"{self.provider}/{self.model}",
                metadata={
                    'query_length': len(query),
                    'word_count': len(query.split()),
                    'has_question_mark': '?' in query or '؟' in query
                },
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"❌ خطأ في التحليل: {str(e)}")

            if self.fallback_to_rules:
                print("⚠️  الرجوع للتحليل القائم على القواعد...")
                return self._analyze_with_rules(query)
            else:
                raise

    def _analyze_with_openai(self, query: str) -> str:
        """تحليل باستخدام OpenAI"""
        prompt = self.analysis_prompt.format(query=query)

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "أنت محلل أسئلة خبير. أجب دائماً بصيغة JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def _analyze_with_gemini(self, query: str) -> str:
        """تحليل باستخدام Gemini"""
        prompt = self.analysis_prompt.format(query=query)

        response = self.gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3
            }
        )

        return response.text

    def _analyze_with_claude(self, query: str) -> str:
        """تحليل باستخدام Claude"""
        prompt = self.analysis_prompt.format(query=query)

        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    def _parse_llm_response(self, response: str, query: str) -> Dict[str, Any]:
        """معالجة استجابة LLM"""
        try:
            # استخراج JSON من الاستجابة
            # قد يكون هناك نص إضافي قبل/بعد JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group()

            data = json.loads(response)

            # التحقق من الحقول المطلوبة
            required_fields = [
                'language', 'query_type', 'keywords', 'main_topic',
                'sub_topics', 'detail_level', 'complexity',
                'ai_interpretation', 'confidence', 'search_strategy'
            ]

            for field in required_fields:
                if field not in data:
                    raise ValueError(f"حقل مفقود: {field}")

            return data

        except Exception as e:
            print(f"⚠️  خطأ في معالجة استجابة LLM: {str(e)}")
            # fallback للتحليل القائم على القواعد
            raise

    def _analyze_with_rules(self, query: str) -> AIQueryAnalysis:
        """
        تحليل قائم على القواعد (fallback)
        استخدام النسخة القديمة من Query Analyzer
        """
        from step4_query_analyzer import QueryAnalyzer

        old_analyzer = QueryAnalyzer()
        old_analysis = old_analyzer.analyze(query)

        # تحويل إلى AIQueryAnalysis
        return AIQueryAnalysis(
            original_query=query,
            language=old_analysis.language,
            query_type=old_analysis.query_type,
            keywords=old_analysis.keywords,
            main_topic=old_analysis.keywords[0] if old_analysis.keywords else "unknown",
            sub_topics=old_analysis.keywords[1:3] if len(old_analysis.keywords) > 1 else [],
            detail_level=old_analysis.detail_level,
            complexity="moderate",  # افتراضي
            search_strategy=old_analysis.search_strategy,
            ai_interpretation="تحليل قائم على القواعد",
            confidence=old_analysis.query_type_confidence,
            model_used="rules-based",
            metadata=old_analysis.metadata,
            timestamp=old_analysis.timestamp
        )

    def print_analysis(self, analysis: AIQueryAnalysis, verbose: bool = True):
        """طباعة نتيجة التحليل بشكل منسق"""

        print("\n" + "="*70)
        print("🤖 تحليل السؤال بالذكاء الاصطناعي")
        print("="*70)

        print(f"\n📝 السؤال: {analysis.original_query}")
        print(f"🤖 النموذج: {analysis.model_used}")
        print(f"📊 الثقة: {analysis.confidence:.0%}")

        print(f"\n🌐 اللغة: {analysis.language}")
        print(f"📋 نوع السؤال: {analysis.query_type}")
        print(f"📏 مستوى التفصيل: {analysis.detail_level}")
        print(f"🎯 مستوى التعقيد: {analysis.complexity}")

        print(f"\n💡 تفسير AI:")
        print(f"   {analysis.ai_interpretation}")

        print(f"\n🎯 الموضوع الرئيسي: {analysis.main_topic}")

        if analysis.sub_topics:
            print(f"\n📌 المواضيع الفرعية:")
            for i, topic in enumerate(analysis.sub_topics, 1):
                print(f"   {i}. {topic}")

        if analysis.keywords:
            print(f"\n🔑 الكلمات المفتاحية:")
            for i, kw in enumerate(analysis.keywords, 1):
                print(f"   {i}. {kw}")

        print(f"\n🎯 استراتيجية البحث:")
        strategy = analysis.search_strategy
        print(f"   • عدد النتائج: {strategy.get('n_results', 5)}")
        print(f"   • أولوية المستويات: {' → '.join(strategy.get('level_priority', []))}")
        if 'search_focus' in strategy:
            print(f"   • التركيز على: {strategy['search_focus']}")
        if 'suggested_filters' in strategy and strategy['suggested_filters']:
            print(f"   • فلاتر مقترحة: {', '.join(strategy['suggested_filters'])}")

        if verbose:
            print(f"\n📊 معلومات إضافية:")
            for key, value in analysis.metadata.items():
                print(f"   • {key}: {value}")

        print("\n" + "="*70 + "\n")


def main():
    """تجربة AI Query Analyzer"""

    print("\n" + "="*70)
    print("🤖 Step 4 AI: Query Analyzer المدعوم بالذكاء الاصطناعي")
    print("="*70 + "\n")

    # أمثلة على الأسئلة
    test_queries = [
        "من هو الشريف المرتضى؟",
        "ما هو تعريف الإمامة في الفكر الشيعي؟",
        "اشرح بالتفصيل مفهوم العصمة وأدلته",
        "ما الفرق بين الإمامة والخلافة؟",
    ]

    # تجربة مع auto (سيختار المتاح)
    try:
        analyzer = AIQueryAnalyzer(provider="auto", fallback_to_rules=True)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'━'*70}")
            print(f"مثال {i}/{len(test_queries)}")
            print(f"{'━'*70}\n")

            analysis = analyzer.analyze(query)
            analyzer.print_analysis(analysis)

            if i < len(test_queries):
                input("\nاضغط Enter للمثال التالي...")

    except Exception as e:
        print(f"❌ خطأ: {str(e)}")
        print("\n💡 تأكد من:")
        print("   1. تثبيت المكتبات: pip install openai google-generativeai anthropic")
        print("   2. تعيين API key في البيئة:")
        print("      export OPENAI_API_KEY='your-key'")
        print("      export GEMINI_API_KEY='your-key'")
        print("      export ANTHROPIC_API_KEY='your-key'")


if __name__ == "__main__":
    main()
