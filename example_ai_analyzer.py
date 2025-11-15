"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📘 مثال: استخدام AI Query Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

# إضافة مسار build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

# تحميل المتغيرات من ملف .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv غير مثبت. سنستخدم متغيرات البيئة مباشرة.")

from step4_ai_query_analyzer import AIQueryAnalyzer


def example_1_auto_provider():
    """
    مثال 1: استخدام المزود التلقائي
    سيختار النظام أول مزود متاح
    """
    print("\n" + "="*70)
    print("📘 مثال 1: استخدام المزود التلقائي (auto)")
    print("="*70 + "\n")

    # إنشاء محلل مع اختيار تلقائي
    analyzer = AIQueryAnalyzer(
        provider="auto",           # اختيار تلقائي
        fallback_to_rules=True     # الرجوع للقواعد إذا فشل AI
    )

    # تحليل سؤال
    query = "من هو الشريف المرتضى وما هي أهم مؤلفاته؟"
    analysis = analyzer.analyze(query)

    # عرض النتيجة
    analyzer.print_analysis(analysis)


def example_2_specific_provider():
    """
    مثال 2: استخدام مزود محدد (OpenAI)
    """
    print("\n" + "="*70)
    print("📘 مثال 2: استخدام OpenAI GPT-4")
    print("="*70 + "\n")

    # التحقق من وجود API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY غير موجود في البيئة")
        print("   أضفه إلى ملف .env أو استخدم:")
        print("   export OPENAI_API_KEY='your-key'")
        return

    # إنشاء محلل مع OpenAI
    analyzer = AIQueryAnalyzer(
        provider="openai",
        model="gpt-4-turbo-preview"
    )

    # تحليل سؤال معقد
    query = "اشرح بالتفصيل الفرق بين مفهوم الإمامة عند الشيعة والخلافة عند السنة مع ذكر الأدلة"
    analysis = analyzer.analyze(query)

    # عرض النتيجة
    analyzer.print_analysis(analysis)


def example_3_gemini():
    """
    مثال 3: استخدام Google Gemini
    """
    print("\n" + "="*70)
    print("📘 مثال 3: استخدام Google Gemini")
    print("="*70 + "\n")

    # التحقق من وجود API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY غير موجود في البيئة")
        print("   احصل على مفتاح مجاني من: https://makersuite.google.com/")
        return

    # إنشاء محلل مع Gemini
    analyzer = AIQueryAnalyzer(
        provider="gemini",
        model="gemini-pro"
    )

    # تحليل سؤال
    query = "ما هي أنواع الأدلة على الإمامة؟"
    analysis = analyzer.analyze(query)

    # عرض النتيجة
    analyzer.print_analysis(analysis)


def example_4_comparison():
    """
    مثال 4: مقارنة بين المزودين
    """
    print("\n" + "="*70)
    print("📘 مثال 4: مقارنة بين المزودين")
    print("="*70 + "\n")

    query = "ما هو تعريف الإمامة في الفكر الشيعي؟"

    providers = []

    # إضافة المزودين المتاحين
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", "gpt-3.5-turbo"))

    if os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini", "gemini-pro"))

    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("claude", "claude-3-haiku-20240307"))

    if not providers:
        print("⚠️  لم يتم العثور على أي API keys")
        print("   أضف واحد على الأقل إلى ملف .env")
        return

    # تحليل بكل مزود
    for provider, model in providers:
        print(f"\n{'─'*70}")
        print(f"🤖 تحليل باستخدام: {provider} ({model})")
        print(f"{'─'*70}\n")

        try:
            analyzer = AIQueryAnalyzer(provider=provider, model=model)
            analysis = analyzer.analyze(query)

            # عرض ملخص
            print(f"📊 الثقة: {analysis.confidence:.0%}")
            print(f"🎯 النوع: {analysis.query_type}")
            print(f"💡 التفسير: {analysis.ai_interpretation[:100]}...")

        except Exception as e:
            print(f"❌ خطأ: {str(e)}")


def example_5_interactive():
    """
    مثال 5: وضع تفاعلي
    """
    print("\n" + "="*70)
    print("📘 مثال 5: وضع تفاعلي")
    print("="*70 + "\n")

    # إنشاء محلل
    analyzer = AIQueryAnalyzer(provider="auto", fallback_to_rules=True)

    print("💬 اكتب أسئلتك (اكتب 'exit' للخروج)\n")

    while True:
        query = input("❓ السؤال: ").strip()

        if query.lower() in ['exit', 'خروج', 'quit', 'q']:
            print("\n👋 إلى اللقاء!")
            break

        if not query:
            continue

        try:
            analysis = analyzer.analyze(query)
            analyzer.print_analysis(analysis, verbose=False)
        except Exception as e:
            print(f"❌ خطأ: {str(e)}\n")


def main():
    """القائمة الرئيسية"""

    print("\n" + "="*70)
    print("🤖 أمثلة AI Query Analyzer")
    print("="*70 + "\n")

    examples = {
        "1": ("استخدام المزود التلقائي", example_1_auto_provider),
        "2": ("استخدام OpenAI GPT-4", example_2_specific_provider),
        "3": ("استخدام Google Gemini", example_3_gemini),
        "4": ("مقارنة بين المزودين", example_4_comparison),
        "5": ("وضع تفاعلي", example_5_interactive),
    }

    print("اختر مثالاً:\n")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print(f"\n  0. الخروج\n")

    choice = input("اختيارك: ").strip()

    if choice == "0":
        print("\n👋 إلى اللقاء!")
        return

    if choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("\n❌ اختيار غير صحيح")


if __name__ == "__main__":
    main()
