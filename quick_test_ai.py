"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 اختبار سريع - نظام RAG بالذكاء الاصطناعي (مع LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
يحتاج API key لـ OpenAI أو Gemini أو Claude
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os

# تحميل .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ تم تحميل .env\n")
except ImportError:
    print("⚠️  python-dotenv غير مثبت (اختياري)")
    print("   pip install python-dotenv\n")

# إضافة المسار للـ build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    from step5_ai_rag_system import AIRAGSystem
    print("✅ تم استيراد AIRAGSystem بنجاح\n")
except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("\n💡 الحل:")
    print("   تأكد أن ملف build/step5_ai_rag_system.py موجود")
    sys.exit(1)

print("="*70)
print("🤖 اختبار نظام RAG - النسخة المدعومة بالذكاء الاصطناعي")
print("="*70)
print()

# التحقق من API keys
keys_found = []
if os.getenv("OPENAI_API_KEY"):
    keys_found.append("OpenAI")
if os.getenv("GEMINI_API_KEY"):
    keys_found.append("Gemini")
if os.getenv("ANTHROPIC_API_KEY"):
    keys_found.append("Claude")

if not keys_found:
    print("⚠️  لم يتم العثور على API keys!")
    print()
    print("💡 للحصول على API key مجاني (Gemini):")
    print("   1. اذهب إلى: https://makersuite.google.com/app/apikey")
    print("   2. انسخ المفتاح")
    print("   3. أنشئ ملف .env:")
    print("      cp .env.example .env")
    print("   4. أضف: GEMINI_API_KEY=your-key-here")
    print()
    print("💡 أو استخدم النظام بدون AI:")
    print("   python quick_test_basic.py")
    sys.exit(1)

print(f"✅ API Keys موجودة: {', '.join(keys_found)}\n")

# أسئلة اختبار
test_queries = [
    "من هو الشريف المرتضى؟",
    "ما هو تعريف الإمامة في الفكر الشيعي؟",
]

try:
    # إنشاء نظام RAG بـ AI
    print("🔄 تهيئة نظام RAG بالذكاء الاصطناعي...")
    print("   (قد يستغرق دقيقة لتحميل النموذج في المرة الأولى)\n")

    rag = AIRAGSystem(
        llm_provider="auto",       # سيختار أول مزود متاح
        use_ai_analyzer=True       # استخدام AI
    )

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'━'*70}")
        print(f"سؤال {i}/{len(test_queries)}")
        print(f"{'━'*70}\n")

        response = rag.ask(query)

        if i < len(test_queries):
            input("\n⏸  اضغط Enter للسؤال التالي...")

    print("\n" + "="*70)
    print("✅ الاختبار مكتمل!")
    print("="*70)

    print("\n💡 التالي:")
    print("   - جرب أسئلتك الخاصة")
    print("   - اقرأ AI_POWERED_README.md للمزيد")
    print("   - جرب مزودين مختلفين (OpenAI, Gemini, Claude)")

except Exception as e:
    print(f"\n❌ خطأ: {type(e).__name__}")
    print(f"   {str(e)}")

    if "API" in str(e) or "key" in str(e).lower():
        print("\n💡 المشكلة قد تكون في API key:")
        print("   1. تأكد من صحة المفتاح في .env")
        print("   2. تأكد من وجود رصيد (OpenAI/Claude)")
        print("   3. جرب Gemini (مجاني): https://makersuite.google.com/")

    print("\n💡 أو استخدم النظام بدون AI:")
    print("   python quick_test_basic.py")

    import traceback
    print("\n🔍 التفاصيل الكاملة:")
    traceback.print_exc()
