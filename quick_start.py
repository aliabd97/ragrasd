"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 دليل البداية السريع - Quick Start Guide
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ابدأ استخدام نظام RAG بالذكاء الاصطناعي في 5 دقائق!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(__doc__)

import os
import sys

def check_step(step_num, description, check_func):
    """التحقق من خطوة"""
    print(f"\n{'─'*70}")
    print(f"الخطوة {step_num}: {description}")
    print(f"{'─'*70}")

    result, message = check_func()

    if result:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

    return result


def check_database():
    """التحقق من قاعدة البيانات"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="data/database/chroma_db")
        collections = client.list_collections()

        if len(collections) == 0:
            return False, "قاعدة البيانات فارغة! شغّل: python build/step3_embeddings_E5.py"

        for col in collections:
            count = col.count()
            if count > 0:
                return True, f"قاعدة البيانات جاهزة! ({count} عنصر في '{col.name}')"

        return False, "قاعدة البيانات موجودة لكنها فارغة"

    except Exception as e:
        return False, f"خطأ: {str(e)}\nتأكد من تثبيت chromadb: pip install chromadb"


def check_embeddings_model():
    """التحقق من نموذج Embeddings"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        return True, "نموذج E5 Embeddings جاهز"
    except Exception as e:
        return False, f"خطأ: {str(e)}\nتأكد من: pip install sentence-transformers"


def check_llm_keys():
    """التحقق من API keys"""
    keys_found = []

    # تحميل .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

    if os.getenv("OPENAI_API_KEY"):
        keys_found.append("OpenAI")

    if os.getenv("GEMINI_API_KEY"):
        keys_found.append("Gemini")

    if os.getenv("ANTHROPIC_API_KEY"):
        keys_found.append("Claude")

    if keys_found:
        return True, f"API Keys موجودة: {', '.join(keys_found)}"
    else:
        return False, "لم يتم العثور على API keys\n   💡 يمكنك استخدام النظام بدون AI (سيستخدم القواعد)"


def main():
    """الدليل الرئيسي"""

    print("\n🔍 فحص النظام...\n")

    # الخطوة 1: قاعدة البيانات
    db_ok = check_step(1, "التحقق من قاعدة البيانات", check_database)

    # الخطوة 2: نموذج Embeddings
    model_ok = check_step(2, "التحقق من نموذج Embeddings", check_embeddings_model)

    # الخطوة 3: API Keys
    llm_ok = check_step(3, "التحقق من API Keys للـ LLM", check_llm_keys)

    # النتيجة
    print(f"\n{'='*70}")
    print("📊 ملخص الحالة")
    print(f"{'='*70}\n")

    if db_ok and model_ok:
        print("✅ النظام جاهز للاستخدام!")

        print(f"\n{'─'*70}")
        print("🎯 ماذا تريد أن تجرب؟")
        print(f"{'─'*70}\n")

        if llm_ok:
            print("1️⃣  نظام RAG بالذكاء الاصطناعي (مع LLM)")
            print("   python quick_test_ai.py")
            print()
            print("2️⃣  نظام RAG القائم على القواعد (بدون LLM)")
            print("   python quick_test_basic.py")
            print()
            print("3️⃣  أمثلة شاملة للـ AI Analyzer")
            print("   python example_ai_analyzer.py")
        else:
            print("1️⃣  نظام RAG القائم على القواعد (موصى به)")
            print("   python quick_test_basic.py")
            print()
            print("2️⃣  إعداد LLM (اختياري)")
            print("   - انسخ: cp .env.example .env")
            print("   - عدّل .env وأضف API key")
            print("   - احصل على Gemini key مجاني: https://makersuite.google.com/")

        print(f"\n{'─'*70}")
        print("📚 مراجع مفيدة:")
        print(f"{'─'*70}\n")
        print("• التوثيق الكامل للـ AI: AI_POWERED_README.md")
        print("• التوثيق الكامل للنظام القديم: STEP4_5_README.md")
        print("• أمثلة الاستخدام: example_ai_analyzer.py")

    else:
        print("⚠️  يوجد مشاكل تحتاج للحل:")
        print()

        if not db_ok:
            print("❌ قاعدة البيانات:")
            print("   python build/step3_embeddings_E5.py")
            print()

        if not model_ok:
            print("❌ نموذج Embeddings:")
            print("   pip install sentence-transformers torch")
            print()

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
