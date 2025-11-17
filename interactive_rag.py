#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive RAG System
=====================

نظام تفاعلي للإجابة على الأسئلة باستخدام RAG
"""

import sys
import os

# إضافة مجلد build إلى path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

from step5_rag_system import RAGSystem
from dotenv import load_dotenv


def main():
    """الدالة الرئيسية"""

    print("=" * 70)
    print("🚀 نظام RAG التفاعلي للإجابة على الأسئلة الدينية")
    print("=" * 70)
    print()

    # تحميل environment variables
    load_dotenv()

    # التحقق من وجود API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ تحذير: OPENAI_API_KEY غير موجود في ملف .env")
        print("   يمكنك:")
        print("   1. إنشاء ملف .env ووضع API key فيه")
        print("   2. تعديل step5_rag_system.py لاستخدام النموذج المحلي")
        print()
        choice = input("هل تريد المتابعة بالنموذج المحلي؟ (y/n): ")
        if choice.lower() != 'y':
            return

        embeddings_provider = "sentence_transformers"
        embeddings_model = "intfloat/multilingual-e5-large"
        print("⚙️ سيتم استخدام النموذج المحلي...")
    else:
        embeddings_provider = "openai"
        embeddings_model = "text-embedding-3-small"

    # تهيئة النظام
    print("\n⚙️ تهيئة نظام RAG...")
    print("   (قد يستغرق هذا بضع ثوان...)\n")

    try:
        rag = RAGSystem(
            embeddings_provider=embeddings_provider,
            embeddings_model=embeddings_model,
            llm_provider="openai",
            llm_model="gpt-4o-mini"
        )
    except Exception as e:
        print(f"❌ خطأ في تهيئة النظام: {e}")
        print("\nتأكد من:")
        print("1. تشغيل step3_embeddings_openai.py أولاً")
        print("2. وجود قاعدة البيانات في data/database/chroma_db")
        print("3. صحة OPENAI_API_KEY في ملف .env")
        return

    print("✅ النظام جاهز!\n")
    print("=" * 70)
    print("📖 يمكنك الآن طرح أسئلتك")
    print("   - اكتب 'exit' أو 'خروج' للخروج")
    print("   - اكتب 'help' للمساعدة")
    print("=" * 70)

    # حلقة الأسئلة
    while True:
        try:
            question = input("\n🔍 سؤالك: ")

            # معالجة الأوامر الخاصة
            if question.lower() in ['exit', 'quit', 'خروج']:
                print("\n👋 وداعاً!")
                break

            if question.lower() == 'help':
                print("\n📖 المساعدة:")
                print("   - اطرح أي سؤال عن الإمامة والتشيع")
                print("   - مثال: ما هو مفهوم الإمامة؟")
                print("   - مثال: من هو الإمام الرضا؟")
                print("   - exit/خروج: للخروج من البرنامج")
                continue

            if not question.strip():
                continue

            # الحصول على الإجابة
            print("\n⏳ جاري البحث وتوليد الإجابة...")
            print("   (قد يستغرق هذا بضع ثوان...)\n")

            result = rag.query(question, n_results=10)

            # عرض الإجابة
            print("\n" + "=" * 70)
            print("📝 الإجابة:")
            print("=" * 70)
            print()
            print(result['answer'])
            print()
            print("=" * 70)
            print(f"✅ تم استخدام {result['num_sources']} مصدر")
            print(f"⏰ الوقت: {result['timestamp']}")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 تم إيقاف البرنامج. وداعاً!")
            break

        except Exception as e:
            print(f"\n❌ خطأ: {e}")
            print("   حاول مرة أخرى أو اكتب 'exit' للخروج")


if __name__ == "__main__":
    main()
