#!/bin/bash

# =============================================================================
# Step 3: Embeddings - سكريبت تشغيل سريع
# =============================================================================

echo "=========================================="
echo "🚀 Step 3: Embeddings"
echo "=========================================="
echo ""

# التحقق من Python
if ! command -v python &> /dev/null; then
    echo "❌ Python غير مثبت!"
    exit 1
fi

echo "✅ Python موجود: $(python --version)"
echo ""

# التحقق من المكتبات
echo "📦 التحقق من المكتبات..."
python -c "import sentence_transformers; import chromadb; import yaml; import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ بعض المكتبات غير مثبتة"
    echo "📦 تثبيت المكتبات..."
    pip install -r requirements.txt
fi

echo "✅ كل المكتبات موجودة"
echo ""

# التحقق من البيانات
echo "📂 التحقق من البيانات..."
if [ ! -f "data/processed/documents.json" ]; then
    echo "❌ data/processed/documents.json غير موجود!"
    exit 1
fi
if [ ! -f "data/processed/sections.json" ]; then
    echo "❌ data/processed/sections.json غير موجود!"
    exit 1
fi
if [ ! -f "data/processed/paragraphs.json" ]; then
    echo "❌ data/processed/paragraphs.json غير موجود!"
    exit 1
fi

echo "✅ كل الملفات موجودة"
echo ""

# التشغيل
echo "=========================================="
echo "🔥 بدء التشغيل..."
echo "=========================================="
echo ""

python build/step3_embeddings_E5.py

# التحقق من النجاح
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ تم بنجاح!"
    echo "=========================================="
    echo ""
    echo "📂 الملفات الناتجة:"
    echo "   - data/database/chroma_db/"
    echo "   - data/database/embeddings_stats.json"
    echo ""
    echo "🧪 الخطوة التالية: تشغيل الاختبارات"
    echo "   python build/test_embeddings.py"
    echo ""
else
    echo ""
    echo "❌ حدث خطأ!"
    echo "راجع الأخطاء أعلاه"
    echo ""
    exit 1
fi
