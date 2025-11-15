# 📘 Step 3: Embeddings - دليل الاستخدام

## 📋 نظرة عامة

**المهمة:** تحويل البيانات إلى embeddings وبناء ChromaDB للبحث

**المدة المتوقعة:** 10-20 دقيقة

**المتطلبات:**
- ✅ Step 2 مكتمل (documents.json, sections.json, paragraphs.json)
- ✅ Python 3.8+
- ✅ المكتبات مثبتة

---

## 🚀 التشغيل السريع

### الخطوة 1: التحقق من المتطلبات

```bash
# تحقق من وجود البيانات
ls data/processed/documents.json
ls data/processed/sections.json
ls data/processed/paragraphs.json

# تحقق من المكتبات
python -c "import chromadb; import sentence_transformers; print('✅ كل شيء جاهز')"
```

### الخطوة 2: التشغيل

```bash
# انتقل لمجلد المشروع
cd /path/to/project

# شغل السكريبت
python build/step3_embeddings.py
```

**الانتظار:** 10-20 دقيقة حسب سرعة الجهاز

---

## 📊 ما الذي يحدث؟

### المراحل الثمانية:

1. **تحميل البيانات**
   - documents.json (4 عناصر)
   - sections.json (315 عنصر)
   - paragraphs.json (434 عنصر)

2. **تهيئة Embeddings Generator**
   - تحميل paraphrase-multilingual-mpnet-base-v2
   - الحجم: 420 MB
   - الوقت: 10-30 ثانية

3. **تهيئة ChromaDB**
   - إنشاء قاعدة بيانات جديدة
   - حذف القديمة إن وجدت

4. **معالجة Documents**
   - 4 documents → 4 embeddings
   - الوقت: ~5 ثوانٍ

5. **معالجة Sections**
   - 315 sections → 315 embeddings
   - الوقت: ~2-3 دقائق

6. **معالجة Paragraphs**
   - 434 paragraphs → 434 embeddings
   - الوقت: ~3-5 دقائق

7. **الإحصائيات**
   - حفظ embeddings_stats.json

8. **اختبار البحث**
   - تجربة بحث بسيطة

---

## 📤 الناتج المتوقع

### الملفات المنشأة:

```
data/
├── database/
│   ├── chroma_db/           # قاعدة البيانات
│   │   ├── chroma.sqlite3
│   │   └── ... (ملفات ChromaDB)
│   └── embeddings_stats.json
```

### embeddings_stats.json:

```json
{
  "timestamp": "2025-11-15T...",
  "model": "paraphrase-multilingual-mpnet-base-v2",
  "embedding_dimension": 768,
  
  "data": {
    "documents": 4,
    "sections": 315,
    "paragraphs": 434,
    "total": 753
  },
  
  "database": {
    "total_items": 753,
    "documents": 4,
    "sections": 315,
    "paragraphs": 434
  },
  
  "performance": {
    "total_time_seconds": 720,
    "total_time_minutes": 12,
    "items_per_second": 1.05
  }
}
```

---

## ✅ الاختبار

```bash
# تشغيل الاختبارات
python build/test_embeddings.py
```

### الاختبارات الثمانية:

1. ✅ وجود قاعدة البيانات
2. ✅ وجود collection
3. ✅ عدد العناصر (~753)
4. ✅ أنواع البيانات (4 + 315 + 434)
5. ✅ البحث الأساسي
6. ✅ البحث متعدد المستويات
7. ✅ جودة Embeddings
8. ✅ النتائج المتشابهة

**النتيجة المتوقعة:**

```
✅ نجحت جميع الاختبارات!
🎉 قاعدة البيانات جاهزة للاستخدام!
```

---

## 🔍 اختبار البحث يدوياً

```python
import chromadb
from sentence_transformers import SentenceTransformer

# فتح قاعدة البيانات
client = chromadb.PersistentClient(path="data/database/chroma_db")
collection = client.get_collection("islamic_books")

# تحميل النموذج
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# البحث
query = "ما هي الإمامة؟"
query_embedding = model.encode([query], normalize_embeddings=True)[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

# النتائج
for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0]), 1):
    print(f"{i}. {doc_id} ({metadata['type']})")
```

---

## 🎯 البحث متعدد المستويات

### البحث في مستوى واحد:

```python
# البحث في Paragraphs فقط
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"type": "paragraph"}
)
```

### البحث في مستويين:

```python
# البحث في Sections
results_sec = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    where={"type": "section"}
)

# البحث في Paragraphs
results_para = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"type": "paragraph"}
)
```

---

## 📊 الإحصائيات المتوقعة

### السرعة:

```
الوقت الإجمالي: 10-20 دقيقة
السرعة: ~1 عنصر/ثانية
حجم قاعدة البيانات: ~100-200 MB
```

### الجودة:

```
Embedding Dimension: 768
Model: paraphrase-multilingual-mpnet-base-v2
Similarity Metric: Cosine
```

---

## ❌ حل المشاكل

### مشكلة 1: "No module named 'chromadb'"

```bash
pip install chromadb
```

### مشكلة 2: "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### مشكلة 3: بطء شديد

```bash
# استخدم GPU إذا متوفر
# عدّل في config.yaml:
embeddings:
  device: "cuda"
```

### مشكلة 4: "Memory Error"

```bash
# قلل batch_size
# عدّل في config.yaml:
embeddings:
  batch_size: 16  # بدلاً من 32
```

### مشكلة 5: "Files not found"

```bash
# تأكد من تشغيل Step 2 أولاً
python build/step2_multilevel_chunking.py
```

---

## 📝 ملاحظات مهمة

### 1. التحميل الأول:

- النموذج يُحمّل مرة واحدة فقط
- الحجم: 420 MB
- يُحفظ في: cache/embeddings/

### 2. إعادة التشغيل:

- يحذف قاعدة البيانات القديمة
- يُنشئ قاعدة جديدة
- كل مرة: 10-20 دقيقة

### 3. الأداء:

- CPU: ~1 عنصر/ثانية
- GPU: ~10-20 عنصر/ثانية

---

## 🎓 الخطوة التالية

بعد إكمال هذه الخطوة:

✅ **Step 3 مكتمل**
- قاعدة بيانات جاهزة
- 753 عنصر
- بحث يعمل

⏳ **Step 4: Query Analyzer**
- تحليل الأسئلة بالـ AI
- استراتيجيات بحث ديناميكية

---

## 📞 المساعدة

**مشاكل؟**

1. راجع الـ logs
2. شغّل test_embeddings.py
3. تحقق من config.yaml

**أسئلة؟**

- GitHub Issues
- Documentation

---

**آخر تحديث:** نوفمبر 15, 2025  
**النسخة:** 1.0
