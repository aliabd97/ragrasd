# 🚀 Step 3: Embeddings - مشروع كامل

## 📋 نظرة عامة

مشروع Python كامل لتوليد embeddings وبناء ChromaDB باستخدام **أقوى نموذج للعربية**.

**النموذج:** intfloat/multilingual-e5-large  
**الأبعاد:** 1024  
**الجودة:** ⭐⭐⭐⭐⭐

---

## 📂 بنية المشروع

```
step3_complete_project/
├── build/                          # الأكواد
│   ├── step3_embeddings_E5.py     # السكريبت الرئيسي ⭐
│   └── test_embeddings.py         # الاختبارات
│
├── data/                           # البيانات
│   ├── processed/                 # البيانات المعالجة
│   │   ├── documents.json         # 4 وثائق
│   │   ├── sections.json          # ~315 قسم
│   │   └── paragraphs.json        # ~434 فقرة
│   └── database/                  # قاعدة البيانات (ستُنشأ)
│       └── chroma_db/
│
├── docs/                           # التوثيق
│   ├── START_HERE_STEP3.md        # 🎯 ابدأ هنا!
│   ├── INSTALLATION.md            # تعليمات التثبيت
│   ├── MODELS_COMPARISON.md       # مقارنة النماذج
│   └── ... (8 ملفات توثيق أخرى)
│
├── config.yaml                     # الإعدادات
├── requirements.txt                # المكتبات المطلوبة
├── logs/                          # السجلات (ستُنشأ)
├── cache/                         # التخزين المؤقت (ستُنشأ)
└── README.md                      # هذا الملف
```

---

## 🚀 البدء السريع (5 دقائق)

### 1. التثبيت

```bash
# إنشاء بيئة افتراضية (اختياري لكن موصى به)
python -m venv venv
source venv/bin/activate  # على Linux/Mac
# أو
venv\Scripts\activate  # على Windows

# تثبيت المكتبات
pip install -r requirements.txt
```

### 2. التشغيل

```bash
# تشغيل السكريبت
python build/step3_embeddings_E5.py
```

**الوقت المتوقع:** 10-20 دقيقة

### 3. الاختبار

```bash
# تشغيل الاختبارات
python build/test_embeddings.py
```

**النتيجة المتوقعة:**
```
✅ نجحت جميع الاختبارات!
🎉 قاعدة البيانات جاهزة للاستخدام!
```

---

## 📊 المتطلبات

### النظام:
- Python 3.8+
- 8 GB RAM (4 GB كحد أدنى)
- 3 GB مساحة فارغة

### المكتبات:
- sentence-transformers
- chromadb
- pyyaml
- tqdm

---

## 🎯 ماذا يفعل؟

### المراحل الثمانية:

1. ✅ تحميل البيانات (documents, sections, paragraphs)
2. ✅ تحميل نموذج E5 (أقوى نموذج للعربية)
3. ✅ إنشاء ChromaDB
4. ✅ معالجة Documents → embeddings
5. ✅ معالجة Sections → embeddings
6. ✅ معالجة Paragraphs → embeddings
7. ✅ حفظ الإحصائيات
8. ✅ اختبار البحث

### الناتج:

```
✅ data/database/chroma_db/          (قاعدة البيانات)
✅ data/database/embeddings_stats.json  (الإحصائيات)
✅ logs/                              (السجلات)
```

---

## 📝 ملاحظات مهمة

### 1. البيانات التجريبية

المشروع يحتوي على **بيانات تجريبية** (3 عناصر فقط) للاختبار.

**للاستخدام الفعلي:**
- ضع ملفاتك الحقيقية في `data/processed/`:
  - documents.json
  - sections.json
  - paragraphs.json

### 2. النموذج E5

النموذج **intfloat/multilingual-e5-large** يحتاج:
- Prefix للنصوص: `"passage: "`
- Prefix للاستعلامات: `"query: "`

السكريبت يتعامل مع هذا تلقائياً ✅

### 3. GPU vs CPU

```yaml
# في config.yaml
embeddings:
  device: "cpu"   # الافتراضي
  # أو
  device: "cuda"  # إذا كان لديك GPU
```

**مع GPU:** أسرع 10-20 مرة!

---

## ❌ حل المشاكل

### "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### "CUDA out of memory"

```yaml
# في config.yaml قلل batch_size
embeddings:
  batch_size: 8  # بدلاً من 32
```

### "بطء شديد"

```yaml
# استخدم GPU
embeddings:
  device: "cuda"
```

### المزيد من الحلول:
راجع `docs/INSTALLATION.md`

---

## 📚 التوثيق

### للبدء السريع:
➡️ `docs/START_HERE_STEP3.md`

### للتثبيت:
➡️ `docs/INSTALLATION.md`

### لمقارنة النماذج:
➡️ `docs/MODELS_COMPARISON.md`

### للدليل الشامل:
➡️ `docs/STEP3_GUIDE.md`

---

## 🎓 الأمثلة

### استخدام قاعدة البيانات:

```python
import chromadb
from sentence_transformers import SentenceTransformer

# فتح قاعدة البيانات
client = chromadb.PersistentClient(path="data/database/chroma_db")
collection = client.get_collection("islamic_books_e5")

# تحميل النموذج
model = SentenceTransformer("intfloat/multilingual-e5-large")

# البحث
query = "query: الإمامة"  # ملاحظة: prefix مطلوب
embedding = model.encode([query], normalize_embeddings=True)[0]

results = collection.query(
    query_embeddings=[embedding.tolist()],
    n_results=5
)

# النتائج
for doc_id in results['ids'][0]:
    print(f"✅ {doc_id}")
```

---

## ✅ Checklist

- [ ] قرأت README.md
- [ ] ثبّت المكتبات
- [ ] شغّلت step3_embeddings_E5.py
- [ ] نجح التشغيل
- [ ] شغّلت test_embeddings.py
- [ ] نجحت كل الاختبارات

---

## 🎯 الخطوة التالية

بعد إكمال هذه الخطوة:

✅ **لديك:**
- قاعدة بيانات كاملة
- أفضل embeddings للعربية
- بحث سريع ودقيق

⏳ **الخطوة التالية:**
- Step 4: Query Analyzer
- تحليل الأسئلة بالـ AI

---

## 📞 الدعم

**أسئلة؟ مشاكل؟**

1. راجع `docs/` - كل شيء موثق
2. GitHub Issues
3. Email: support@example.com

---

## 📄 الترخيص

MIT License

---

**جاهز للبدء؟**

```bash
pip install -r requirements.txt
python build/step3_embeddings_E5.py
```

**حظاً موفقاً! 🎉**

---

**آخر تحديث:** نوفمبر 15, 2025  
**النسخة:** 1.0  
**الحالة:** ✅ جاهز للإنتاج
