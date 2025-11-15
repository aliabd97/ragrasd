# 🔧 Step 3: تعليمات التثبيت والتشغيل

## 📋 قبل البدء

### المتطلبات الأساسية:

✅ Python 3.8 أو أحدث  
✅ pip محدّث  
✅ 2 GB RAM على الأقل  
✅ 1 GB مساحة فارغة  
✅ Step 2 مكتمل (753 عنصر جاهز)

---

## 🚀 التثبيت السريع (5 دقائق)

### 1. تثبيت المكتبات

```bash
# تحديث pip
pip install --upgrade pip

# تثبيت المكتبات الأساسية
pip install sentence-transformers==2.2.2
pip install chromadb==0.4.18
pip install pyyaml==6.0.1
pip install tqdm==4.66.1

# أو استخدم requirements.txt
pip install -r requirements.txt
```

---

### 2. التحقق من التثبيت

```bash
# تحقق من sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers')"

# تحقق من chromadb
python -c "import chromadb; print('✅ chromadb')"

# تحقق من yaml
python -c "import yaml; print('✅ yaml')"

# تحقق من tqdm
python -c "from tqdm import tqdm; print('✅ tqdm')"
```

**النتيجة المتوقعة:**
```
✅ sentence-transformers
✅ chromadb
✅ yaml
✅ tqdm
```

---

## 📂 التحقق من البيانات

```bash
# تحقق من وجود الملفات
ls -lh data/processed/documents.json
ls -lh data/processed/sections.json
ls -lh data/processed/paragraphs.json

# عد العناصر
python -c "
import json
docs = json.load(open('data/processed/documents.json'))
secs = json.load(open('data/processed/sections.json'))
paras = json.load(open('data/processed/paragraphs.json'))
print(f'Documents: {len(docs)}')
print(f'Sections: {len(secs)}')
print(f'Paragraphs: {len(paras)}')
print(f'Total: {len(docs) + len(secs) + len(paras)}')
"
```

**النتيجة المتوقعة:**
```
Documents: 4
Sections: 315
Paragraphs: 434
Total: 753
```

---

## 🎯 التشغيل (10-20 دقيقة)

### الطريقة 1: تشغيل مباشر

```bash
# انتقل لمجلد المشروع
cd /path/to/project

# تشغيل السكريبت
python build/step3_embeddings.py
```

---

### الطريقة 2: تشغيل مع متابعة

```bash
# تشغيل مع حفظ الـ output
python build/step3_embeddings.py | tee step3_output.log
```

---

### الطريقة 3: تشغيل في الخلفية

```bash
# للأجهزة البطيئة
nohup python build/step3_embeddings.py > step3.log 2>&1 &

# متابعة التقدم
tail -f step3.log
```

---

## 📊 متابعة التقدم

### Progress Bars:

السكريبت يعرض progress bars لكل مرحلة:

```
📂 المرحلة 4: معالجة Documents
----------------------------------------------------------------------
🔢 توليد embeddings لـ 4 document...
100%|████████████████████████████| 4/4 [00:05<00:00,  1.25s/it]

📂 المرحلة 5: معالجة Sections
----------------------------------------------------------------------
🔢 توليد embeddings لـ 315 section...
100%|██████████████████████| 315/315 [05:15<00:00,  1.00it/s]

📂 المرحلة 6: معالجة Paragraphs
----------------------------------------------------------------------
🔢 توليد embeddings لـ 434 paragraph...
100%|██████████████████████| 434/434 [07:10<00:00,  1.01it/s]
```

---

## ✅ التحقق من النجاح

### 1. تحقق من الملفات الناتجة:

```bash
# قاعدة البيانات
ls -lh data/database/chroma_db/

# الإحصائيات
cat data/database/embeddings_stats.json | python -m json.tool
```

---

### 2. تشغيل الاختبارات:

```bash
python build/test_embeddings.py
```

**النتيجة المتوقعة:**
```
======================================================================
✅ نجحت جميع الاختبارات!
======================================================================
🎉 قاعدة البيانات جاهزة للاستخدام!
```

---

### 3. اختبار بحث يدوي:

```bash
python -c "
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path='data/database/chroma_db')
collection = client.get_collection('islamic_books')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

query = 'الإمامة'
embedding = model.encode([query], normalize_embeddings=True)[0]
results = collection.query(query_embeddings=[embedding.tolist()], n_results=3)

print('✅ البحث يعمل!')
print(f'النتائج: {len(results[\"ids\"][0])}')
"
```

---

## ❌ حل المشاكل

### مشكلة 1: "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers==2.2.2
```

---

### مشكلة 2: "No module named 'chromadb'"

```bash
pip install chromadb==0.4.18
```

---

### مشكلة 3: "torch not found" أو "torch CPU version"

```bash
# CPU version (الافتراضي)
pip install torch==2.1.0

# GPU version (إذا كان لديك CUDA)
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

### مشكلة 4: بطء شديد

```yaml
# عدّل config.yaml:
embeddings:
  batch_size: 8  # قلل من 32 إلى 8
  device: "cpu"
```

أو استخدم GPU:

```yaml
embeddings:
  device: "cuda"  # بدلاً من cpu
```

---

### مشكلة 5: "CUDA out of memory"

```yaml
# قلل batch_size أكثر:
embeddings:
  batch_size: 4
```

---

### مشكلة 6: "Collection already exists"

```bash
# احذف قاعدة البيانات القديمة
rm -rf data/database/chroma_db

# أعد التشغيل
python build/step3_embeddings.py
```

---

### مشكلة 7: "Files not found"

```bash
# تأكد من تشغيل Step 2 أولاً
python build/step2_multilevel_chunking.py

# ثم أعد تشغيل Step 3
python build/step3_embeddings.py
```

---

## 🔧 التخصيص

### تغيير batch_size:

```yaml
# في config.yaml
embeddings:
  batch_size: 16  # القيمة الافتراضية: 32
```

**التأثير:**
- أصغر (8): أبطأ، لكن يستهلك ذاكرة أقل
- أكبر (64): أسرع، لكن يحتاج ذاكرة أكثر

---

### استخدام GPU:

```yaml
# في config.yaml
embeddings:
  device: "cuda"  # بدلاً من "cpu"
```

**التأثير:**
- السرعة: 10-20x أسرع
- الوقت: ~1-2 دقيقة بدلاً من 10-20

---

### تغيير النموذج:

```yaml
# في config.yaml (غير موصى به)
embeddings:
  model: "LaBSE"  # بديل
```

**ملاحظة:** النموذج الحالي (paraphrase-multilingual-mpnet-base-v2) هو الأفضل

---

## 📊 الأداء المتوقع

### CPU (عادي):

```
الوقت: 12-15 دقيقة
السرعة: ~1 عنصر/ثانية
الذاكرة: ~2 GB
```

---

### CPU (قوي):

```
الوقت: 6-8 دقائق
السرعة: ~2 عنصر/ثانية
الذاكرة: ~2 GB
```

---

### GPU:

```
الوقت: 1-2 دقيقة
السرعة: ~10-20 عنصر/ثانية
الذاكرة: ~4 GB VRAM
```

---

## 💾 حجم البيانات

```
النموذج (cache): 420 MB (يُحمل مرة واحدة)
قاعدة البيانات: 100-200 MB
الإحصائيات: <1 MB
الإجمالي: ~520-620 MB
```

---

## 🎯 الخطوة التالية

بعد إكمال Step 3 بنجاح:

✅ **لديك:**
- 753 embeddings
- ChromaDB جاهزة
- بحث يعمل

⏳ **الخطوة التالية:**
- Step 4: Query Analyzer
- تحليل الأسئلة بالـ AI

---

## 📞 المساعدة

**مشاكل؟**

1. راجع الـ logs
2. شغّل test_embeddings.py
3. تحقق من config.yaml
4. GitHub Issues

---

**آخر تحديث:** نوفمبر 15, 2025  
**النسخة:** 1.0  
**الحالة:** ✅ جاهز
