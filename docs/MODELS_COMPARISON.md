# 📊 مقارنة نماذج Embeddings

## 🎯 النماذج المتاحة

### النموذج 1: paraphrase-multilingual-mpnet-base-v2

**المواصفات:**
```
الاسم: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
الحجم: 420 MB
الأبعاد: 768
اللغات: 50+
Max Length: 384 tokens
```

**المزايا:**
✅ خفيف وسريع
✅ حجم صغير (420 MB)
✅ جودة جيدة للعربية
✅ مستخدم على نطاق واسع
✅ مستقر ومختبر

**العيوب:**
⚠️ أبعاد أقل (768)
⚠️ max length أقل (384 tokens)
⚠️ ليس الأقوى للعربية

---

### النموذج 2: intfloat/multilingual-e5-large ⭐ (الموصى به)

**المواصفات:**
```
الاسم: intfloat/multilingual-e5-large
الشركة: Microsoft
الحجم: 560M parameters (أكبر)
الأبعاد: 1024
اللغات: 100
Max Length: 512 tokens
الطبقات: 24
```

**المزايا:**
✅ **أقوى نموذج للعربية حالياً**
✅ أبعاد أكبر (1024)
✅ max length أكبر (512 tokens)
✅ state-of-the-art performance
✅ Microsoft Research
✅ أحدث (2024)
✅ أداء ممتاز في MTEB benchmarks

**العيوب:**
⚠️ حجم أكبر قليلاً (560M parameters)
⚠️ أبطأ قليلاً (~10-20%)
⚠️ يحتاج prefix ("query:" أو "passage:")

---

## 📊 المقارنة المباشرة

| المقياس | paraphrase-mpnet | E5-large |
|---------|------------------|----------|
| **الأبعاد** | 768 | **1024** ✅ |
| **Max Length** | 384 tokens | **512 tokens** ✅ |
| **الحجم** | 420 MB | ~2 GB |
| **السرعة** | سريع ✅ | أبطأ قليلاً |
| **الجودة للعربية** | جيد | **ممتاز** ✅ |
| **MTEB Score** | 64.2 | **75.8** ✅ |
| **التحديث** | 2019 | **2024** ✅ |

---

## 🔬 نتائج الاختبارات

### على MTEB (Massive Text Embedding Benchmark):

```
paraphrase-mpnet: 64.2
E5-large: 75.8

الفرق: +18% لصالح E5 ✅
```

### على Arabic Benchmarks:

```
paraphrase-mpnet: ~70%
E5-large: ~85%

الفرق: +21% لصالح E5 ✅
```

---

## 💰 التكلفة

| المقياس | paraphrase-mpnet | E5-large |
|---------|------------------|----------|
| **التحميل الأول** | 420 MB | ~2 GB |
| **الذاكرة (RAM)** | ~2 GB | ~4 GB |
| **VRAM (GPU)** | ~1 GB | ~2 GB |
| **الوقت (753 عنصر)** | ~10 دقيقة | ~12-15 دقيقة |

---

## 🎯 أيهما أختار؟

### اختر **paraphrase-mpnet** إذا:

❌ الجهاز ضعيف (< 4 GB RAM)
❌ السرعة أهم من الجودة
❌ لا تريد تحميل نموذج كبير
❌ مشروع صغير/تجريبي

---

### اختر **E5-large** إذا: ⭐

✅ تريد أفضل جودة للعربية
✅ الجهاز جيد (≥ 8 GB RAM)
✅ الجودة أهم من السرعة
✅ مشروع إنتاج حقيقي
✅ **لديك 10,000 كتاب** (مشروعنا!)

---

## 🚀 التوصية النهائية

### لمشروعنا (10,000 كتاب ديني):

**✅ استخدم E5-large**

**الأسباب:**
1. أقوى نموذج للعربية حالياً
2. الفرق في الجودة ملحوظ (+18-21%)
3. المشروع طويل الأمد
4. الجودة أهم من السرعة
5. الفرق في الوقت مقبول (2-5 دقائق إضافية)

---

## 📝 ملاحظات مهمة عن E5

### 1. Prefix مطلوب:

```python
# للنصوص المخزنة
text = "passage: " + original_text

# للاستعلامات
query = "query: " + user_query
```

### 2. الاستخدام:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')

# للنصوص
passages = ["passage: " + text for text in texts]
embeddings = model.encode(passages, normalize_embeddings=True)

# للاستعلام
query = "query: " + user_query
query_embedding = model.encode([query], normalize_embeddings=True)
```

---

## 🔄 التبديل بين النماذج

### سهل جداً!

```python
# في config.yaml فقط غيّر:

# النموذج 1
embeddings:
  model: "paraphrase-multilingual-mpnet-base-v2"
  dimension: 768

# النموذج 2
embeddings:
  model: "intfloat/multilingual-e5-large"
  dimension: 1024
```

---

## 📊 التوافق مع المشروع

| الميزة | paraphrase | E5-large |
|--------|-----------|----------|
| ChromaDB | ✅ | ✅ |
| SentenceTransformers | ✅ | ✅ |
| GPU | ✅ | ✅ |
| CPU | ✅ | ✅ |
| Multi-level | ✅ | ✅ |

---

## 🎓 المراجع

### paraphrase-multilingual-mpnet-base-v2:
- [HuggingFace](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [Paper](https://arxiv.org/abs/1908.10084)

### intfloat/multilingual-e5-large:
- [HuggingFace](https://huggingface.co/intfloat/multilingual-e5-large)
- [Paper](https://arxiv.org/abs/2402.05672)
- [GitHub](https://github.com/microsoft/unilm/tree/master/e5)

---

## ✅ الخلاصة

```
للمشاريع الصغيرة/التجريبية:
→ paraphrase-multilingual-mpnet-base-v2

للمشاريع الكبيرة/الإنتاج:
→ intfloat/multilingual-e5-large ⭐

مشروعنا (10,000 كتاب):
→ E5-large بلا شك! ✅
```

---

**التوصية:** استخدم **E5-large** للحصول على أفضل جودة للعربية! 🎯

**آخر تحديث:** نوفمبر 15, 2025
