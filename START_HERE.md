# 🚀 ابدأ هنا - Quick Start

## الخطوة 1: تحديث المشروع

```bash
git pull origin claude/backend-setup-01WLoo4KUJee2DiJM1qbUNdE
```

---

## الخطوة 2: تثبيت المكتبات الأساسية

```bash
pip install -r requirements.txt
```

---

## الخطوة 3: فحص النظام

```bash
python quick_start.py
```

هذا سيفحص:
- ✅ قاعدة البيانات
- ✅ نموذج Embeddings
- ✅ API Keys (اختياري)

---

## الخطوة 4: اختبار النظام

### الخيار 1: بدون AI (موصى به للبداية) ⭐

```bash
python quick_test_basic.py
```

- ✅ **لا يحتاج API keys**
- ✅ مجاني 100%
- ✅ يعمل فوراً
- ✅ دقة 85-90%

---

### الخيار 2: مع AI (دقة أعلى 95-99%)

#### 2.1 احصل على API key مجاني (Gemini)

1. اذهب إلى: https://makersuite.google.com/app/apikey
2. سجّل دخول بحساب Google
3. اضغط "Create API Key"
4. انسخ المفتاح

#### 2.2 أضف المفتاح

**Windows PowerShell:**
```powershell
cp .env.example .env
notepad .env
```

**Mac/Linux:**
```bash
cp .env.example .env
nano .env
```

**أضف في الملف:**
```env
GEMINI_API_KEY=your-key-here
```

#### 2.3 ثبت مكتبة LLM

```bash
pip install google-generativeai
```

#### 2.4 شغّل الاختبار

```bash
python quick_test_ai.py
```

---

## 🎯 الخطوة 5: جرب أسئلتك

### استخدام بسيط (بدون AI):

```python
from build.step5_rag_system import RAGSystem

rag = RAGSystem()
response = rag.ask("سؤالك هنا")
```

### استخدام مع AI:

```python
from build.step5_ai_rag_system import AIRAGSystem

rag = AIRAGSystem(llm_provider="auto")
response = rag.ask("سؤالك هنا")
```

---

## 📚 التالي؟

### للمبتدئين:
1. ✅ شغّل `quick_test_basic.py`
2. ✅ جرب أسئلتك الخاصة
3. ✅ اقرأ `STEP4_5_README.md`

### للمتقدمين:
1. ✅ احصل على Gemini API key (مجاني)
2. ✅ شغّل `quick_test_ai.py`
3. ✅ جرب `example_ai_analyzer.py`
4. ✅ اقرأ `AI_POWERED_README.md`

---

## 🐛 حل المشاكل

### خطأ: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### خطأ: Collection not found

```bash
python build/step3_embeddings_E5.py
```

سينشئ قاعدة البيانات (~15 دقيقة في المرة الأولى)

### خطأ: API key not found

- تأكد من ملف `.env` موجود
- تأكد من إضافة المفتاح الصحيح
- أو استخدم `quick_test_basic.py` (بدون API)

---

## 💡 نصائح

- **للتطوير**: استخدم Gemini (مجاني)
- **للإنتاج الرخيص**: استخدم Claude Haiku ($0.13/1000 سؤال)
- **للدقة العالية**: استخدم GPT-4 ($5/1000 سؤال)
- **بدون ميزانية**: استخدم النظام بدون AI (مجاني، دقة 85-90%)

---

## 📖 التوثيق الكامل

- **AI_POWERED_README.md** - دليل كامل للنسخة AI (16 KB)
- **STEP4_5_README.md** - دليل النسخة القديمة (13 KB)
- **quick_start.py** - فحص النظام
- **example_ai_analyzer.py** - أمثلة شاملة

---

## ⚡ البداية السريعة (خلاصة)

```bash
# 1. تحديث
git pull

# 2. تثبيت
pip install -r requirements.txt

# 3. اختبار
python quick_test_basic.py

# أو مع AI (بعد إعداد .env)
python quick_test_ai.py
```

**خلصت؟ جرب أسئلتك الآن! 🚀**
