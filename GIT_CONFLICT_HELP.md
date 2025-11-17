# 🔧 حل مشكلة Git Merge Conflict

## المشكلة
```
error: Pulling is not possible because you have unmerged files.
hint: Fix them up in the work tree, and then use 'git add/rm <file>'
hint: as appropriate to mark resolution and make a commit.
fatal: Exiting because of an unresolved conflict.
```

## الحل

### الطريقة 1: إعادة تعيين branch (الأسهل) ⭐

```bash
# 1. احفظ أي تغييرات محلية (إذا كان لديك)
git stash

# 2. إعادة تعيين branch للنسخة من GitHub
git fetch origin
git reset --hard origin/claude/fix-source-citations-01EQdDv99XcXKMuospmox71Q

# 3. استرجاع التغييرات المحلية (إذا كنت حفظتها)
git stash pop
```

### الطريقة 2: حل الـ conflict يدوياً

```bash
# 1. شاهد الملفات المتعارضة
git status

# 2. افتح كل ملف وابحث عن:
<<<<<<< HEAD
... your changes ...
=======
... incoming changes ...
>>>>>>> branch-name

# 3. احذف العلامات وحدد أي نسخة تريد

# 4. بعد التعديل:
git add <file-name>
git commit -m "حل الـ conflict"
```

### الطريقة 3: إلغاء الـ merge

```bash
# إلغاء عملية merge الحالية
git merge --abort

# ثم حاول مرة أخرى
git pull origin claude/fix-source-citations-01EQdDv99XcXKMuospmox71Q
```

### الطريقة 4: clone جديد (إذا فشل كل شيء)

```bash
# في مكان آخر:
git clone <repository-url>
cd ragrasd
git checkout claude/fix-source-citations-01EQdDv99XcXKMuospmox71Q
```

## التحقق من النجاح

```bash
# يجب أن ترى:
git status
# On branch claude/fix-source-citations-01EQdDv99XcXKMuospmox71Q
# nothing to commit, working tree clean

# تحقق من آخر commit:
git log --oneline -1
# e57c771 تحسين نظام RAG: إجابات مفصلة، مصادر واضحة، ودعم OpenAI
```

## ملاحظات مهمة

1. **احفظ عملك دائماً** قبل أي عملية reset أو merge
2. استخدم `git stash` لحفظ تغييراتك مؤقتاً
3. إذا كنت غير متأكد، اعمل نسخة احتياطية من المجلد كله

## الملفات الجديدة المتوقعة

بعد الـ pull الناجح، يجب أن ترى:

```
✅ .env.example
✅ README_NEW.md
✅ USAGE_GUIDE.md
✅ build/step3_embeddings_openai.py
✅ build/step4_query_analyzer.py
✅ build/step5_rag_system.py
✅ interactive_rag.py
✅ config.yaml (محدث)
✅ requirements.txt (محدث)
```

## الخطوات التالية

بعد حل المشكلة:

1. تثبيت المكتبات الجديدة:
   ```bash
   pip install -r requirements.txt
   ```

2. إعداد .env:
   ```bash
   cp .env.example .env
   # ثم عدّل .env ووضع OPENAI_API_KEY
   ```

3. تجربة النظام:
   ```bash
   python interactive_rag.py
   ```

---

**إذا استمرت المشكلة، أخبرني وسأساعدك! 🚀**
