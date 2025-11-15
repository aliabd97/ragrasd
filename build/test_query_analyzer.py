"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧪 اختبارات Query Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from pathlib import Path

# إضافة build إلى المسار
sys.path.insert(0, str(Path(__file__).parent))

from step4_query_analyzer import QueryAnalyzer


class TestQueryAnalyzer:
    """مجموعة اختبارات لمحلل الأسئلة"""

    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.passed = 0
        self.failed = 0

    def assert_equal(self, actual, expected, test_name):
        """تأكيد التساوي"""
        if actual == expected:
            print(f"   ✅ {test_name}")
            self.passed += 1
            return True
        else:
            print(f"   ❌ {test_name}")
            print(f"      متوقع: {expected}")
            print(f"      الفعلي: {actual}")
            self.failed += 1
            return False

    def assert_in(self, item, collection, test_name):
        """تأكيد الاحتواء"""
        if item in collection:
            print(f"   ✅ {test_name}")
            self.passed += 1
            return True
        else:
            print(f"   ❌ {test_name}")
            print(f"      المتوقع: {item} موجود في {collection}")
            self.failed += 1
            return False

    def assert_greater(self, actual, threshold, test_name):
        """تأكيد أكبر من"""
        if actual > threshold:
            print(f"   ✅ {test_name}")
            self.passed += 1
            return True
        else:
            print(f"   ❌ {test_name}")
            print(f"      متوقع أكبر من: {threshold}")
            print(f"      الفعلي: {actual}")
            self.failed += 1
            return False

    def test_language_detection(self):
        """اختبار كشف اللغة"""
        print("\n📝 اختبار 1: كشف اللغة")
        print("─" * 50)

        # عربي
        analysis = self.analyzer.analyze("من هو الشريف المرتضى؟")
        self.assert_equal(analysis.language, "arabic", "كشف اللغة العربية")
        self.assert_greater(analysis.language_confidence, 0.7, "ثقة اللغة العربية > 70%")

        # إنجليزي
        analysis = self.analyzer.analyze("What is Imamah?")
        self.assert_equal(analysis.language, "english", "كشف اللغة الإنجليزية")
        self.assert_greater(analysis.language_confidence, 0.7, "ثقة اللغة الإنجليزية > 70%")

        # مختلط
        analysis = self.analyzer.analyze("ما هو Imamah في الإسلام؟")
        self.assert_equal(analysis.language, "mixed", "كشف اللغة المختلطة")

    def test_query_type_classification(self):
        """اختبار تصنيف نوع السؤال"""
        print("\n📝 اختبار 2: تصنيف نوع السؤال")
        print("─" * 50)

        # تعريف
        analysis = self.analyzer.analyze("ما هو تعريف الإمامة؟")
        self.assert_equal(analysis.query_type, "definition", "تصنيف سؤال التعريف")

        # شرح
        analysis = self.analyzer.analyze("اشرح مفهوم العصمة")
        self.assert_equal(analysis.query_type, "explanation", "تصنيف سؤال الشرح")

        # مقارنة
        analysis = self.analyzer.analyze("ما الفرق بين الإمامة والخلافة؟")
        self.assert_equal(analysis.query_type, "comparison", "تصنيف سؤال المقارنة")

        # قائمة
        analysis = self.analyzer.analyze("اذكر أنواع الأدلة على الإمامة")
        self.assert_equal(analysis.query_type, "list", "تصنيف سؤال القائمة")

        # حقيقي
        analysis = self.analyzer.analyze("من هو الشريف المرتضى؟")
        self.assert_equal(analysis.query_type, "factual", "تصنيف سؤال حقيقي")

    def test_keyword_extraction(self):
        """اختبار استخراج الكلمات المفتاحية"""
        print("\n📝 اختبار 3: استخراج الكلمات المفتاحية")
        print("─" * 50)

        analysis = self.analyzer.analyze("من هو الشريف المرتضى مؤلف كتاب الشافي؟")

        # التحقق من وجود الكلمات المفتاحية
        self.assert_in("الشريف", analysis.keywords, "استخراج 'الشريف'")
        self.assert_in("المرتضى", analysis.keywords, "استخراج 'المرتضى'")

        # التحقق من عدم وجود كلمات السؤال
        self.assert_equal(
            "من" in analysis.keywords,
            False,
            "عدم استخراج كلمة السؤال 'من'"
        )

    def test_question_words_extraction(self):
        """اختبار استخراج كلمات السؤال"""
        print("\n📝 اختبار 4: استخراج كلمات السؤال")
        print("─" * 50)

        # من
        analysis = self.analyzer.analyze("من هو الشريف المرتضى؟")
        self.assert_in("من", analysis.question_words, "استخراج 'من'")

        # ما
        analysis = self.analyzer.analyze("ما هو تعريف الإمامة؟")
        self.assert_in("ما", analysis.question_words, "استخراج 'ما'")

        # كيف
        analysis = self.analyzer.analyze("كيف يُثبت وجوب الإمامة؟")
        self.assert_in("كيف", analysis.question_words, "استخراج 'كيف'")

        # الإنجليزية
        analysis = self.analyzer.analyze("What is Imamah?")
        self.assert_in("what", analysis.question_words, "استخراج 'what'")

    def test_detail_level_detection(self):
        """اختبار كشف مستوى التفصيل"""
        print("\n📝 اختبار 5: كشف مستوى التفصيل")
        print("─" * 50)

        # موجز
        analysis = self.analyzer.analyze("اشرح بإيجاز مفهوم الإمامة")
        self.assert_equal(analysis.detail_level, "brief", "كشف مستوى موجز")

        # مفصل
        analysis = self.analyzer.analyze("اشرح بالتفصيل جميع أنواع الأدلة")
        self.assert_equal(analysis.detail_level, "detailed", "كشف مستوى مفصل")

        # متوسط (افتراضي)
        analysis = self.analyzer.analyze("ما هو تعريف الإمامة؟")
        self.assert_equal(analysis.detail_level, "moderate", "كشف مستوى متوسط (افتراضي)")

    def test_search_strategy(self):
        """اختبار بناء استراتيجية البحث"""
        print("\n📝 اختبار 6: بناء استراتيجية البحث")
        print("─" * 50)

        # موجز = 3 نتائج
        analysis = self.analyzer.analyze("اشرح بإيجاز مفهوم الإمامة")
        self.assert_equal(
            analysis.search_strategy['n_results'],
            3,
            "عدد النتائج للمستوى الموجز = 3"
        )

        # متوسط = 5 نتائج
        analysis = self.analyzer.analyze("ما هو تعريف الإمامة؟")
        self.assert_equal(
            analysis.search_strategy['n_results'],
            5,
            "عدد النتائج للمستوى المتوسط = 5"
        )

        # مفصل = 10 نتائج
        analysis = self.analyzer.analyze("اشرح بالتفصيل مفهوم الإمامة")
        self.assert_equal(
            analysis.search_strategy['n_results'],
            10,
            "عدد النتائج للمستوى المفصل = 10"
        )

        # التحقق من أولوية المستويات
        analysis = self.analyzer.analyze("من هو الشريف المرتضى؟")
        self.assert_equal(
            analysis.search_strategy['level_priority'][0],
            "paragraph",
            "أولوية المستوى الأول للسؤال الحقيقي = paragraph"
        )

    def test_metadata(self):
        """اختبار المعلومات الإضافية"""
        print("\n📝 اختبار 7: المعلومات الإضافية")
        print("─" * 50)

        # مع علامة استفهام
        analysis = self.analyzer.analyze("من هو الشريف المرتضى؟")
        self.assert_equal(
            analysis.metadata['has_question_mark'],
            True,
            "اكتشاف علامة الاستفهام"
        )

        # بدون علامة استفهام
        analysis = self.analyzer.analyze("اشرح مفهوم الإمامة")
        self.assert_equal(
            analysis.metadata['has_question_mark'],
            False,
            "عدم وجود علامة استفهام"
        )

        # عدد الكلمات
        analysis = self.analyzer.analyze("من هو الشريف المرتضى مؤلف كتاب الشافي")
        self.assert_greater(
            analysis.metadata['word_count'],
            5,
            "عدد الكلمات > 5"
        )

    def test_complex_queries(self):
        """اختبار الأسئلة المعقدة"""
        print("\n📝 اختبار 8: الأسئلة المعقدة")
        print("─" * 50)

        # سؤال طويل ومعقد
        query = "اشرح بالتفصيل الفرق بين مفهوم الإمامة عند الشيعة ومفهوم الخلافة عند السنة، مع ذكر الأدلة"
        analysis = self.analyzer.analyze(query)

        self.assert_equal(analysis.language, "arabic", "كشف اللغة للسؤال المعقد")
        self.assert_equal(analysis.query_type, "comparison", "تصنيف السؤال المعقد كمقارنة")
        self.assert_equal(analysis.detail_level, "detailed", "كشف مستوى التفصيل للسؤال المعقد")
        self.assert_greater(len(analysis.keywords), 3, "استخراج كلمات مفتاحية متعددة")

    def test_edge_cases(self):
        """اختبار الحالات الحدية"""
        print("\n📝 اختبار 9: الحالات الحدية")
        print("─" * 50)

        # سؤال قصير جداً
        analysis = self.analyzer.analyze("الإمامة؟")
        self.assert_equal(analysis.language, "arabic", "كشف اللغة للسؤال القصير")

        # سؤال بدون كلمات سؤال صريحة
        analysis = self.analyzer.analyze("الإمامة في الفكر الشيعي")
        self.assert_equal(
            analysis.query_type in ["definition", "factual"],
            True,
            "تصنيف سؤال بدون كلمات سؤال صريحة"
        )

        # سؤال مع مسافات زائدة
        analysis = self.analyzer.analyze("من    هو   الشريف    المرتضى؟")
        self.assert_greater(len(analysis.keywords), 0, "معالجة المسافات الزائدة")

    def run_all_tests(self):
        """تشغيل جميع الاختبارات"""
        print("\n" + "="*70)
        print("🧪 بدء اختبارات Query Analyzer")
        print("="*70)

        self.test_language_detection()
        self.test_query_type_classification()
        self.test_keyword_extraction()
        self.test_question_words_extraction()
        self.test_detail_level_detection()
        self.test_search_strategy()
        self.test_metadata()
        self.test_complex_queries()
        self.test_edge_cases()

        # النتيجة النهائية
        print("\n" + "="*70)
        print("📊 نتائج الاختبارات")
        print("="*70)
        print(f"\n✅ نجح: {self.passed}")
        print(f"❌ فشل: {self.failed}")
        print(f"📊 المجموع: {self.passed + self.failed}")
        print(f"📈 نسبة النجاح: {self.passed / (self.passed + self.failed) * 100:.1f}%")

        if self.failed == 0:
            print("\n🎉 جميع الاختبارات نجحت!")
        else:
            print(f"\n⚠️  {self.failed} اختبار فشل")

        print("="*70 + "\n")

        return self.failed == 0


def main():
    """تشغيل جميع الاختبارات"""
    tester = TestQueryAnalyzer()
    success = tester.run_all_tests()

    # رمز الخروج
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
