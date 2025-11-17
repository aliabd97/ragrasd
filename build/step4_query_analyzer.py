#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Query Analyzer
=====================

المهمة:
- تحليل استعلام المستخدم
- تحديد نوع السؤال
- توسيع الاستعلام بمرادفات ومصطلحات ذات صلة
"""

from typing import Dict, List


class QueryAnalyzer:
    """محلل الاستعلامات"""

    def __init__(self):
        """التهيئة"""
        # أنواع الأسئلة
        self.question_types = {
            'who': ['من', 'من هو', 'من هي', 'من هم'],
            'what': ['ما', 'ماذا', 'ما هو', 'ما هي'],
            'when': ['متى', 'في أي وقت', 'في أي زمان'],
            'where': ['أين', 'في أي مكان'],
            'why': ['لماذا', 'لم', 'ما سبب', 'ما علة'],
            'how': ['كيف', 'بأي طريقة'],
            'definition': ['عرف', 'تعريف', 'معنى', 'مفهوم'],
            'ruling': ['حكم', 'جواز', 'يجوز', 'حلال', 'حرام'],
            'evidence': ['دليل', 'برهان', 'حجة']
        }

    def analyze(self, query: str) -> Dict:
        """
        تحليل الاستعلام

        Args:
            query: استعلام المستخدم

        Returns:
            معلومات التحليل
        """
        # تحديد نوع السؤال
        question_type = self._detect_question_type(query)

        # استخراج الكلمات المفتاحية
        keywords = self._extract_keywords(query)

        return {
            'original_query': query,
            'question_type': question_type,
            'keywords': keywords,
            'requires_detailed_answer': self._requires_detailed_answer(question_type)
        }

    def _detect_question_type(self, query: str) -> str:
        """تحديد نوع السؤال"""
        query_lower = query.lower().strip()

        for qtype, patterns in self.question_types.items():
            for pattern in patterns:
                if query_lower.startswith(pattern):
                    return qtype

        return 'general'

    def _extract_keywords(self, query: str) -> List[str]:
        """استخراج الكلمات المفتاحية"""
        # إزالة حروف الجر والضمائر
        stop_words = {'من', 'في', 'إلى', 'على', 'عن', 'هو', 'هي', 'هم',
                     'الذي', 'التي', 'الذين', 'ما', 'ماذا', 'كيف', 'أين'}

        words = query.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _requires_detailed_answer(self, question_type: str) -> bool:
        """هل يحتاج السؤال لإجابة مفصلة؟"""
        # أسئلة التعريف والشرح والحكم تحتاج إجابات مفصلة
        detailed_types = ['what', 'definition', 'ruling', 'why', 'how']
        return question_type in detailed_types
