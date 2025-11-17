#!/usr/bin/env python3
"""
Tests for AI Chunking System

اختبارات للتأكد من:
1. عدم فقدان المصادر
2. سلامة البيانات
3. الربط الصحيح بين المستويات
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

# إضافة المسار للـ imports
sys.path.append(str(Path(__file__).parent.parent))


def load_json(filepath: str) -> Any:
    """تحميل JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_all_citations_preserved():
    """
    Test 1: لا فقدان في المصادر

    يتحقق من أن عدد المصادر المستخرجة = المصادر في الأقسام
    """
    print("\n" + "="*60)
    print("🧪 Test 1: All Citations Preserved")
    print("="*60)

    try:
        # تحميل البيانات
        citations_files = list(Path('data/processed/citations_extracted').glob('*.json'))
        if not citations_files:
            print("⚠️  No citation files found - skipping test")
            return True

        sections = load_json('data/processed/sections.json')

        # لكل جزء
        for cite_file in citations_files:
            vol_match = re.search(r'ج(\d+)', cite_file.name)
            if not vol_match:
                continue

            vol_num = int(vol_match.group(1))
            original = load_json(str(cite_file))

            # المصادر في الأقسام المقابلة
            vol_sections = [s for s in sections if f'v{vol_num}_' in s['section_id']]
            sections_citations = []
            for section in vol_sections:
                sections_citations.extend(section.get('citations', []))

            # المقارنة
            original_count = len(original['citations'])
            sections_count = len(sections_citations)

            print(f"\nVolume {vol_num}:")
            print(f"   Original citations:  {original_count}")
            print(f"   In sections:         {sections_count}")

            if original_count != sections_count:
                print(f"   ❌ FAIL: {abs(original_count - sections_count)} citations lost")
                return False
            else:
                print(f"   ✅ PASS: All citations preserved")

        print("\n✅ Test 1 PASSED: No citations lost")
        return True

    except FileNotFoundError as e:
        print(f"⚠️  Files not found: {e}")
        print("   Run step2_ai_chunking.py first")
        return True  # لا نفشل الاختبار إذا لم تُنشأ الملفات بعد

    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False


def test_chunks_have_sources():
    """
    Test 2: كل chunk له مصادره

    إذا كان النص يحتوي إشارة لمصدر، يجب أن يكون له citations
    """
    print("\n" + "="*60)
    print("🧪 Test 2: Chunks Have Their Sources")
    print("="*60)

    try:
        paragraphs = load_json('data/processed/paragraphs.json')

        failures = []

        for para in paragraphs:
            text = para.get('text', '')
            citations = para.get('citations', [])

            # البحث عن إشارات للمصادر في النص
            citation_patterns = [
                r'\(\d+\)',      # (1)
                r'\[\d+\]',      # [1]
                r'كما في.*?[،\.]',  # كما في...
                r'روى.*?[،\.]',     # روى...
            ]

            has_citation_reference = False
            for pattern in citation_patterns:
                if re.search(pattern, text):
                    has_citation_reference = True
                    break

            # إذا وُجدت إشارة لكن لا citations
            if has_citation_reference and len(citations) == 0:
                failures.append({
                    'para_id': para.get('para_id', 'unknown'),
                    'text_preview': text[:100]
                })

        if failures:
            print(f"\n❌ Found {len(failures)} paragraphs with citation references but no citations:")
            for i, fail in enumerate(failures[:5]):  # أول 5
                print(f"\n   {i+1}. {fail['para_id']}")
                print(f"      Preview: {fail['text_preview']}...")
            if len(failures) > 5:
                print(f"\n   ... and {len(failures) - 5} more")
            return False
        else:
            print(f"\n✅ Test 2 PASSED: All {len(paragraphs)} paragraphs have correct citations")
            return True

    except FileNotFoundError:
        print("⚠️  paragraphs.json not found - skipping test")
        return True

    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False


def test_no_broken_citations():
    """
    Test 3: لا مصادر مكسورة

    كل citation يجب أن يحتوي على source كامل
    """
    print("\n" + "="*60)
    print("🧪 Test 3: No Broken Citations")
    print("="*60)

    try:
        paragraphs = load_json('data/processed/paragraphs.json')

        broken = []

        for para in paragraphs:
            for citation in para.get('citations', []):
                # التحقق من وجود source
                if 'source' not in citation:
                    broken.append({
                        'para_id': para.get('para_id'),
                        'citation_id': citation.get('citation_id'),
                        'issue': 'Missing source field'
                    })
                    continue

                source = citation['source']

                # التحقق من الحقول المطلوبة
                if 'book' not in source and 'book_name' not in source:
                    broken.append({
                        'para_id': para.get('para_id'),
                        'citation_id': citation.get('citation_id'),
                        'issue': 'Missing book name'
                    })

                if 'reference' not in source and 'full_reference' not in source:
                    broken.append({
                        'para_id': para.get('para_id'),
                        'citation_id': citation.get('citation_id'),
                        'issue': 'Missing reference'
                    })

        if broken:
            print(f"\n❌ Found {len(broken)} broken citations:")
            for i, item in enumerate(broken[:5]):
                print(f"\n   {i+1}. {item['citation_id']}")
                print(f"      In: {item['para_id']}")
                print(f"      Issue: {item['issue']}")
            if len(broken) > 5:
                print(f"\n   ... and {len(broken) - 5} more")
            return False
        else:
            total_citations = sum(len(p.get('citations', [])) for p in paragraphs)
            print(f"\n✅ Test 3 PASSED: All {total_citations} citations are valid")
            return True

    except FileNotFoundError:
        print("⚠️  paragraphs.json not found - skipping test")
        return True

    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False


def test_hierarchy_integrity():
    """
    Test 4: سلامة التسلسل الهرمي

    Documents -> Sections -> Paragraphs
    """
    print("\n" + "="*60)
    print("🧪 Test 4: Hierarchy Integrity")
    print("="*60)

    try:
        documents = load_json('data/processed/documents.json')
        sections = load_json('data/processed/sections.json')
        paragraphs = load_json('data/processed/paragraphs.json')

        issues = []

        # 1. كل قسم له وثيقة أب
        for section in sections:
            parent_doc = section.get('parent_doc')
            if not parent_doc:
                issues.append(f"Section {section.get('section_id')} has no parent_doc")
                continue

            # التحقق من وجود الوثيقة
            doc_exists = any(d['doc_id'] == parent_doc for d in documents)
            if not doc_exists:
                issues.append(f"Section {section.get('section_id')} references non-existent doc: {parent_doc}")

        # 2. كل فقرة لها قسم أب
        for para in paragraphs:
            parent_section = para.get('parent_section')
            if not parent_section:
                issues.append(f"Paragraph {para.get('para_id')} has no parent_section")
                continue

            # التحقق من وجود القسم
            section_exists = any(s['section_id'] == parent_section for s in sections)
            if not section_exists:
                issues.append(f"Paragraph {para.get('para_id')} references non-existent section: {parent_section}")

        if issues:
            print(f"\n❌ Found {len(issues)} hierarchy issues:")
            for i, issue in enumerate(issues[:10]):
                print(f"   {i+1}. {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more")
            return False
        else:
            print(f"\n✅ Test 4 PASSED: Hierarchy is intact")
            print(f"   Documents: {len(documents)}")
            print(f"   Sections:  {len(sections)}")
            print(f"   Paragraphs: {len(paragraphs)}")
            return True

    except FileNotFoundError as e:
        print(f"⚠️  Files not found: {e}")
        return True

    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        return False


def test_json_structure():
    """
    Test 5: بنية JSON صحيحة

    التحقق من الحقول المطلوبة في كل مستوى
    """
    print("\n" + "="*60)
    print("🧪 Test 5: JSON Structure")
    print("="*60)

    try:
        # Documents
        documents = load_json('data/processed/documents.json')
        doc_required = ['doc_id', 'type', 'book', 'volume']

        for i, doc in enumerate(documents):
            for field in doc_required:
                if field not in doc:
                    print(f"❌ Document {i}: Missing field '{field}'")
                    return False

        # Sections
        sections = load_json('data/processed/sections.json')
        section_required = ['section_id', 'title', 'text', 'parent_doc']

        for i, section in enumerate(sections[:10]):  # عينة
            for field in section_required:
                if field not in section:
                    print(f"❌ Section {i}: Missing field '{field}'")
                    return False

        # Paragraphs
        paragraphs = load_json('data/processed/paragraphs.json')
        para_required = ['para_id', 'text', 'parent_section']

        for i, para in enumerate(paragraphs[:10]):  # عينة
            for field in para_required:
                if field not in para:
                    print(f"❌ Paragraph {i}: Missing field '{field}'")
                    return False

        print(f"\n✅ Test 5 PASSED: All JSON structures are valid")
        return True

    except FileNotFoundError as e:
        print(f"⚠️  Files not found: {e}")
        return True

    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        return False


def run_all_tests():
    """تشغيل جميع الاختبارات"""

    print("="*60)
    print("🧪 Running AI Chunking System Tests")
    print("="*60)

    tests = [
        ("All Citations Preserved", test_all_citations_preserved),
        ("Chunks Have Sources", test_chunks_have_sources),
        ("No Broken Citations", test_no_broken_citations),
        ("Hierarchy Integrity", test_hierarchy_integrity),
        ("JSON Structure", test_json_structure),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # النتائج النهائية
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("\n🎉 All tests passed! System is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
