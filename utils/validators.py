"""
Validation Framework for AI Chunking System
التحقق من بيانات المصادر والتقسيمات
"""

import re
from typing import Dict, List, Any


def validate_citations(citations_data: Dict[str, Any]) -> bool:
    """
    التحقق من بيانات المصادر المستخرجة

    Args:
        citations_data: البيانات المستخرجة من المرحلة 1

    Returns:
        bool: True إذا كانت البيانات صحيحة

    Raises:
        AssertionError: إذا فشل أي تحقق
    """
    print("\n🔍 Validating citations data...")

    # التحقق من وجود الحقول الأساسية
    assert 'citations' in citations_data, "Missing 'citations' field"
    assert len(citations_data['citations']) > 0, "No citations found"

    citations = citations_data['citations']

    # التحقق من كل مصدر
    for i, cite in enumerate(citations):
        # الحقول المطلوبة
        required_fields = ['citation_id', 'source', 'context']
        for field in required_fields:
            assert field in cite, f"Citation {i}: Missing required field '{field}'"

        # التحقق من بنية المصدر
        assert 'book_name' in cite['source'], f"Citation {i}: Missing 'book_name' in source"
        assert 'full_reference' in cite['source'], f"Citation {i}: Missing 'full_reference' in source"

        # التحقق من أن citation_id بالصيغة الصحيحة
        assert re.match(r'cite_v\d+_\d+', cite['citation_id']), \
            f"Citation {i}: Invalid citation_id format: {cite['citation_id']}"

    print(f"✅ Validated {len(citations)} citations")
    print(f"   Formats found: {citations_data.get('citation_formats_found', ['unknown'])}")

    return True


def validate_sections_citations(sections_data: Dict[str, Any],
                                  original_citations: Dict[str, Any]) -> bool:
    """
    التحقق من أن كل المصادر موجودة في sections

    Args:
        sections_data: البيانات من المرحلة 2
        original_citations: البيانات الأصلية من المرحلة 1

    Returns:
        bool: True إذا كانت المصادر متطابقة
    """
    print("\n🔍 Validating sections citations...")

    # جمع كل المصادر من sections
    sections_citations = []
    for section in sections_data['sections']:
        sections_citations.extend(section.get('citations', []))

    # المقارنة
    original_count = len(original_citations['citations'])
    sections_count = len(sections_citations)

    if original_count != sections_count:
        print(f"⚠️  WARNING: Citation count mismatch!")
        print(f"   Original citations:  {original_count}")
        print(f"   In sections:         {sections_count}")
        print(f"   Difference:          {abs(original_count - sections_count)}")

        # محاولة إيجاد المصادر المفقودة
        find_missing_citations(original_citations, sections_citations)

        return False
    else:
        print(f"✅ All {original_count} citations preserved in sections")
        return True


def validate_paragraphs_citations(paragraphs_data: Dict[str, Any],
                                    section_citations: List[Dict]) -> bool:
    """
    التحقق من المصادر في paragraphs

    Args:
        paragraphs_data: البيانات من المرحلة 3
        section_citations: المصادر من القسم الأب

    Returns:
        bool: True إذا كانت المصادر متطابقة
    """
    print("\n🔍 Validating paragraphs citations...")

    # جمع المصادر من الفقرات
    para_citations = []
    for para in paragraphs_data['paragraphs']:
        para_citations.extend(para.get('citations', []))

    section_count = len(section_citations)
    para_count = len(para_citations)

    if section_count != para_count:
        print(f"⚠️  WARNING: Citations mismatch!")
        print(f"   In section:     {section_count}")
        print(f"   In paragraphs:  {para_count}")
        print(f"   Difference:     {abs(section_count - para_count)}")
        return False

    print(f"✅ All {section_count} citations preserved in paragraphs")
    return True


def validate_chunk_citations(chunk: Dict[str, Any]) -> bool:
    """
    التحقق من chunk واحد - أن المصادر في metadata تطابق ما في النص

    Args:
        chunk: قطعة من النص (section أو paragraph)

    Returns:
        bool: True إذا كانت المصادر متسقة
    """
    text = chunk.get('text', '')
    citations = chunk.get('citations', [])

    # استخراج أرقام المصادر من النص
    patterns = [
        r'\((\d+)\)',           # (1)
        r'\[(\d+)\]',           # [1]
        r'(?:كما في|ذكر في|روى)\s+([^،\.]+)',  # نصية
    ]

    found_in_text = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        found_in_text.update(matches)

    # المصادر في metadata
    in_metadata = {
        c.get('appearance', c.get('citation_id', ''))
        for c in citations
    }

    # المقارنة
    if len(found_in_text) > 0 and len(found_in_text) != len(in_metadata):
        chunk_id = chunk.get('section_id') or chunk.get('para_id', 'unknown')
        print(f"⚠️  WARNING in {chunk_id}:")
        print(f"   Found in text: {found_in_text}")
        print(f"   In metadata:   {in_metadata}")
        return False

    return True


def find_missing_citations(original_citations: Dict[str, Any],
                           sections_citations: List[Dict]) -> None:
    """
    محاولة إيجاد المصادر المفقودة

    Args:
        original_citations: المصادر الأصلية
        sections_citations: المصادر في الأقسام
    """
    original_ids = {c['citation_id'] for c in original_citations['citations']}
    section_ids = {c.get('citation_id', '') for c in sections_citations}

    missing = original_ids - section_ids
    extra = section_ids - original_ids

    if missing:
        print(f"\n❌ Missing citations ({len(missing)}):")
        for cid in list(missing)[:10]:  # أول 10
            print(f"   - {cid}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

    if extra:
        print(f"\n➕ Extra citations ({len(extra)}):")
        for cid in list(extra)[:10]:
            print(f"   + {cid}")
        if len(extra) > 10:
            print(f"   ... and {len(extra) - 10} more")


def validate_text_preservation(original_text: str,
                                chunks_texts: List[str],
                                tolerance: float = 0.05) -> bool:
    """
    التحقق من عدم فقدان النص أثناء التقسيم

    Args:
        original_text: النص الأصلي
        chunks_texts: قائمة نصوص القطع
        tolerance: نسبة الخطأ المسموحة (5% افتراضياً)

    Returns:
        bool: True إذا كان الفرق ضمن المسموح
    """
    original_words = len(original_text.split())
    chunks_words = sum(len(text.split()) for text in chunks_texts)

    difference = abs(original_words - chunks_words)
    percentage = difference / original_words if original_words > 0 else 0

    if percentage > tolerance:
        print(f"⚠️  WARNING: Text loss detected!")
        print(f"   Original:  {original_words} words")
        print(f"   In chunks: {chunks_words} words")
        print(f"   Loss:      {difference} words ({percentage*100:.1f}%)")
        return False

    print(f"✅ Text preserved: {chunks_words}/{original_words} words ({(1-percentage)*100:.1f}%)")
    return True


def validate_json_structure(data: Dict[str, Any],
                            required_keys: List[str]) -> bool:
    """
    التحقق من بنية JSON

    Args:
        data: البيانات المراد التحقق منها
        required_keys: المفاتيح المطلوبة

    Returns:
        bool: True إذا كانت البنية صحيحة
    """
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: {key}")
            return False

    return True


# ============================================================================
# Test Functions
# ============================================================================

def test_validators():
    """اختبار سريع للـ validators"""

    print("🧪 Testing validators...")

    # بيانات تجريبية
    test_citations = {
        'volume_number': 1,
        'total_citations': 2,
        'citation_formats_found': ['رقمية'],
        'citations': [
            {
                'citation_id': 'cite_v1_001',
                'appearance_in_text': '(1)',
                'source': {
                    'book_name': 'تاريخ الطبري',
                    'full_reference': 'تاريخ الطبري، ج3، ص45'
                },
                'context': 'كما ذكر في المصدر (1) أن...'
            },
            {
                'citation_id': 'cite_v1_002',
                'appearance_in_text': '(2)',
                'source': {
                    'book_name': 'الكامل في التاريخ',
                    'full_reference': 'الكامل في التاريخ، ابن الأثير'
                },
                'context': 'روى في (2) عن...'
            }
        ]
    }

    # اختبار
    try:
        validate_citations(test_citations)
        print("✅ Citations validation test passed")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False

    print("✅ All validator tests passed")
    return True


if __name__ == '__main__':
    test_validators()
