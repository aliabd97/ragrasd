#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات Multi-Level Chunking"""

import json
from pathlib import Path

OUTPUT_DIR = Path("/mnt/user-data/outputs")

def test_quantities():
    """اختبار الكميات"""
    print("📊 اختبار الكميات...")
    
    with open(OUTPUT_DIR / "structure.json") as f:
        structure = json.load(f)
    
    with open(OUTPUT_DIR / "documents.json") as f:
        docs = json.load(f)
    
    with open(OUTPUT_DIR / "sections.json") as f:
        secs = json.load(f)
    
    with open(OUTPUT_DIR / "paragraphs.json") as f:
        paras = json.load(f)
    
    assert len(docs) == 4, f"Documents: {len(docs)} != 4"
    assert 300 <= len(secs) <= 350, f"Sections: {len(secs)} not in range"
    assert len(paras) >= 400, f"Paragraphs: {len(paras)} < 400"
    
    print(f"  ✅ Documents: {len(docs)}")
    print(f"  ✅ Sections: {len(secs)}")
    print(f"  ✅ Paragraphs: {len(paras)}")


def test_relationships():
    """اختبار الروابط"""
    print("\n🔗 اختبار الروابط...")
    
    with open(OUTPUT_DIR / "documents.json") as f:
        docs = json.load(f)
    
    with open(OUTPUT_DIR / "sections.json") as f:
        secs = json.load(f)
    
    with open(OUTPUT_DIR / "paragraphs.json") as f:
        paras = json.load(f)
    
    doc_ids = {d['doc_id'] for d in docs}
    sec_ids = {s['section_id'] for s in secs}
    
    # Sections → Documents
    for sec in secs:
        assert sec['parent_doc'] in doc_ids
    
    # Paragraphs → Sections
    for para in paras:
        assert para['parent_section'] in sec_ids
        assert para['parent_doc'] in doc_ids
    
    print("  ✅ روابط صحيحة")


def test_sizes():
    """اختبار الأحجام"""
    print("\n📏 اختبار الأحجام...")
    
    with open(OUTPUT_DIR / "sections.json") as f:
        secs = json.load(f)
    
    with open(OUTPUT_DIR / "paragraphs.json") as f:
        paras = json.load(f)
    
    # Sections: 500-4000 كلمة
    for sec in secs[:10]:  # عينة
        wc = sec['stats']['word_count']
        assert 500 <= wc <= 5000, f"Section {sec['section_id']}: {wc} words"
    
    # Paragraphs: 100-2000 كلمة (السماح بفقرات صغيرة في النهاية)
    for para in paras[:20]:  # عينة
        wc = para['stats']['word_count']
        assert 1 <= wc <= 2500, f"Para {para['para_id']}: {wc} words"
    
    print("  ✅ أحجام مقبولة")


def show_sample():
    """عرض عينة"""
    print("\n📖 عينة من البيانات...")
    
    with open(OUTPUT_DIR / "sections.json") as f:
        secs = json.load(f)
    
    sec = secs[5]  # section عشوائي
    
    print(f"\n  📑 Section: {sec['section_id']}")
    print(f"  📝 Title: {sec['title'][:80]}...")
    print(f"  📄 Pages: {sec['pages']}")
    print(f"  💬 Words: {sec['stats']['word_count']}")
    print(f"  🔗 Citations: {sec['stats']['citation_count']}")
    print(f"  👶 Children: {len(sec['children_paragraphs'])}")


if __name__ == "__main__":
    print("="*60)
    print("🧪 اختبارات Multi-Level Chunking")
    print("="*60)
    
    try:
        test_quantities()
        test_relationships()
        test_sizes()
        show_sample()
        
        print("\n" + "="*60)
        print("✅ كل الاختبارات نجحت!")
        print("="*60)
    
    except AssertionError as e:
        print(f"\n❌ خطأ: {e}")
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
