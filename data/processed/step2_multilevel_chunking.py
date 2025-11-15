#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Level Chunking للنصوص الدينية
======================================
تقطيع هرمي: Document → Section → Paragraph
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# المسارات
# ═══════════════════════════════════════════════════════════════

RAW_DIR = Path("/mnt/user-data/uploads")
OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# وظائف مساعدة
# ═══════════════════════════════════════════════════════════════

def extract_pages(text: str) -> List[Dict]:
    """استخراج الصفحات من النص"""
    pages = []
    
    # نمط 1: الصفحة X
    # نمط 2: [الصفحة X من Y]
    pattern = r'(?:^|\n)(?:\[)?الصفحة\s+(\d+)'
    
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        
        # نهاية = بداية الصفحة التالية أو نهاية النص
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        
        page_text = text[start:end].strip()
        
        if page_text:
            pages.append({
                "page_num": page_num,
                "text": page_text
            })
    
    return pages


def count_words(text: str) -> int:
    """عد الكلمات"""
    return len(text.split())


def count_citations(text: str) -> int:
    """عد المصادر"""
    pattern = r'\((\d+)\)'
    return len(re.findall(pattern, text))


def extract_citation_refs(text: str) -> List[str]:
    """استخراج أرقام المصادر"""
    pattern = r'\((\d+)\)'
    return re.findall(pattern, text)


def generate_title(text: str, max_len: int = 60) -> str:
    """توليد عنوان من النص"""
    # أول جملة
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return "بدون عنوان"
    
    first = lines[0]
    # تنظيف
    first = re.sub(r'\(\d+\)', '', first)
    first = first.strip()
    
    if len(first) > max_len:
        return first[:max_len] + "..."
    return first


def classify_content(text: str) -> str:
    """تصنيف نوع المحتوى"""
    text_lower = text[:500].lower()
    
    if any(x in text_lower for x in ['مقدمة', 'بسم الله', 'الحمد لله']):
        return "مقدمة"
    elif any(x in text_lower for x in ['قال', 'فقالت', 'قالوا']):
        return "عرض آراء"
    elif any(x in text_lower for x in ['دليل', 'برهان', 'الحجة']):
        return "أدلة"
    elif any(x in text_lower for x in ['الجواب', 'فنقول', 'والرد']):
        return "ردود"
    elif any(x in text_lower for x in ['خلاصة', 'الخلاصة', 'فتبين']):
        return "خلاصة"
    else:
        return "نص عام"


def is_good_break(para: str) -> bool:
    """هل نقطة مناسبة للقطع؟"""
    para = para.strip()
    
    # نهاية طبيعية
    if para.endswith(('.', '؟', '!')):
        # ليس بعد مصدر مباشرة
        if not re.search(r'\(\d+\)$', para):
            return True
    
    return False


# ═══════════════════════════════════════════════════════════════
# المستوى 1: Documents
# ═══════════════════════════════════════════════════════════════

def create_documents(volumes: List[Tuple[int, str, Dict]]) -> List[Dict]:
    """
    إنشاء Documents
    كل جزء = document واحد
    """
    documents = []
    
    for vol_num, vol_text, citations_data in volumes:
        pages = extract_pages(vol_text)
        
        # ملخص بسيط (أول 500 كلمة)
        first_text = ' '.join([p['text'] for p in pages[:3]])
        summary_words = first_text.split()[:500]
        summary = ' '.join(summary_words) + "..."
        
        doc = {
            "doc_id": f"shafi_v{vol_num}",
            "type": "document",
            "book": "الشافي في الإمامة",
            "volume": vol_num,
            "author": "الشريف المرتضى (355-436 هـ)",
            
            "summary": summary,
            
            "stats": {
                "pages": len(pages),
                "words": count_words(vol_text),
                "citations": citations_data.get('total_citations', 0)
            },
            
            "children_sections": []
        }
        
        documents.append(doc)
        
        print(f"✅ Document {doc['doc_id']}: {doc['stats']['pages']} صفحة")
    
    return documents


# ═══════════════════════════════════════════════════════════════
# المستوى 2: Sections
# ═══════════════════════════════════════════════════════════════

def create_sections(doc_id: str, pages: List[Dict], 
                   citations: List[Dict], 
                   pages_per_section: int = 4) -> List[Dict]:
    """
    إنشاء Sections
    كل 4 صفحات = section
    """
    sections = []
    section_num = 1
    
    for i in range(0, len(pages), pages_per_section):
        section_pages = pages[i:i+pages_per_section]
        
        # دمج آخر section صغير مع السابق
        if len(section_pages) < 2 and sections:
            last_sec = sections[-1]
            last_sec['pages'].extend([p['page_num'] for p in section_pages])
            last_sec['text'] += "\n\n" + "\n\n".join([p['text'] for p in section_pages])
            last_sec['stats']['word_count'] = count_words(last_sec['text'])
            continue
        
        section_text = "\n\n".join([p['text'] for p in section_pages])
        page_nums = [p['page_num'] for p in section_pages]
        
        # عنوان
        title = generate_title(section_text)
        
        # مصادر في هذا القسم
        cite_refs = extract_citation_refs(section_text)
        section_citations = [
            c for c in citations 
            if c.get('number') in cite_refs
        ]
        
        section = {
            "section_id": f"{doc_id}_sec_{section_num:03d}",
            "type": "section",
            "parent_doc": doc_id,
            
            "title": title,
            "pages": page_nums,
            "text": section_text,
            
            "content_type": classify_content(section_text),
            
            "stats": {
                "word_count": count_words(section_text),
                "citation_count": len(section_citations)
            },
            
            "citations": section_citations[:10],  # أول 10 فقط
            
            "children_paragraphs": [],
            
            "next_section": None,
            "prev_section": None
        }
        
        sections.append(section)
        section_num += 1
    
    # روابط next/prev
    for i, sec in enumerate(sections):
        if i > 0:
            sec['prev_section'] = sections[i-1]['section_id']
        if i < len(sections) - 1:
            sec['next_section'] = sections[i+1]['section_id']
    
    return sections


# ═══════════════════════════════════════════════════════════════
# المستوى 3: Paragraphs
# ═══════════════════════════════════════════════════════════════

def create_paragraphs(section: Dict, citations: List[Dict],
                     min_words: int = 800, 
                     max_words: int = 1200) -> List[Dict]:
    """
    إنشاء Paragraphs
    تقسيم ذكي: 800-1200 كلمة
    """
    section_text = section['text']
    section_id = section['section_id']
    parent_doc = section['parent_doc']
    
    paragraphs = []
    para_num = 1
    
    # تقسيم أولي بالفقرات الطبيعية
    natural_paras = section_text.split('\n\n')
    
    current_text = ""
    
    for nat_para in natural_paras:
        nat_para = nat_para.strip()
        if not nat_para:
            continue
        
        current_text += nat_para + "\n\n"
        word_count = count_words(current_text)
        
        should_end = False
        
        if word_count >= max_words:
            should_end = True
        elif word_count >= min_words and is_good_break(nat_para):
            should_end = True
        
        if should_end:
            # إنشاء paragraph
            cite_refs = extract_citation_refs(current_text)
            para_citations = [
                c for c in citations 
                if c.get('number') in cite_refs
            ]
            
            # استخراج رقم الصفحة (أول رقم في النص)
            page_match = re.search(r'الصفحة (\d+)', current_text)
            page_num = int(page_match.group(1)) if page_match else section['pages'][0]
            
            paragraph = {
                "para_id": f"{section_id}_para_{para_num:03d}",
                "type": "paragraph",
                "parent_section": section_id,
                "parent_doc": parent_doc,
                
                "text": current_text.strip(),
                
                "stats": {
                    "word_count": word_count,
                    "page": page_num,
                    "citation_count": len(para_citations)
                },
                
                "content_type": classify_content(current_text),
                "citations": para_citations[:5],  # أول 5
                
                "next_para": None,
                "prev_para": None
            }
            
            paragraphs.append(paragraph)
            
            current_text = ""
            para_num += 1
    
    # البقية
    if current_text.strip():
        cite_refs = extract_citation_refs(current_text)
        para_citations = [
            c for c in citations 
            if c.get('number') in cite_refs
        ]
        
        page_match = re.search(r'الصفحة (\d+)', current_text)
        page_num = int(page_match.group(1)) if page_match else section['pages'][0]
        
        paragraph = {
            "para_id": f"{section_id}_para_{para_num:03d}",
            "type": "paragraph",
            "parent_section": section_id,
            "parent_doc": parent_doc,
            
            "text": current_text.strip(),
            
            "stats": {
                "word_count": count_words(current_text),
                "page": page_num,
                "citation_count": len(para_citations)
            },
            
            "content_type": classify_content(current_text),
            "citations": para_citations[:5],
            
            "next_para": None,
            "prev_para": None
        }
        
        paragraphs.append(paragraph)
    
    # روابط next/prev
    for i, para in enumerate(paragraphs):
        if i > 0:
            para['prev_para'] = paragraphs[i-1]['para_id']
        if i < len(paragraphs) - 1:
            para['next_para'] = paragraphs[i+1]['para_id']
    
    return paragraphs


# ═══════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("🎯 Multi-Level Chunking")
    print("="*70)
    
    # تحميل البيانات
    print("\n📂 تحميل البيانات...")
    
    volumes = []
    all_citations = {}
    
    for vol_num in range(1, 5):
        txt_file = RAW_DIR / f"الشافي_في_الإمامة_ج{vol_num}.txt"
        cite_file = RAW_DIR / f"citations_ج{vol_num}.json"
        
        print(f"  - الجزء {vol_num}...", end=" ")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        with open(cite_file, 'r', encoding='utf-8') as f:
            citations = json.load(f)
        
        volumes.append((vol_num, text, citations))
        all_citations[vol_num] = citations.get('citations', [])
        
        print("✅")
    
    # ═══════════════════════════════════════════════════════════
    # المستوى 1: Documents
    # ═══════════════════════════════════════════════════════════
    
    print("\n📚 المستوى 1: Documents...")
    documents = create_documents(volumes)
    
    # ═══════════════════════════════════════════════════════════
    # المستوى 2 & 3: Sections & Paragraphs
    # ═══════════════════════════════════════════════════════════
    
    print("\n📑 المستوى 2-3: Sections & Paragraphs...")
    
    all_sections = []
    all_paragraphs = []
    
    for vol_num, vol_text, citations_data in volumes:
        doc_id = f"shafi_v{vol_num}"
        print(f"\n  📖 معالجة {doc_id}...")
        
        pages = extract_pages(vol_text)
        citations = all_citations[vol_num]
        
        # Sections
        sections = create_sections(doc_id, pages, citations)
        print(f"    ✅ {len(sections)} sections")
        
        # Paragraphs
        section_para_count = 0
        for section in sections:
            paras = create_paragraphs(section, citations)
            section['children_paragraphs'] = [p['para_id'] for p in paras]
            all_paragraphs.extend(paras)
            section_para_count += len(paras)
        
        print(f"    ✅ {section_para_count} paragraphs")
        
        # تحديث document
        doc = next(d for d in documents if d['doc_id'] == doc_id)
        doc['children_sections'] = [s['section_id'] for s in sections]
        doc['stats']['sections'] = len(sections)
        doc['stats']['paragraphs'] = section_para_count
        
        all_sections.extend(sections)
    
    # ═══════════════════════════════════════════════════════════
    # Structure
    # ═══════════════════════════════════════════════════════════
    
    print("\n🗺️ بناء Structure...")
    
    structure = {
        "book": "الشافي في الإمامة",
        "author": "الشريف المرتضى",
        "total_volumes": 4,
        "total_documents": len(documents),
        "total_sections": len(all_sections),
        "total_paragraphs": len(all_paragraphs),
        
        "total_pages": sum(d['stats']['pages'] for d in documents),
        "total_words": sum(d['stats']['words'] for d in documents),
        "total_citations": sum(d['stats']['citations'] for d in documents),
        
        "hierarchy": {
            "documents": [d['doc_id'] for d in documents],
            "sections_per_document": {
                d['doc_id']: d['stats']['sections'] 
                for d in documents
            },
            "paragraphs_per_document": {
                d['doc_id']: d['stats']['paragraphs'] 
                for d in documents
            }
        },
        
        "created_at": datetime.now().isoformat()
    }
    
    # ═══════════════════════════════════════════════════════════
    # حفظ الملفات
    # ═══════════════════════════════════════════════════════════
    
    print("\n💾 حفظ الملفات...")
    
    files = {
        'structure.json': structure,
        'documents.json': documents,
        'sections.json': all_sections,
        'paragraphs.json': all_paragraphs
    }
    
    for filename, data in files.items():
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✅ {filename}: {size_kb:.1f} KB")
    
    # ═══════════════════════════════════════════════════════════
    # الإحصائيات النهائية
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("📊 الإحصائيات النهائية")
    print("="*70)
    print(f"📚 Documents: {len(documents)}")
    print(f"📑 Sections: {len(all_sections)}")
    print(f"📝 Paragraphs: {len(all_paragraphs)}")
    print(f"📄 Pages: {structure['total_pages']}")
    print(f"📖 Words: {structure['total_words']:,}")
    print(f"🔗 Citations: {structure['total_citations']}")
    print("="*70)
    print("✨ اكتمل بنجاح!")
    print("="*70)


if __name__ == "__main__":
    main()
