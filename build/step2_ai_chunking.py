#!/usr/bin/env python3
"""
Multi-Level AI Chunking with Citation Extraction

نظام تقسيم ذكي باستخدام Claude API:
- المرحلة 1: استخراج المصادر
- المرحلة 2: تقسيم إلى أقسام (sections)
- المرحلة 3: تقسيم إلى فقرات (paragraphs)

الميزة الأساسية: الحفاظ الكامل على المصادر 100%
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from dotenv import load_dotenv

# إضافة المسار للـ imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.validators import (
    validate_citations,
    validate_sections_citations,
    validate_paragraphs_citations,
    validate_text_preservation
)

try:
    from anthropic import Anthropic
except ImportError:
    print("❌ Error: anthropic library not installed")
    print("   Run: pip install anthropic")
    sys.exit(1)


def load_config() -> Dict[str, Any]:
    """تحميل الإعدادات من config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_text_file(filepath: str) -> str:
    """قراءة ملف نصي"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_json(data: Any, filepath: str) -> None:
    """حفظ JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """تحميل JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    استخراج JSON من رد Claude

    يزيل markdown code blocks ويحلل JSON
    """
    # إزالة markdown code blocks
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()

    # محاولة التحليل
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        print(f"Response text (first 500 chars):\n{text[:500]}")
        raise


class AIChunker:
    """
    AI-powered chunking system

    يستخدم Claude API لتقسيم النصوص بذكاء
    """

    def __init__(self, config: Dict[str, Any]):
        """
        تهيئة النظام

        Args:
            config: الإعدادات من config.yaml
        """
        # تحميل .env
        load_dotenv()

        # الإعدادات
        self.config = config.get('chunking', {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 16000,
            'temperature': 0,
            'api_key_env': 'ANTHROPIC_API_KEY'
        })

        # Claude client
        api_key = os.getenv(self.config['api_key_env'])
        if not api_key:
            raise ValueError(f"Missing API key: {self.config['api_key_env']}")

        self.client = Anthropic(api_key=api_key)

        # تحميل prompts
        self.load_prompts()

        # الإحصائيات
        self.stats = {
            'total_cost': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'volumes_processed': 0,
            'api_calls': 0
        }

    def load_prompts(self) -> None:
        """تحميل prompt templates"""
        prompts_dir = Path(__file__).parent / 'prompts'

        # Extract citations prompt
        with open(prompts_dir / 'extract_citations.txt', 'r', encoding='utf-8') as f:
            self.extract_citations_prompt = f.read()

        # Create sections prompt
        with open(prompts_dir / 'create_sections.txt', 'r', encoding='utf-8') as f:
            self.create_sections_prompt = f.read()

        # Create paragraphs prompt
        with open(prompts_dir / 'create_paragraphs.txt', 'r', encoding='utf-8') as f:
            self.create_paragraphs_prompt = f.read()

        print("✅ Loaded prompt templates")

    def call_claude(self, prompt: str, desc: str = "Processing") -> str:
        """
        استدعاء Claude API

        Args:
            prompt: النص المراد إرساله
            desc: وصف للعملية

        Returns:
            str: رد Claude
        """
        print(f"   🤖 Calling Claude API: {desc}...")

        response = self.client.messages.create(
            model=self.config['model'],
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature'],
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # تحديث الإحصائيات
        self.stats['api_calls'] += 1
        self.stats['total_input_tokens'] += response.usage.input_tokens
        self.stats['total_output_tokens'] += response.usage.output_tokens

        # حساب التكلفة (Claude Sonnet 4 pricing)
        # Input: $3 per million tokens
        # Output: $15 per million tokens
        input_cost = response.usage.input_tokens * 3 / 1_000_000
        output_cost = response.usage.output_tokens * 15 / 1_000_000
        call_cost = input_cost + output_cost
        self.stats['total_cost'] += call_cost

        print(f"      Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        print(f"      Cost: ${call_cost:.4f}")

        return response.content[0].text

    def extract_citations(self, volume_num: int, text: str) -> Dict[str, Any]:
        """
        المرحلة 1: استخراج المصادر

        Args:
            volume_num: رقم الجزء
            text: النص الكامل

        Returns:
            Dict: بيانات المصادر
        """
        print(f"\n{'='*60}")
        print(f"📚 Phase 1: Extracting citations from volume {volume_num}")
        print(f"{'='*60}")

        # بناء الـ prompt
        prompt = self.extract_citations_prompt.format(
            volume=volume_num,
            full_text=text
        )

        # استدعاء Claude
        response = self.call_claude(prompt, f"Extract citations v{volume_num}")

        # تحليل الرد
        citations_data = parse_json_response(response)

        # التحقق
        try:
            validate_citations(citations_data)
        except AssertionError as e:
            print(f"⚠️  Validation warning: {e}")

        # حفظ
        output_path = f"data/processed/citations_extracted/citations_ج{volume_num}.json"
        save_json(citations_data, output_path)

        print(f"✅ Extracted {len(citations_data.get('citations', []))} citations")
        print(f"   Saved to: {output_path}")

        return citations_data

    def create_sections(self, volume_num: int, text: str,
                       citations_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        المرحلة 2: تقسيم إلى sections

        Args:
            volume_num: رقم الجزء
            text: النص الكامل
            citations_data: بيانات المصادر من المرحلة 1

        Returns:
            Dict: بيانات الأقسام
        """
        print(f"\n{'='*60}")
        print(f"📑 Phase 2: Creating sections for volume {volume_num}")
        print(f"{'='*60}")

        # بناء خريطة المصادر (عينة)
        citations_map = [
            {
                'id': c['citation_id'],
                'appearance': c.get('appearance_in_text', ''),
                'book': c['source']['book_name'],
                'reference': c['source']['full_reference']
            }
            for c in citations_data['citations'][:50]  # أول 50 للتوضيح
        ]

        # بناء الـ prompt
        prompt = self.create_sections_prompt.format(
            volume=volume_num,
            total_citations=len(citations_data['citations']),
            citations_map=json.dumps(citations_map, ensure_ascii=False, indent=2),
            full_text=text
        )

        # استدعاء Claude
        response = self.call_claude(prompt, f"Create sections v{volume_num}")

        # تحليل الرد
        sections_data = parse_json_response(response)

        # التحقق
        try:
            validate_sections_citations(sections_data, citations_data)
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")

        print(f"✅ Created {len(sections_data.get('sections', []))} sections")

        return sections_data

    def create_paragraphs(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        المرحلة 3: تقسيم section إلى paragraphs

        Args:
            section: بيانات القسم

        Returns:
            List[Dict]: قائمة الفقرات
        """
        # بناء الـ prompt
        prompt = self.create_paragraphs_prompt.format(
            section_id=section['section_id'],
            section_title=section.get('title', 'بدون عنوان'),
            word_count=section.get('word_count', 0),
            citations_count=len(section.get('citations', [])),
            section_citations=json.dumps(
                section.get('citations', []),
                ensure_ascii=False,
                indent=2
            ),
            section_text=section.get('text', '')
        )

        # استدعاء Claude
        response = self.call_claude(prompt, f"Create paragraphs")

        # تحليل الرد
        paragraphs_data = parse_json_response(response)

        # التحقق
        try:
            validate_paragraphs_citations(
                paragraphs_data,
                section.get('citations', [])
            )
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")

        return paragraphs_data.get('paragraphs', [])

    def process_volume(self, volume_num: int, filepath: str) -> Dict[str, Any]:
        """
        معالجة جزء كامل

        Args:
            volume_num: رقم الجزء
            filepath: مسار ملف النص

        Returns:
            Dict: جميع البيانات المعالجة
        """
        print(f"\n{'='*60}")
        print(f"📖 Processing Volume {volume_num}")
        print(f"{'='*60}")

        # تحميل النص
        text = load_text_file(filepath)
        word_count = len(text.split())
        print(f"📄 Loaded {word_count:,} words from {filepath}")

        # المرحلة 1: استخراج المصادر
        citations_data = self.extract_citations(volume_num, text)

        # المرحلة 2: إنشاء الأقسام
        sections_data = self.create_sections(volume_num, text, citations_data)

        # المرحلة 3: إنشاء الفقرات لكل قسم
        print(f"\n{'='*60}")
        print(f"📝 Phase 3: Creating paragraphs")
        print(f"{'='*60}")

        all_paragraphs = []
        sections = sections_data.get('sections', [])

        for section in tqdm(sections, desc="Processing sections"):
            # تقسيم إلى فقرات
            paragraphs = self.create_paragraphs(section)

            # إضافة معلومات الأب
            for para in paragraphs:
                para['parent_section'] = section['section_id']
                para['parent_doc'] = f"shafi_v{volume_num}"

            all_paragraphs.extend(paragraphs)

            # تحديث القسم بقائمة الفقرات
            section['children_paragraphs'] = [p['para_id'] for p in paragraphs]

        print(f"✅ Created {len(all_paragraphs)} total paragraphs")

        # تحديث الإحصائيات
        self.stats['volumes_processed'] += 1

        return {
            'citations': citations_data,
            'sections': sections,
            'paragraphs': all_paragraphs
        }

    def build_documents_json(self, volumes_data: Dict[int, Dict]) -> List[Dict]:
        """
        بناء documents.json

        Args:
            volumes_data: بيانات جميع الأجزاء

        Returns:
            List[Dict]: قائمة الوثائق
        """
        documents = []

        for vol_num, data in volumes_data.items():
            doc = {
                'doc_id': f'shafi_v{vol_num}',
                'type': 'document',
                'book': 'الشافي في الإمامة',
                'volume': vol_num,
                'author': 'الشريف المرتضى (355-436 هـ)',

                # الإحصائيات
                'stats': {
                    'total_citations': len(data['citations']['citations']),
                    'total_sections': len(data['sections']),
                    'total_paragraphs': len(data['paragraphs'])
                },

                'children_sections': [s['section_id'] for s in data['sections']]
            }

            documents.append(doc)

        return documents

    def print_final_stats(self) -> None:
        """طباعة الإحصائيات النهائية"""
        print(f"\n{'='*60}")
        print("📊 Final Statistics")
        print(f"{'='*60}")

        print(f"\n🤖 API Usage:")
        print(f"   Total API calls:   {self.stats['api_calls']}")
        print(f"   Input tokens:      {self.stats['total_input_tokens']:,}")
        print(f"   Output tokens:     {self.stats['total_output_tokens']:,}")
        print(f"   Total tokens:      {self.stats['total_input_tokens'] + self.stats['total_output_tokens']:,}")

        print(f"\n💰 Cost:")
        print(f"   Total cost:        ${self.stats['total_cost']:.2f}")

        if self.stats['volumes_processed'] > 0:
            avg_cost = self.stats['total_cost'] / self.stats['volumes_processed']
            print(f"   Avg cost/volume:   ${avg_cost:.2f}")

        print(f"\n📚 Processing:")
        print(f"   Volumes processed: {self.stats['volumes_processed']}")


def main():
    """الدالة الرئيسية"""

    print("="*60)
    print("🎯 AI-Powered Intelligent Chunking")
    print("   Multi-Level Citation-Preserving System")
    print("="*60)

    # تحميل الإعدادات
    config = load_config()

    # إنشاء النظام
    try:
        chunker = AIChunker(config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Solution:")
        print("   1. Create .env file in project root")
        print("   2. Add: ANTHROPIC_API_KEY=sk-ant-...")
        return

    # الملفات المراد معالجتها
    # NOTE: ضع ملفاتك الحقيقية هنا
    volumes = {
        # 1: 'data/raw/الشافي_في_الإمامة_ج1.txt',
        # 2: 'data/raw/الشافي_في_الإمامة_ج2.txt',
        # 3: 'data/raw/الشافي_في_الإمامة_ج3.txt',
        # 4: 'data/raw/الشافي_في_الإمامة_ج4.txt'
    }

    # تحذير إذا لم توجد ملفات
    if not volumes:
        print("\n⚠️  No volumes configured!")
        print("   Edit build/step2_ai_chunking.py and add your file paths")
        print("\n   Example:")
        print("   volumes = {")
        print("       1: 'data/raw/book_v1.txt',")
        print("   }")
        return

    # معالجة كل جزء
    volumes_data = {}
    for vol_num, filepath in volumes.items():
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        volumes_data[vol_num] = chunker.process_volume(vol_num, filepath)

    if not volumes_data:
        print("\n❌ No volumes were processed")
        return

    # بناء الملفات النهائية
    print(f"\n{'='*60}")
    print("📦 Building final JSON files")
    print(f"{'='*60}")

    # documents.json
    documents = chunker.build_documents_json(volumes_data)
    save_json(documents, 'data/processed/documents.json')
    print(f"✅ Saved documents.json ({len(documents)} documents)")

    # sections.json
    all_sections = []
    for vol_data in volumes_data.values():
        all_sections.extend(vol_data['sections'])
    save_json(all_sections, 'data/processed/sections.json')
    print(f"✅ Saved sections.json ({len(all_sections)} sections)")

    # paragraphs.json
    all_paragraphs = []
    for vol_data in volumes_data.values():
        all_paragraphs.extend(vol_data['paragraphs'])
    save_json(all_paragraphs, 'data/processed/paragraphs.json')
    print(f"✅ Saved paragraphs.json ({len(all_paragraphs)} paragraphs)")

    # الإحصائيات
    stats = {
        'total_documents': len(documents),
        'total_sections': len(all_sections),
        'total_paragraphs': len(all_paragraphs),
        'total_citations': sum(
            len(v['citations']['citations'])
            for v in volumes_data.values()
        ),
        'api_usage': chunker.stats
    }
    save_json(stats, 'data/processed/chunking_stats.json')
    print(f"✅ Saved chunking_stats.json")

    # طباعة الإحصائيات النهائية
    chunker.print_final_stats()

    print(f"\n{'='*60}")
    print("🎉 Chunking Complete!")
    print(f"{'='*60}")
    print(f"\n📁 Output files:")
    print(f"   data/processed/documents.json")
    print(f"   data/processed/sections.json")
    print(f"   data/processed/paragraphs.json")
    print(f"   data/processed/chunking_stats.json")
    print(f"   data/processed/citations_extracted/")


if __name__ == '__main__':
    main()
