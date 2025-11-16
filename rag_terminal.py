"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 RAG Terminal App - نظام سؤال وجواب تفاعلي
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تطبيق terminal احترافي مع واجهة rich جميلة

الوضعان:
1. البحث البسيط: عرض النتائج من قاعدة البيانات فقط
2. الوضع المتقدم: توليد إجابة متكاملة باستخدام LLM

الإصدار: 1.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
from typing import Optional, List
from datetime import datetime

# إضافة مسار build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text

# تحميل .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# RAG System imports
from step5_rag_system import RAGSystem
from step5_ai_rag_system import AIRAGSystem

# Console setup
console = Console()


class RAGTerminalApp:
    """تطبيق Terminal للتفاعل مع نظام RAG"""

    def __init__(self):
        """تهيئة التطبيق"""
        self.console = console
        self.rag_basic = None
        self.rag_ai = None
        self.mode = "basic"  # basic أو advanced
        self.history = []

    def show_banner(self):
        """عرض شعار التطبيق"""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]  [bold white]🚀 نظام RAG للمحتوى الديني الإسلامي[/bold white]                          [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]  [dim]Multi-Level Retrieval-Augmented Generation System[/dim]        [bold cyan]║[/bold cyan]
[bold cyan]╚═══════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
        self.console.print(banner)

    def show_main_menu(self):
        """عرض القائمة الرئيسية"""
        self.console.print("\n[bold yellow]📋 القائمة الرئيسية:[/bold yellow]\n")

        menu_table = Table(show_header=False, box=box.SIMPLE)
        menu_table.add_column("Option", style="cyan", width=4)
        menu_table.add_column("Description", style="white")

        menu_table.add_row("1", "🔍 وضع البحث البسيط (بدون AI)")
        menu_table.add_row("2", "🤖 الوضع المتقدم (مع AI لتوليد الإجابات)")
        menu_table.add_row("3", "📊 الإحصائيات")
        menu_table.add_row("4", "📜 سجل الأسئلة")
        menu_table.add_row("5", "⚙️  الإعدادات")
        menu_table.add_row("0", "🚪 خروج")

        self.console.print(menu_table)

    def initialize_basic_mode(self):
        """تهيئة الوضع البسيط"""
        if self.rag_basic is None:
            with self.console.status("[bold green]⏳ تحميل نظام RAG البسيط...", spinner="dots"):
                try:
                    self.rag_basic = RAGSystem()
                    self.console.print("[bold green]✅ تم تحميل النظام البسيط بنجاح![/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]❌ خطأ في تحميل النظام: {e}[/bold red]")
                    return False
        return True

    def initialize_ai_mode(self):
        """تهيئة الوضع المتقدم"""
        if self.rag_ai is None:
            with self.console.status("[bold green]⏳ تحميل نظام RAG المتقدم...", spinner="dots"):
                try:
                    self.rag_ai = AIRAGSystem(llm_provider="auto", use_ai_analyzer=True)
                    self.console.print("[bold green]✅ تم تحميل النظام المتقدم بنجاح![/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]❌ خطأ في تحميل النظام: {e}[/bold red]")
                    self.console.print("[yellow]💡 تلميح: تأكد من إعداد API key في ملف .env[/yellow]")
                    return False
        return True

    def search_basic(self, query: str):
        """البحث في الوضع البسيط"""
        self.console.print("\n[bold cyan]🔍 البحث في الوضع البسيط...[/bold cyan]\n")

        with self.console.status("[bold green]⏳ جاري البحث...", spinner="dots"):
            try:
                response = self.rag_basic.search(query)

                # حفظ في السجل
                self.history.append({
                    'query': query,
                    'mode': 'basic',
                    'timestamp': datetime.now(),
                    'results_count': response.total_results
                })

                # عرض النتائج
                self._display_search_results(response)

            except Exception as e:
                self.console.print(f"[bold red]❌ خطأ في البحث: {e}[/bold red]")

    def search_advanced(self, query: str):
        """البحث في الوضع المتقدم مع توليد إجابة"""
        self.console.print("\n[bold magenta]🤖 البحث في الوضع المتقدم...[/bold magenta]\n")

        with self.console.status("[bold green]⏳ تحليل السؤال وتوليد الإجابة...", spinner="dots"):
            try:
                # البحث مع AI
                response = self.rag_ai.search(query)

                # حفظ في السجل
                self.history.append({
                    'query': query,
                    'mode': 'advanced',
                    'timestamp': datetime.now(),
                    'results_count': response.total_results
                })

                # عرض تحليل AI
                self._display_ai_analysis(response.ai_analysis)

                # عرض النتائج
                self._display_search_results(response)

                # توليد إجابة متكاملة
                self._generate_answer(query, response)

            except Exception as e:
                self.console.print(f"[bold red]❌ خطأ في البحث: {e}[/bold red]")

    def _display_ai_analysis(self, analysis):
        """عرض تحليل AI للسؤال"""
        self.console.print("\n[bold yellow]🤖 تحليل السؤال بالذكاء الاصطناعي:[/bold yellow]\n")

        analysis_panel = Panel(
            f"""[cyan]💡 التفسير:[/cyan] {analysis.ai_interpretation}

[cyan]📊 نوع السؤال:[/cyan] {analysis.query_type}
[cyan]🌐 اللغة:[/cyan] {analysis.language}
[cyan]🎯 الموضوع الرئيسي:[/cyan] {analysis.main_topic}
[cyan]📏 مستوى التعقيد:[/cyan] {analysis.complexity}
[cyan]📊 الثقة:[/cyan] {analysis.confidence:.0%}
[cyan]🤖 النموذج:[/cyan] {analysis.model_used}""",
            title="[bold cyan]تحليل AI[/bold cyan]",
            border_style="cyan"
        )

        self.console.print(analysis_panel)

    def _display_search_results(self, response):
        """عرض نتائج البحث"""
        self.console.print(f"\n[bold green]📊 تم العثور على {response.total_results} نتيجة في {response.search_time:.2f} ثانية[/bold green]\n")

        # جدول النتائج
        results_table = Table(
            title="[bold cyan]🎯 أفضل النتائج[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )

        results_table.add_column("#", style="cyan", width=3)
        results_table.add_column("النوع", style="yellow", width=10)
        results_table.add_column("النقاط", style="green", width=8)
        results_table.add_column("المحتوى", style="white")

        for i, result in enumerate(response.results[:5], 1):  # أول 5 نتائج
            content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content

            results_table.add_row(
                str(i),
                result.type.upper(),
                f"{result.score:.3f}",
                content_preview
            )

        self.console.print(results_table)

        # عرض تفاصيل إضافية للنتيجة الأولى
        if response.results:
            best_result = response.results[0]
            self.console.print(f"\n[bold yellow]📖 تفاصيل النتيجة الأولى:[/bold yellow]\n")

            details_panel = Panel(
                f"""[cyan]ID:[/cyan] {best_result.id}
[cyan]النوع:[/cyan] {best_result.type}
[cyan]النقاط:[/cyan] {best_result.score:.4f}

[cyan]المحتوى الكامل:[/cyan]
{best_result.content}

[dim]عدد الكلمات: {best_result.metadata.get('word_count', 'N/A')}[/dim]""",
                title="[bold green]أفضل نتيجة[/bold green]",
                border_style="green"
            )

            self.console.print(details_panel)

    def _generate_answer(self, query: str, response):
        """توليد إجابة متكاملة باستخدام LLM"""
        self.console.print("\n[bold magenta]🤖 توليد إجابة متكاملة...[/bold magenta]\n")

        # جمع المحتوى من أفضل النتائج
        context_parts = []
        for i, result in enumerate(response.results[:3], 1):
            context_parts.append(f"[مصدر {i}]: {result.content}")

        context = "\n\n".join(context_parts)

        # Prompt لتوليد الإجابة
        answer_prompt = f"""بناءً على المصادر التالية من الكتب الدينية الإسلامية، قدم إجابة شاملة ودقيقة للسؤال.

السؤال: {query}

المصادر:
{context}

تعليمات:
1. قدم إجابة واضحة ومباشرة
2. استخدم المعلومات من المصادر فقط
3. اذكر المصدر عند الاقتباس
4. كن دقيقاً وموضوعياً
5. إذا كانت المعلومات غير كافية، أخبر بذلك

الإجابة:"""

        with self.console.status("[bold green]⏳ توليد الإجابة...", spinner="dots"):
            try:
                # استخدام نفس LLM المستخدم في التحليل
                if hasattr(self.rag_ai.analyzer, 'provider'):
                    provider = self.rag_ai.analyzer.provider

                    if provider == "claude":
                        answer = self._generate_with_claude(answer_prompt)
                    elif provider == "openai":
                        answer = self._generate_with_openai(answer_prompt)
                    elif provider == "gemini":
                        answer = self._generate_with_gemini(answer_prompt)
                    else:
                        answer = "⚠️ لم يتم تهيئة LLM. استخدم الوضع البسيط."

                    # عرض الإجابة
                    answer_panel = Panel(
                        Markdown(answer),
                        title="[bold green]💡 الإجابة المتكاملة[/bold green]",
                        border_style="green",
                        padding=(1, 2)
                    )

                    self.console.print(answer_panel)

            except Exception as e:
                self.console.print(f"[bold red]❌ خطأ في توليد الإجابة: {e}[/bold red]")

    def _generate_with_claude(self, prompt: str) -> str:
        """توليد إجابة باستخدام Claude"""
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_with_openai(self, prompt: str) -> str:
        """توليد إجابة باستخدام OpenAI"""
        import openai

        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )

        return response.choices[0].message.content

    def _generate_with_gemini(self, prompt: str) -> str:
        """توليد إجابة باستخدام Gemini"""
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        response = model.generate_content(prompt)
        return response.text

    def show_statistics(self):
        """عرض إحصائيات الاستخدام"""
        self.console.print("\n[bold cyan]📊 الإحصائيات:[/bold cyan]\n")

        stats_table = Table(box=box.ROUNDED)
        stats_table.add_column("المقياس", style="cyan")
        stats_table.add_column("القيمة", style="green")

        total_queries = len(self.history)
        basic_queries = len([h for h in self.history if h['mode'] == 'basic'])
        advanced_queries = len([h for h in self.history if h['mode'] == 'advanced'])

        stats_table.add_row("إجمالي الأسئلة", str(total_queries))
        stats_table.add_row("البحث البسيط", str(basic_queries))
        stats_table.add_row("الوضع المتقدم", str(advanced_queries))

        self.console.print(stats_table)

    def show_history(self):
        """عرض سجل الأسئلة"""
        self.console.print("\n[bold cyan]📜 سجل الأسئلة:[/bold cyan]\n")

        if not self.history:
            self.console.print("[yellow]لا يوجد سجل بعد[/yellow]")
            return

        history_table = Table(box=box.ROUNDED, show_header=True)
        history_table.add_column("#", style="cyan", width=4)
        history_table.add_column("السؤال", style="white", width=40)
        history_table.add_column("الوضع", style="yellow", width=10)
        history_table.add_column("النتائج", style="green", width=8)
        history_table.add_column("الوقت", style="dim", width=20)

        for i, item in enumerate(reversed(self.history[-10:]), 1):  # آخر 10 أسئلة
            mode_emoji = "🔍" if item['mode'] == 'basic' else "🤖"
            history_table.add_row(
                str(i),
                item['query'][:40] + "..." if len(item['query']) > 40 else item['query'],
                f"{mode_emoji} {item['mode']}",
                str(item['results_count']),
                item['timestamp'].strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(history_table)

    def run(self):
        """تشغيل التطبيق الرئيسي"""
        self.show_banner()

        while True:
            self.show_main_menu()

            choice = Prompt.ask(
                "\n[bold yellow]اختر من القائمة[/bold yellow]",
                choices=["0", "1", "2", "3", "4", "5"],
                default="1"
            )

            if choice == "0":
                self.console.print("\n[bold green]👋 شكراً لاستخدامك النظام![/bold green]\n")
                break

            elif choice == "1":
                # الوضع البسيط
                if self.initialize_basic_mode():
                    while True:
                        query = Prompt.ask("\n[bold cyan]🔍 اكتب سؤالك (أو 'رجوع' للقائمة الرئيسية)[/bold cyan]")

                        if query.lower() in ['رجوع', 'back', 'exit', 'q']:
                            break

                        self.search_basic(query)

            elif choice == "2":
                # الوضع المتقدم
                if self.initialize_ai_mode():
                    while True:
                        query = Prompt.ask("\n[bold magenta]🤖 اكتب سؤالك (أو 'رجوع' للقائمة الرئيسية)[/bold magenta]")

                        if query.lower() in ['رجوع', 'back', 'exit', 'q']:
                            break

                        self.search_advanced(query)

            elif choice == "3":
                self.show_statistics()

            elif choice == "4":
                self.show_history()

            elif choice == "5":
                self.console.print("\n[yellow]⚙️  الإعدادات (قريباً)[/yellow]\n")


def main():
    """نقطة الدخول الرئيسية"""
    try:
        app = RAGTerminalApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]⚠️  تم إيقاف البرنامج بواسطة المستخدم[/bold yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ خطأ غير متوقع: {e}[/bold red]\n")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")


if __name__ == "__main__":
    main()
