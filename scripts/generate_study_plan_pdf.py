from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "ai-engineer-2026-study-plan.pdf"
HERO = ROOT / "assets" / "ai-engineer-2026-hero.png"


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="TitleLarge",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=33,
            textColor=colors.HexColor("#111318"),
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subtitle",
            parent=base["BodyText"],
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#4f5864"),
            spaceAfter=14,
        )
    )
    base.add(
        ParagraphStyle(
            name="Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#111318"),
            spaceBefore=16,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4f5864"),
        )
    )
    base.add(
        ParagraphStyle(
            name="Cell",
            parent=base["BodyText"],
            fontSize=8.8,
            leading=11.5,
            textColor=colors.HexColor("#20242b"),
        )
    )
    base.add(
        ParagraphStyle(
            name="HeaderCell",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=11.5,
            textColor=colors.white,
        )
    )
    base.add(
        ParagraphStyle(
            name="BulletText",
            parent=base["BodyText"],
            fontSize=9.4,
            leading=12.5,
            leftIndent=0,
            textColor=colors.HexColor("#20242b"),
        )
    )
    return base


def para(text, style):
    return Paragraph(text, style)


def bullet_list(items, style):
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
        bulletFontSize=7,
        bulletOffsetY=1,
    )


def table(data, widths, header=True):
    converted = []
    for row_index, row in enumerate(data):
        style = STYLES["HeaderCell"] if header and row_index == 0 else STYLES["Cell"]
        converted.append([Paragraph(str(cell), style) for cell in row])
    t = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dde3")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111318")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for idx in range(1 if header else 0, len(data)):
        if idx % 2 == 0:
            style.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#f7f4ed")))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#626a73"))
    canvas.drawString(doc.leftMargin, 0.42 * inch, "How to Become an AI Engineer in 2026")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


STYLES = styles()


weeks = [
    ["1", "Python and Git", "Build a CLI helper, call APIs, and keep experiments in Git."],
    ["2", "Math for builders", "Vectors, probability, stats, cosine similarity, and evaluation vocabulary."],
    ["3", "LLM APIs", "Compare OpenAI, Claude, Gemini, Qwen, Kimi, and open model APIs."],
    ["4", "Prompt systems", "Task, context, constraints, schema, examples, critique, and stop rules."],
    ["5", "RAG baseline", "Parse, chunk, embed, retrieve, answer with citations, and test retrieval."],
    ["6", "LangGraph agents", "Planner, search, writer, critic, retries, state, and approvals."],
    ["7", "Advanced agents", "CrewAI, AutoGen, OpenHands, OpenClaw, Hermes Agent, and Agents SDK."],
    ["8", "Coding agents", "Ship a tested feature with Claude Code, Codex, Kimi Code, Cursor, or Cline."],
    ["9", "Vibe-coded UI", "Prototype with Lovable, v0, Bolt, Figma Make, Replit, or Gamma."],
    ["10", "Local inference", "Run Ollama and LM Studio; serve with vLLM and log throughput."],
    ["11", "Evals and monitoring", "Add traces, regression prompts, RAG metrics, cost logs, and safety tests."],
    ["12", "Portfolio capstone", "Publish an AI app with README, screenshots, evals, deploy notes, and pitch deck."],
]

tools = [
    ["Coding", "Claude Code, OpenAI Codex, Kimi Code, Cursor, Windsurf, Cline, Roo Code"],
    ["Agents", "LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, OpenHands, OpenClaw, Hermes Agent"],
    ["RAG", "LlamaIndex, LangChain, Haystack, Chroma, Qdrant, Milvus, Weaviate, rerankers"],
    ["Local inference", "Ollama, LM Studio, llama.cpp, vLLM, Hugging Face models"],
    ["UI/product", "Lovable, v0, Bolt, Figma Make, Streamlit, Chainlit, Next.js, Gamma"],
    ["LLMOps", "LangSmith, Phoenix, MLflow, W&B Weave, DeepEval, Ragas, Evidently"],
]

projects = [
    "Personal document search with citations and retrieval metrics.",
    "PDF-to-structured-data extractor with validation and failure examples.",
    "LangGraph research agent with planner, search, writer, critic, and approval checkpoint.",
    "Local inference lab with Ollama model switcher and vLLM benchmark report.",
    "Coding-agent feature shipped through Claude Code, Codex, or Kimi Code with tests.",
    "Portfolio AI product with polished UI, monitoring, eval report, deploy notes, and Gamma pitch.",
]

prompt_tips = [
    "Separate task, context, constraints, output format, examples, and quality bar.",
    "Ask for assumptions and missing information before final output when the task is ambiguous.",
    "Use a critique pass for weak spots, missing tests, risky assumptions, and better alternatives.",
    "For coding agents, include files to inspect, commands to run, repo rules, and definition of done.",
    "For autonomous agents, define tool permissions, stop conditions, budget, memory policy, and escalation rules.",
]


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.65 * inch,
        title="AI Engineer 2026 Study Plan",
    )
    story = []

    if HERO.exists():
        img = Image(str(HERO), width=7.05 * inch, height=2.65 * inch)
        img.hAlign = "LEFT"
        story.extend([img, Spacer(1, 14)])

    story.extend(
        [
            para("AI Engineer 2026 Study Plan", STYLES["TitleLarge"]),
            para(
                "A practical 12-week roadmap for Python, LLM APIs, prompt systems, RAG, LangGraph agents, coding assistants, local inference, UI prototyping, and LLMOps.",
                STYLES["Subtitle"],
            ),
            para("12-week path", STYLES["Section"]),
            table([["Wk", "Focus", "Build proof"], *weeks], [0.48 * inch, 1.65 * inch, 4.7 * inch]),
            PageBreak(),
            KeepTogether(
                [
                    para("Best tools by use case", STYLES["Section"]),
                    table([["Use case", "Recommended tools"], *tools], [1.35 * inch, 5.48 * inch]),
                ]
            ),
            para("Prompt enhancement checklist", STYLES["Section"]),
            bullet_list(prompt_tips, STYLES["BulletText"]),
            PageBreak(),
            para("Portfolio project ladder", STYLES["Section"]),
            bullet_list(projects, STYLES["BulletText"]),
            para("Recommended learning resources", STYLES["Section"]),
        ]
    )

    resources = [
        ["AI Python", "https://www.deeplearning.ai/short-courses/ai-python-for-beginners/"],
        ["AI Agents in LangGraph", "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/"],
        ["Multi AI Agent Systems with CrewAI", "https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/"],
        ["Building and Evaluating Data Agents", "https://www.deeplearning.ai/courses/building-and-evaluating-data-agents/"],
        ["Claude Code", "https://docs.anthropic.com/en/docs/claude-code/overview"],
        ["OpenAI Codex", "https://openai.com/codex/"],
        ["Kimi Code", "https://www.kimi.com/code/en"],
        ["Ollama", "https://docs.ollama.com/"],
        ["vLLM", "https://docs.vllm.ai/en/latest/"],
        ["Lovable", "https://lovable.dev/"],
        ["Gamma", "https://gamma.app/"],
        ["LangSmith", "https://docs.smith.langchain.com/"],
    ]
    story.append(table([["Resource", "Link"], *resources], [2.28 * inch, 4.55 * inch]))
    story.extend(
        [
            para("Definition of ready", STYLES["Section"]),
            bullet_list(
                [
                    "Build a Python or JavaScript AI app from scratch.",
                    "Explain embeddings, attention, tokens, context windows, tool calling, and RAG.",
                    "Use an AI coding agent responsibly and review its diffs.",
                    "Build a LangGraph or equivalent agent with state, tools, and evaluation.",
                    "Run a local model with Ollama and explain when vLLM is useful.",
                    "Deploy a small app and monitor latency, cost, quality, and safety.",
                ],
                STYLES["BulletText"],
            ),
            Spacer(1, 10),
            KeepTogether(
                [
                    para("Companion files", STYLES["Section"]),
                    para(
                        "Interactive HTML: docs/ai-engineer-2026-study-plan.html<br/>Repository README: README.md",
                        STYLES["Small"],
                    ),
                ]
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print(OUT)
