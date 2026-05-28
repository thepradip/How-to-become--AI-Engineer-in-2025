from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "GenAI2025.png"
OUT = ROOT / "assets" / "ai-engineer-2026-hero.png"


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(draw, text, xy, max_width, font_obj, fill, line_gap=8):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font_obj)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font_obj)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_height
    return y


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    src = Image.open(SOURCE).convert("RGB")
    crop = src.crop((0, 167, 1172, 1094))
    bg = ImageOps.fit(crop, (1800, 1000), method=Image.Resampling.LANCZOS, centering=(0.34, 0.5))
    bg = ImageEnhance.Contrast(bg).enhance(1.05)
    bg = ImageEnhance.Color(bg).enhance(0.9)

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, 1800, 1000), fill=(6, 10, 18, 88))
    od.rectangle((0, 0, 1040, 1000), fill=(6, 10, 18, 118))
    od.rectangle((980, 0, 1800, 1000), fill=(247, 244, 237, 236))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay)

    d = ImageDraw.Draw(bg)
    white = (255, 255, 255, 255)
    mint = (174, 224, 200, 255)
    ink = (17, 19, 24, 255)
    muted = (80, 88, 98, 255)
    blue = (36, 88, 211, 255)
    green = (22, 132, 91, 255)
    red = (195, 72, 49, 255)
    gold = (184, 110, 0, 255)
    violet = (91, 79, 196, 255)

    d.text((80, 92), "2026 ROADMAP", font=font(34, True), fill=mint)
    d.text((78, 150), "AI Engineer", font=font(106, True), fill=white)
    d.text((82, 270), "Roadmap", font=font(106, True), fill=white)
    draw_wrapped(
        d,
        "Agents, coding assistants, RAG, local inference, UI shipping, and LLMOps.",
        (84, 420),
        790,
        font(31, False),
        (232, 238, 244, 255),
    )

    pills = [
        ("Claude Code", blue),
        ("OpenAI Codex", green),
        ("Kimi Code", red),
        ("LangGraph", violet),
        ("Ollama + vLLM", gold),
    ]
    x, y = 84, 520
    for label, color in pills:
        w = d.textbbox((0, 0), label, font=font(25, True))[2] + 42
        rounded(d, (x, y, x + w, y + 54), 27, color)
        d.text((x + 21, y + 13), label, font=font(25, True), fill=white)
        x += w + 16
        if x > 850:
            x, y = 84, y + 70

    d.text((1060, 78), "Build in layers", font=font(50, True), fill=ink)
    d.text((1062, 142), "A practical path from basics to portfolio.", font=font(25, False), fill=muted)

    tracks = [
        ("01", "Foundation", "Python, JavaScript, math, Git, APIs", blue),
        ("02", "LLM Core", "Transformers, prompting, embeddings", green),
        ("03", "RAG Systems", "Chunking, vector DBs, reranking, evals", gold),
        ("04", "AI Agents", "LangGraph, CrewAI, AutoGen, OpenHands", red),
        ("05", "Coding Agents", "Claude Code, Codex, Kimi Code, Cursor", violet),
        ("06", "Local Inference", "Ollama, LM Studio, llama.cpp, vLLM", blue),
        ("07", "Product UI", "Lovable, v0, Gamma, Streamlit, Next.js", green),
        ("08", "LLMOps", "Tracing, safety, monitoring, cost control", red),
    ]

    start_y = 224
    for i, (num, title, desc, color) in enumerate(tracks):
        top = start_y + i * 86
        rounded(d, (1060, top, 1716, top + 66), 18, (255, 255, 255, 255), outline=(218, 222, 228, 255), width=2)
        rounded(d, (1080, top + 13, 1130, top + 63 - 13), 12, color)
        d.text((1094, top + 23), num, font=font(20, True), fill=white)
        d.text((1150, top + 10), title, font=font(26, True), fill=ink)
        d.text((1150, top + 40), desc, font=font(18, False), fill=muted)

    draw_wrapped(
        d,
        "Capstone: evaluated agentic AI app with polished UI and deploy notes",
        (84, 850),
        805,
        font(28, True),
        white,
    )
    d.text((1062, 930), "Updated for 2026", font=font(24, True), fill=ink)

    # A soft finish prevents harsh compression edges in GitHub previews.
    bg = bg.filter(ImageFilter.UnsharpMask(radius=1, percent=105, threshold=3))
    bg.convert("RGB").save(OUT, quality=94)
    print(OUT)


if __name__ == "__main__":
    main()
