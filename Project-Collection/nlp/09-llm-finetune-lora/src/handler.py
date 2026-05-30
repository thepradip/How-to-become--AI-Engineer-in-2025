"""LoRA fine-tune brain. Uses the shared chat UI (data-prep demo; training needs a GPU)."""

from __future__ import annotations

from . import finetune as F


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    cmd = message.strip().lower()
    if cmd in {"data", "format", "demo", "", "go", "start", "train"}:
        examples = F.SAMPLE_DATA
        formatted = F.build_dataset(examples, eos="</s>")
        return [
            Reply(f"Prepared **{len(formatted)}** instruction examples in Alpaca format. "
                  "Training itself runs with Unsloth on a 16 GB GPU — see the README."),
            Reply(formatted[0], "code", {"language": "text"}),
            Reply("Replace `SAMPLE_DATA` with your domain examples and run `train_lora(...)` on a GPU.", "text"),
        ]
    return Reply("I prepare data and fine-tune an LLM with **LoRA/QLoRA via Unsloth** (GPU). "
                 "Type `data` to see a formatted training example.")
