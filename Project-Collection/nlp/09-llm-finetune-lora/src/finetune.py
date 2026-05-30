"""Fine-tune a small LLM with LoRA / QLoRA using Unsloth (16 GB GPU).

LoRA trains tiny low-rank adapters instead of all weights; QLoRA adds 4-bit quantization
so a 7–8B model fits a single T4/M2 (16 GB). **Unsloth** makes this ~2× faster with less
memory. Actual training needs a GPU, so this module separates the **offline, tested**
data-prep utilities from the **GPU-only** ``train_lora`` routine (documented, importable).
"""

from __future__ import annotations

# Alpaca-style instruction template — the standard SFT format.
PROMPT = (
    "Below is an instruction that describes a task, paired with an input. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

SAMPLE_DATA = [
    {"instruction": "Classify the sentiment.", "input": "I love this!", "output": "positive"},
    {"instruction": "Summarize in one word.", "input": "The meeting was long and boring.", "output": "tedious"},
    {"instruction": "Translate to French.", "input": "good morning", "output": "bonjour"},
]


def format_example(ex: dict, eos: str = "") -> str:
    """Render one example into the training prompt string."""
    return PROMPT.format(instruction=ex.get("instruction", ""),
                         input=ex.get("input", ""),
                         output=ex.get("output", "")) + eos


def build_dataset(examples: list[dict], eos: str = "</s>") -> list[str]:
    """Turn raw examples into ready-to-tokenize training strings."""
    return [format_example(e, eos) for e in examples]


def train_lora(model_name: str = "unsloth/Qwen3-4B-bnb-4bit",
               examples: list[dict] | None = None,
               max_steps: int = 60, output_dir: str = "lora-out"):  # pragma: no cover - needs GPU
    """Real QLoRA fine-tune with Unsloth + TRL. **Requires a CUDA GPU.**"""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=2048, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    texts = build_dataset(examples or SAMPLE_DATA, eos=tokenizer.eos_token)
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(max_steps=max_steps, per_device_train_batch_size=2,
                       gradient_accumulation_steps=4, learning_rate=2e-4,
                       output_dir=output_dir, logging_steps=5))
    trainer.train()
    model.save_pretrained(output_dir)
    return output_dir
