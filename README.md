# How to Become an AI Engineer in 2026

<img src="assets/ai-engineer-2026-hero.png" alt="AI engineer 2026 roadmap" width="800">

This roadmap is for builders who want to go from Python basics to useful AI products, agents, local inference, evaluation, and UI shipping in 2026. It keeps the original beginner-friendly spirit, but adds the tools that now matter in day-to-day AI engineering: agentic coding assistants, LangGraph, local LLM inference, UI vibe-coding tools, prompt improvement workflows, and production evaluation.

## Study Plan Artifacts

- [Interactive HTML study plan](docs/ai-engineer-2026-study-plan.html)
- [Printable PDF study plan](output/pdf/ai-engineer-2026-study-plan.pdf)

## 2026 Learning Tracks

| Track | Goal | Build proof |
| --- | --- | --- |
| Foundations | Python, JavaScript, math, Git, APIs | CLI data app plus FastAPI endpoint |
| LLM basics | Transformers, embeddings, prompting, multimodal models | Chat app with tool calling |
| RAG | Chunking, retrieval, vector DBs, reranking, evaluation | Document Q&A with citations |
| Agents | LangGraph, CrewAI, Autogen, OpenAI/Anthropic agents, MCP | Research agent with memory and approval gates |
| Coding agents | Claude Code, OpenAI Codex, Kimi Code, Cursor, Cline, Windsurf | Ship a feature with tests using an AI coding workflow |
| Local inference | Ollama, LM Studio, llama.cpp, vLLM, quantization | Local model API plus benchmark notes |
| UI and product | Streamlit, Chainlit, Next.js, Lovable, v0, Gamma | Polished AI demo with a landing page and workflow UI |
| LLMOps | Tracing, evals, safety, monitoring, cost controls | Evaluated app with dashboards and regression tests |

## Free AI Platforms to Try First

- [ChatGPT](https://chat.openai.com/) - general assistant, coding, data analysis, image generation, agent workflows.
- [Claude](https://claude.ai/) - strong writing, coding, analysis, and long-context reasoning.
- [Gemini](https://gemini.google.com/app) - Google ecosystem assistant with multimodal workflows.
- [Kimi](https://www.kimi.com/) - long-context chat and Kimi Code assistant workflows.
- [DeepSeek](https://www.deepseek.com/) - reasoning and coding model family.
- [Qwen Chat](https://chat.qwen.ai/) - Alibaba Qwen models for chat, code, vision, and long-context exploration.
- [Grok](https://x.ai/) - xAI assistant and API ecosystem.
- [GroqCloud](https://console.groq.com/) - very fast hosted inference for open models.
- [Perplexity](https://www.perplexity.ai/) - research-focused answer engine with sources.
- [Poe](https://poe.com/) - try many model providers in one place.

## 1. Foundations

### Programming

- [AI Python for Beginners](https://www.deeplearning.ai/short-courses/ai-python-for-beginners/) - beginner Python with AI assistance.
- [Crash Course on Python](https://www.coursera.org/learn/python-crash-course) - Google/Coursera Python fundamentals.
- [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) - fast pandas orientation.
- [JavaScript Basics](https://www.coursera.org/learn/javascript-basics) - useful if you plan to build AI UIs.
- [Generative AI with JavaScript](https://www.youtube.com/playlist?list=PLlrxD0HtieHi5ZpsHULPLxm839IrhmeDk) - Microsoft JavaScript AI examples.
- [Streamlit](https://docs.streamlit.io/) - fastest path to Python data/AI demos.
- [Chainlit](https://docs.chainlit.io/get-started/overview) - conversational AI apps in Python.
- [FastAPI](https://fastapi.tiangolo.com/) - production-friendly Python APIs.

### Math

You do not need a PhD to start building, but you should understand probability, statistics, vectors, matrices, similarity, gradients, and evaluation.

- [Khan Academy Probability](https://www.youtube.com/playlist?list=PLC58778F28211FA19)
- [Khan Academy Statistics](https://www.youtube.com/playlist?list=PL1328115D3D8A2566)
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Probability and Statistics in Data Science using Python](https://www.edx.org/course/probability-and-statistics-in-data-science-using-python-2)

## 2. Generative AI Core

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) - transformers, tokenizers, datasets, and model workflows.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - visual explanation of transformer architecture.
- [Generative AI for Everyone](https://www.coursera.org/learn/generative-ai-for-everyone) - beginner overview from DeepLearning.AI.
- [Generative AI for Beginners](https://learn.microsoft.com/en-gb/shows/generative-ai-for-beginners/) - Microsoft 18-lesson builder course.
- [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) - image generation fundamentals.
- [OpenAI Cookbook](https://cookbook.openai.com/) - practical API recipes.
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) - Claude examples and patterns.
- [Gemini API Cookbook](https://ai.google.dev/gemini-api/cookbook) - Google Gemini examples.

## 3. Prompt Enhancement Tips and Tricks

Prompting in 2026 is less about magic words and more about reusable task design.

- Write the job, context, constraints, output format, and quality bar separately.
- Ask for assumptions before final output when the task is ambiguous.
- Use examples for style, schema, and edge cases.
- Request a critique pass: "find weak spots, missing facts, and risky assumptions."
- Convert one-off prompts into reusable templates with variables.
- For coding agents, include repo rules, test commands, files to inspect, and definition of done.
- For agents, state tool permissions, stop conditions, budget, memory policy, and escalation rules.
- Keep a prompt changelog when a workflow becomes important.

Recommended resources:

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Prompt Examples](https://platform.openai.com/docs/examples)
- [Anthropic Prompt Library](https://docs.anthropic.com/en/prompt-library/library)

## 4. AI Agents and LangGraph

Start with simple tool calling, then learn graphs, state, memory, retries, and evaluation.

Best free or beginner-friendly courses:

- [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) - a core DeepLearning.AI short course for graph-based agents.
- [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - hands-on multi-agent course.
- [Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) - tool use and agent basics.
- [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) - retrieval plus agent workflows.
- [Building and Evaluating Data Agents](https://www.deeplearning.ai/courses/building-and-evaluating-data-agents/) - data agents with LangGraph-style workflows.
- [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) - structured beginner agent curriculum.

Frameworks to learn:

- [LangGraph](https://www.langchain.com/langgraph) - graph-based orchestration for controllable, stateful agents.
- [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) - event-driven agent and RAG workflows.
- [CrewAI](https://www.crewai.com/) - role-based multi-agent teams.
- [Microsoft AutoGen](https://microsoft.github.io/autogen/) - conversational multi-agent framework.
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - OpenAI-native agent patterns.
- [Smolagents](https://huggingface.co/docs/smolagents/en/index) - lightweight code-agent framework from Hugging Face.
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) - open-source software engineering agent.
- [ElizaOS](https://github.com/elizaOS/eliza) - agent framework for social, app, and workflow agents.
- [OpenClaw](https://github.com/openclaw/openclaw) - emerging self-hosted agent/control-plane ecosystem; verify the repo and docs before adopting in production.
- [Hermes Agent](https://hermes-agent.ai/) - emerging self-hosted agent framework often discussed with Kimi/OpenClaw-style workflows; evaluate maturity before depending on it.

## 5. Agentic Coding Assistants

These tools are now part of the AI engineer's daily workflow. Learn the workflow, not just the brand.

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) - terminal coding agent from Anthropic; useful for repo exploration, edits, tests, and automation.
- [OpenAI Codex](https://openai.com/codex/) and [Codex CLI](https://help.openai.com/en/articles/11096431-openai-codex-cli-getting-started) - OpenAI coding agent for local and cloud software tasks.
- [Kimi Code](https://www.kimi.com/code/en) - Kimi/Moonshot coding assistant for terminal and IDE workflows.
- [Cursor](https://www.cursor.com/) - AI-first IDE with agent mode and codebase context.
- [Windsurf](https://windsurf.com/) - agentic IDE/workflow environment.
- [Cline](https://cline.bot/) - open-source coding agent extension.
- [Roo Code](https://roocode.com/) - coding agent workflow for VS Code style environments.

Practice loop:

1. Ask the agent to inspect before editing.
2. Give it a small, testable task.
3. Review diffs carefully.
4. Run tests locally.
5. Ask it to explain risks and missing coverage.
6. Commit only after you understand the change.

## 6. Vibe Coding for UI

Use vibe coding to move fast, then convert the output into maintainable code.

- [Lovable](https://lovable.dev/) - prompt-to-app builder for quick product prototypes.
- [v0](https://v0.dev/) - UI generation and React component exploration.
- [Bolt](https://bolt.new/) - browser-based full-stack prototyping.
- [Replit](https://replit.com/) - fast cloud development with AI assistance.
- [Figma Make](https://www.figma.com/make/) - design-to-prototype workflows.
- [Gamma](https://gamma.app/) - fast decks, docs, and idea pitches.

Good UI habit: generate the first draft with Lovable/v0/Bolt, then refactor layout, accessibility, state handling, API boundaries, and tests manually or with a coding agent.

## 7. RAG, Search, and Evaluation

- [Building and Evaluating Advanced RAG Applications](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) - DeepLearning.AI course.
- [LangChain Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/) - data framework for RAG and agents.
- [Haystack](https://haystack.deepset.ai/) - production search and RAG pipelines.
- [GraphRAG](https://microsoft.github.io/graphrag/) - graph-based retrieval.
- [RAGFlow](https://github.com/infiniflow/ragflow) - document-focused RAG system.
- [Crawl4AI](https://github.com/unclecode/crawl4ai) - web crawling for AI-ready data.
- [Unstructured](https://github.com/Unstructured-IO/unstructured) - parse messy files into structured content.
- [Chonkie](https://github.com/chonkie-ai/chonkie) and [Semchunk](https://github.com/umarbutler/semchunk) - chunking utilities.

Vector databases:

- [Chroma](https://www.trychroma.com/)
- [Qdrant](https://qdrant.tech/)
- [Milvus](https://milvus.io/)
- [Weaviate](https://weaviate.io/)
- [Pinecone](https://docs.pinecone.io/)

## 8. Local LLM Inference: Ollama, vLLM, and Friends

Learn local inference so you understand latency, memory, quantization, batching, and cost tradeoffs.

- [Ollama](https://docs.ollama.com/) - easiest way to run local LLMs on macOS, Linux, and Windows.
- [LM Studio](https://lmstudio.ai/) - desktop local model runner.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - core local inference engine and quantization ecosystem.
- [vLLM](https://docs.vllm.ai/en/latest/) - high-throughput serving engine for production inference.
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) - broad local model UI.
- [Hugging Face Models](https://huggingface.co/models) - model discovery.

Suggested path:

1. Run a 7B or 8B model locally with Ollama.
2. Compare latency and quality across quantized models.
3. Expose a local OpenAI-compatible endpoint.
4. Serve one model with vLLM and test concurrent requests.
5. Log tokens/sec, time-to-first-token, memory, and cost.

## 9. Fine-Tuning and Customization

- [Fine-Tuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
- [Unsloth](https://github.com/unslothai/unsloth) - efficient fine-tuning.
- [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face AutoTrain](https://huggingface.co/autotrain)
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)

Rule of thumb: try prompting, RAG, tools, and better evals before fine-tuning. Fine-tune when you need consistent style, domain behavior, extraction format, or smaller/cheaper model deployment.

## 10. LLMOps, Monitoring, and Safety

- [LangSmith](https://docs.smith.langchain.com/) - tracing and evaluation for LangChain/LangGraph apps.
- [Arize Phoenix](https://phoenix.arize.com/) - open-source observability and evals.
- [MLflow](https://mlflow.org/) - experiments and model management.
- [Weights & Biases Weave](https://wandb.ai/site/weave/) - LLM tracing and evaluation.
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation framework.
- [Ragas](https://github.com/explodinggradients/ragas) - RAG evaluation.
- [Evidently AI](https://www.evidentlyai.com/) - monitoring and eval workflows.
- [Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/)
- [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)

## 11. Best Tools by Use Case

| Use case | Start with | Upgrade when needed |
| --- | --- | --- |
| General assistant | ChatGPT, Claude, Gemini | Poe or provider APIs |
| Fast research | Perplexity, Gemini, ChatGPT search | Custom research agent with LangGraph |
| Coding | Claude Code, Codex, Cursor | Cline/Roo with custom providers |
| Long-context coding | Claude Code, Kimi Code | Repo-specific agent workflow |
| UI prototype | Lovable, v0, Bolt | Next.js plus design system |
| Decks/docs | Gamma, Tome, Canva | Figma Slides or PowerPoint automation |
| RAG prototype | LlamaIndex, LangChain, Chroma | Qdrant, Weaviate, Milvus, rerankers |
| Local model demo | Ollama, LM Studio | llama.cpp, MLX |
| Production inference | vLLM, TGI, Together, Fireworks | Ray Serve, Kubernetes, custom autoscaling |
| Agents | LangGraph, OpenAI Agents SDK | CrewAI, AutoGen, OpenHands, OpenClaw/Hermes experiments |
| Observability | LangSmith, Phoenix | W&B Weave, MLflow, custom eval harness |
| Fine-tuning | Unsloth notebooks | LLaMA-Factory, HF TRL, managed fine-tuning |

## 12. Projects to Build

1. Personal AI search over your notes with citations.
2. PDF-to-structured-data extractor with validation.
3. RAG chatbot with retrieval evals and hallucination tests.
4. LangGraph research agent with planner, search, writer, and critic nodes.
5. Local Ollama app that can switch between models and show latency.
6. vLLM inference endpoint with a small benchmark report.
7. Agentic coding workflow: ask Claude Code/Codex/Kimi Code to implement a feature and write tests.
8. Vibe-coded UI in Lovable/v0, then refactor into a clean Next.js app.
9. Gamma pitch deck for your AI product idea.
10. Fine-tuned small model with Unsloth for a narrow extraction or style task.

## 13. Research and Reading

- [Hugging Face Papers](https://huggingface.co/papers)
- [ArXiv AI Agent Search](https://arxiv.org/search/?query=AI+agent&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- [OpenAI Research](https://openai.com/research/)
- [Anthropic Research](https://www.anthropic.com/research)
- [Google DeepMind Research](https://deepmind.google/research/)
- [Meta AI Research](https://ai.meta.com/research/)
- [Foundations of Large Language Models](https://arxiv.org/pdf/2501.09223)
- [LLM Patterns by Eugene Yan](https://eugeneyan.com/writing/llm-patterns/)
- [Chip Huyen: Designing Machine Learning Systems](https://huyenchip.com/mlops/)
- [Kaggle AI Agents Whitepaper](https://www.kaggle.com/whitepaper-agents)

## 90-Day Path

| Days | Focus | Output |
| --- | --- | --- |
| 1-15 | Python, APIs, prompting, Git | CLI app plus prompt notebook |
| 16-30 | LLM APIs, embeddings, Streamlit/Chainlit | Chat app with tools |
| 31-45 | RAG, chunking, vector DBs, evals | Document Q&A app |
| 46-60 | LangGraph and multi-agent systems | Research agent with checkpoints |
| 61-75 | Local inference, Ollama, vLLM | Local model server and benchmark |
| 76-90 | UI, deployment, monitoring, portfolio | Polished capstone with README, demo, evals |

## Definition of Ready for an AI Engineer Role

You are ready to apply when you can:

- Build a Python or JavaScript AI app from scratch.
- Explain embeddings, attention, tokens, context windows, tool calling, and RAG.
- Use at least one coding agent responsibly and review its diffs.
- Build a LangGraph or equivalent agent with state and tool calls.
- Run a local model with Ollama and understand when vLLM is useful.
- Evaluate a RAG or agent app with repeatable test cases.
- Deploy a small app and monitor latency, cost, and quality.
- Communicate tradeoffs clearly: model choice, safety, privacy, UX, and budget.
