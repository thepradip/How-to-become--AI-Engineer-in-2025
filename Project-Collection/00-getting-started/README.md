# 00 · Getting Started — Let AI Write the Code *With* You

You don't have to type every line. In 2026 the fastest way to ship these projects — whether you're a
senior engineer or a non-coder — is to **drive an AI coding tool** and review what it produces. This
guide gets you productive with **Claude Code**, **OpenAI Codex**, and **Lovable / v0** before you
touch Project 1.

> **Non-coder?** Read the "Plain-English workflow" box in each section. You'll describe what you want;
> the tool writes, runs, and fixes the code; you review and approve. That's a real, employable skill now.

---

## 1. Claude Code (terminal-native pair programmer)

Install: `npm i -g @anthropic-ai/claude-code` then run `claude` inside a project folder.

### The moves that matter
| Move | What it does | Why it saves you time |
|------|--------------|-----------------------|
| `/init` | Generates a `CLAUDE.md` describing your repo | The agent stops re-learning your project every session |
| **Plan mode** (`shift+tab` to cycle modes) | Agent researches & proposes a plan before editing | Catches wrong approaches *before* code is written |
| `/clear` | Wipes conversation context | Cheaper, sharper responses; do it between unrelated tasks |
| `/review` | Reviews a diff / PR for bugs | Free senior-eng code review |
| `/agents` | Create & run sub-agents | Parallelize: one explores, one writes tests |
| **`@path/to/file`** | Pin a file into context | The agent edits the *right* file, not a guess |
| **`!command`** | Run a shell command inline | Output lands in context (great for `!pytest`) |
| `#` a note | Saves a memory to `CLAUDE.md` | Teaches the agent your conventions permanently |
| `/mcp` | Connect MCP servers (DBs, Figma, browsers) | Give the agent real tools, not just the filesystem |

### `CLAUDE.md` — your project's "house rules"
Put conventions the agent must follow (style, test command, "never touch `scripts/`"). It's loaded
every session. Keep it short and specific. Example:
```md
- Run tests with `pytest -q`. Lint with `ruff`.
- UI = the shared chat component in _shared/chat_ui.py. Don't build bespoke UIs.
- Use real datasets via _shared/data.py. Never commit data/ or model weights.
```

### Tips & tricks (efficient + cheap)
- **Plan first, then build.** For anything non-trivial, ask for a plan, approve it, *then* let it code.
- **Small context = better answers.** `/clear` often; pin only the files that matter with `@`.
- **Let it run its own tests.** "Write the test, run it with `!pytest`, fix until green."
- **One task per session.** Churn vs. fraud are different sessions — don't mix.
- **Review every diff.** Treat the AI like a fast junior: trust, but verify the change.
- **Use sub-agents for fan-out** (e.g., "explore the dataset" + "draft the README" in parallel).

> **Plain-English workflow (non-coder):** open the project folder, run `claude`, and say:
> *"Set up this project, download the real dataset, train the model, and launch the app. Show me a
> plan first."* Approve the plan. When it asks to run commands, say yes. Read its summary at the end.

---

## 2. OpenAI Codex (cloud + CLI agent)

Codex shines for parallel, sandboxed tasks and long autonomous runs. Use the **CLI** (`codex`) locally
or the cloud agent for "go fix this and open a PR" work.

- **Give it a crisp task + acceptance test.** "Add SMOTE to the fraud pipeline; `pytest` must pass."
- **Sandbox by default.** It runs in an isolated env — safe to let it install deps and run code.
- **Diffs are the deliverable.** Review the proposed patch before merging.
- **Great for breadth:** kick off several independent project scaffolds at once, review later.

Claude Code vs. Codex: use **Claude Code** for interactive, in-repo iteration with tight feedback;
use **Codex** to dispatch well-scoped tasks you'll review asynchronously. Both read your repo and
run tests — the skill is *writing the spec*, not the syntax.

---

## 3. Lovable / v0 (UI from a prompt)

For the few projects that want a polished web front-end (the Next.js showcases), prototype the UI in
**Lovable** or **v0** in minutes, then wire it to your Python backend.

- Describe the screen in plain English; iterate by pointing at what's wrong ("make the chat bubbles
  wider, add a model dropdown").
- Export the React/Next.js code, then have Claude Code connect it to your FastAPI endpoint.
- Keep the **shared chat layout** — consistency beats novelty for a course.

---

## 4. The universal loop (works with any tool)
1. **Describe** the goal + the real dataset + the acceptance test.
2. **Plan** — make the AI propose steps; approve or correct.
3. **Build** in small steps; have it run code/tests after each.
4. **Review** the diff; ask "what could break?"
5. **Document** — "update the README with how to run and the results."

Master this loop once and all 53 projects become repetitions of the same motion.

Next → [`environment-setup.md`](environment-setup.md), then `ml/01-customer-churn/`.
