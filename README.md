# repo-probe

A self-hosted code understanding engine that uses agentic exploration to answer deep questions about any codebase, powered by local LLMs via Ollama.

## What it does

Repo-probe implements the multi-step tool-use reasoning patterns validated by SWE-QA-Pro research to close a ~13-point accuracy gap versus direct answering. It orchestrates local models (default: qwen2.5-coder:7b) through agentic exploration — file browsing, grep, symbol navigation, semantic search — so that 8B models can outperform GPT-4o at repository-level questions without code ever leaving your network.

## Install

```bash
go install github.com/timholm/repo-probe@latest
```

## Usage

### Query a repository

```go
import (
    "github.com/timholm/repo-probe/internal/agent"
    "github.com/timholm/repo-probe/internal/config"
    "github.com/timholm/repo-probe/internal/llm"
)

cfg := config.LoadFromEnv()
provider, _ := llm.NewProvider(&cfg.LLM)

// Set up agent tools
repoRoot, _ := cfg.ResolveRepoRoot()
reg := agent.NewRegistry()
reg.Register(agent.NewReadFileTool(repoRoot))
reg.Register(agent.NewListDirTool(repoRoot, cfg.Repo.Exclude))
reg.Register(agent.NewGrepTool(repoRoot, cfg.Repo.Exclude))

// Decompose and plan
planner := agent.NewPlanner(provider)
plan := planner.Decompose(ctx, "How does the routing layer handle fallback?")
```

### Configuration

Configure via JSON file or environment variables:

```bash
REPO_PROBE_PROVIDER=ollama        # or "openai"
REPO_PROBE_BASE_URL=http://localhost:11434
REPO_PROBE_MODEL=qwen2.5-coder:7b
REPO_PROBE_EMBED_MODEL=nomic-embed-text
REPO_PROBE_REPO_ROOT=.
REPO_PROBE_MAX_STEPS=20
REPO_PROBE_SERVER_ADDR=:8080
```

## Agent Tools

- **read_file** — Read lines from a source file with optional range
- **list_dir** — List files and directories at a path
- **grep** — Search for regex patterns across the repository
- **find_symbol** — Look up symbol definitions (func, type, interface, const, var)
- **semantic_search** — Search code chunks by semantic similarity using embeddings
- **summarize_file** — Get an LLM-generated summary of what a file does

## Architecture

- **`internal/agent/`** — Core agent loop: planner decomposes queries into subgoals, tools execute exploration steps, trajectory tracks the reasoning chain, memory persists context
- **`internal/llm/`** — LLM provider abstraction supporting Ollama and OpenAI-compatible backends with streaming and embeddings
- **`internal/config/`** — Configuration loading from JSON files and environment variables with sensible defaults

## References

- [SWE-QA-Pro: Agentic Repository-Level Question Answering](https://arxiv.org/abs/2603.16124) — Research showing that agentic exploration closes a ~13-point accuracy gap versus direct answering for repository-level code questions

## License

MIT
