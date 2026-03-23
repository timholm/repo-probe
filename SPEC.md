# repo-probe

**Language:** Go
**Source:** https://arxiv.org/abs/2603.16124
**Estimated lines:** 4500

## Problem

Engineering teams need deep, accurate answers about complex codebases but cloud LLMs hallucinate on private repos and local models lack the agentic exploration capabilities proven necessary by SWE-QA-Pro research.

## Solution

A self-hosted code understanding engine that implements agentic repository exploration — multi-step tool-use reasoning with file browsing, grep, symbol navigation — on top of local LLMs via Ollama. By orchestrating small models through the exploration patterns validated by SWE-QA-Pro (which close a ~13-point accuracy gap versus direct answering), repo-probe makes local 8B models outperform GPT-4o at repository-level questions without code ever leaving the network.

## Expected Files

["main.go","internal/agent/engine.go","internal/agent/tools.go","internal/agent/planner.go","internal/agent/memory.go","internal/agent/trajectory.go","internal/index/indexer.go","internal/index/symbols.go","internal/index/graph.go","internal/index/embeddings.go","internal/llm/ollama.go","internal/llm/openai.go","internal/llm/router.go","internal/llm/prompt.go","internal/server/handler.go","internal/server/stream.go","internal/server/middleware.go","internal/cli/root.go","internal/cli/query.go","internal/cli/index.go","internal/cli/serve.go","internal/config/config.go","internal/config/defaults.go","internal/eval/scorer.go","internal/eval/benchmark.go"]
