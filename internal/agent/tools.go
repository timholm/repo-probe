package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/factory/repo-probe/internal/llm"
)

// ToolResult holds the output of a tool execution.
type ToolResult struct {
	Output string
	IsErr  bool
}

// Tool is the interface implemented by all agent tools.
type Tool interface {
	Name() string
	Description() string
	// Schema returns a human-readable argument description for the system prompt.
	Schema() string
	Execute(ctx context.Context, args map[string]string) ToolResult
}

// Registry holds all registered tools, keyed by name.
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a tool to the registry.
func (r *Registry) Register(t Tool) {
	r.tools[t.Name()] = t
}

// Get returns the tool with the given name, or nil if not found.
func (r *Registry) Get(name string) Tool {
	return r.tools[name]
}

// All returns all registered tools.
func (r *Registry) All() []Tool {
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// Schemas returns ToolSchema entries for all registered tools.
func (r *Registry) Schemas() []llm.ToolSchema {
	schemas := make([]llm.ToolSchema, 0, len(r.tools))
	for _, t := range r.tools {
		schemas = append(schemas, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			ArgsSchema:  t.Schema(),
		})
	}
	return schemas
}

// safePath ensures the given user-supplied path stays within repoRoot.
// Returns an error if the path escapes the repo root.
func safePath(repoRoot, userPath string) (string, error) {
	if userPath == "" {
		return repoRoot, nil
	}
	// If the path is absolute, reject it if it doesn't start with repoRoot.
	var resolved string
	if filepath.IsAbs(userPath) {
		resolved = filepath.Clean(userPath)
	} else {
		resolved = filepath.Clean(filepath.Join(repoRoot, userPath))
	}
	if !strings.HasPrefix(resolved+string(filepath.Separator), repoRoot+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes repository root", userPath)
	}
	return resolved, nil
}

// parseArgs parses a JSON args string into a map.
func parseArgs(raw string) (map[string]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "{}" {
		return map[string]string{}, nil
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return nil, fmt.Errorf("parsing args JSON %q: %w", raw, err)
	}
	result := make(map[string]string, len(m))
	for k, v := range m {
		result[k] = fmt.Sprintf("%v", v)
	}
	return result, nil
}

// ─── ReadFileTool ─────────────────────────────────────────────────────────────

// ReadFileTool reads lines from a file within the repo.
type ReadFileTool struct{ repoRoot string }

func NewReadFileTool(repoRoot string) *ReadFileTool { return &ReadFileTool{repoRoot: repoRoot} }

func (t *ReadFileTool) Name() string        { return "read_file" }
func (t *ReadFileTool) Description() string { return "Read lines from a source file" }
func (t *ReadFileTool) Schema() string {
	return `path: string (required), start_line: int (optional, default 1), end_line: int (optional, default 200)`
}

func (t *ReadFileTool) Execute(_ context.Context, args map[string]string) ToolResult {
	path, err := safePath(t.repoRoot, args["path"])
	if err != nil {
		return ToolResult{Output: err.Error(), IsErr: true}
	}

	startLine := 1
	if v := args["start_line"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			startLine = n
		}
	}
	endLine := startLine + 199 // default 200 lines
	if v := args["end_line"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			endLine = n
		}
	}

	f, err := os.Open(path)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("cannot open %s: %v", args["path"], err), IsErr: true}
	}
	defer f.Close()

	var sb strings.Builder
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum < startLine {
			continue
		}
		if lineNum > endLine {
			sb.WriteString(fmt.Sprintf("... (truncated at line %d)\n", endLine))
			break
		}
		sb.WriteString(fmt.Sprintf("%d: %s\n", lineNum, scanner.Text()))
	}
	if lineNum == 0 {
		return ToolResult{Output: "(empty file)"}
	}
	out := sb.String()
	if len(out) > 8000 {
		out = out[:8000] + "\n[truncated]"
	}
	return ToolResult{Output: out}
}

// ─── ListDirTool ──────────────────────────────────────────────────────────────

// ListDirTool lists the contents of a directory.
type ListDirTool struct {
	repoRoot string
	exclude  []string
}

func NewListDirTool(repoRoot string, exclude []string) *ListDirTool {
	return &ListDirTool{repoRoot: repoRoot, exclude: exclude}
}

func (t *ListDirTool) Name() string        { return "list_dir" }
func (t *ListDirTool) Description() string { return "List files and directories at a path" }
func (t *ListDirTool) Schema() string      { return `path: string (optional, default repo root)` }

func (t *ListDirTool) Execute(_ context.Context, args map[string]string) ToolResult {
	dir, err := safePath(t.repoRoot, args["path"])
	if err != nil {
		return ToolResult{Output: err.Error(), IsErr: true}
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("cannot read dir %s: %v", args["path"], err), IsErr: true}
	}

	var sb strings.Builder
	relDir, _ := filepath.Rel(t.repoRoot, dir)
	sb.WriteString(fmt.Sprintf("Contents of %s/:\n", relDir))

	for _, e := range entries {
		name := e.Name()
		if t.isExcluded(name) {
			continue
		}
		info, _ := e.Info()
		if e.IsDir() {
			sb.WriteString(fmt.Sprintf("  %s/\n", name))
		} else {
			size := ""
			if info != nil {
				size = fmt.Sprintf(" (%d bytes)", info.Size())
			}
			sb.WriteString(fmt.Sprintf("  %s%s\n", name, size))
		}
	}
	return ToolResult{Output: sb.String()}
}

func (t *ListDirTool) isExcluded(name string) bool {
	for _, pat := range t.exclude {
		if matched, _ := filepath.Match(pat, name); matched {
			return true
		}
		if name == pat {
			return true
		}
	}
	return false
}

// ─── GrepTool ─────────────────────────────────────────────────────────────────

// GrepTool searches for a regex pattern in the repository.
type GrepTool struct {
	repoRoot string
	exclude  []string
}

func NewGrepTool(repoRoot string, exclude []string) *GrepTool {
	return &GrepTool{repoRoot: repoRoot, exclude: exclude}
}

func (t *GrepTool) Name() string        { return "grep" }
func (t *GrepTool) Description() string { return "Search for a regex pattern in files" }
func (t *GrepTool) Schema() string {
	return `pattern: string (required), path: string (optional, default repo root), file_pattern: string (optional, e.g. "*.go")`
}

func (t *GrepTool) Execute(_ context.Context, args map[string]string) ToolResult {
	pattern := args["pattern"]
	if pattern == "" {
		return ToolResult{Output: "pattern is required", IsErr: true}
	}

	re, err := regexp.Compile(pattern)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("invalid pattern %q: %v", pattern, err), IsErr: true}
	}

	searchRoot, err := safePath(t.repoRoot, args["path"])
	if err != nil {
		return ToolResult{Output: err.Error(), IsErr: true}
	}

	filePattern := args["file_pattern"]

	var matches []string
	err = filepath.WalkDir(searchRoot, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			if t.isExcluded(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if t.isExcluded(d.Name()) {
			return nil
		}
		if filePattern != "" {
			if matched, _ := filepath.Match(filePattern, d.Name()); !matched {
				return nil
			}
		}

		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()

		relPath, _ := filepath.Rel(t.repoRoot, path)
		lineNum := 0
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			lineNum++
			line := scanner.Text()
			if re.MatchString(line) {
				matches = append(matches, fmt.Sprintf("%s:%d: %s", relPath, lineNum, strings.TrimSpace(line)))
				if len(matches) >= 100 {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})
	if err != nil && err != filepath.SkipAll {
		return ToolResult{Output: fmt.Sprintf("walk error: %v", err), IsErr: true}
	}

	if len(matches) == 0 {
		return ToolResult{Output: fmt.Sprintf("no matches for %q", pattern)}
	}

	out := strings.Join(matches, "\n")
	if len(matches) == 100 {
		out += "\n[results truncated at 100 matches]"
	}
	return ToolResult{Output: out}
}

func (t *GrepTool) isExcluded(name string) bool {
	for _, pat := range t.exclude {
		if matched, _ := filepath.Match(pat, name); matched {
			return true
		}
		if name == pat {
			return true
		}
	}
	return false
}

// ─── FindSymbolTool ───────────────────────────────────────────────────────────

// SymbolLookup is the interface for looking up symbols in an index.
type SymbolLookup interface {
	FindSymbol(name string) []SymbolInfo
}

// SymbolInfo describes a found symbol.
type SymbolInfo struct {
	Name    string
	Kind    string
	File    string
	Line    int
	Package string
}

// FindSymbolTool looks up symbols by name.
type FindSymbolTool struct {
	repoRoot string
	lookup   SymbolLookup
}

func NewFindSymbolTool(repoRoot string, lookup SymbolLookup) *FindSymbolTool {
	return &FindSymbolTool{repoRoot: repoRoot, lookup: lookup}
}

func (t *FindSymbolTool) Name() string        { return "find_symbol" }
func (t *FindSymbolTool) Description() string { return "Find where a symbol is defined in the codebase" }
func (t *FindSymbolTool) Schema() string {
	return `name: string (required), kind: string (optional: func/type/interface/const/var)`
}

func (t *FindSymbolTool) Execute(_ context.Context, args map[string]string) ToolResult {
	name := args["name"]
	if name == "" {
		return ToolResult{Output: "name is required", IsErr: true}
	}
	if t.lookup == nil {
		return ToolResult{Output: "symbol index not available (run 'repo-probe index' first)", IsErr: true}
	}

	kind := args["kind"]
	symbols := t.lookup.FindSymbol(name)
	if len(symbols) == 0 {
		return ToolResult{Output: fmt.Sprintf("symbol %q not found in index", name)}
	}

	var sb strings.Builder
	for _, sym := range symbols {
		if kind != "" && sym.Kind != kind {
			continue
		}
		relPath, _ := filepath.Rel(t.repoRoot, sym.File)
		sb.WriteString(fmt.Sprintf("%s %s — %s:%d (package %s)\n",
			sym.Kind, sym.Name, relPath, sym.Line, sym.Package))
	}
	if sb.Len() == 0 {
		return ToolResult{Output: fmt.Sprintf("symbol %q not found with kind %q", name, kind)}
	}
	return ToolResult{Output: sb.String()}
}

// ─── SemanticSearchTool ───────────────────────────────────────────────────────

// VectorSearcher is the interface for semantic search.
type VectorSearcher interface {
	Search(query string, topK int) []SearchResult
}

// SearchResult is a single semantic search result.
type SearchResult struct {
	ID      string  // "file:start-end"
	Text    string
	Score   float32
}

// SemanticSearchTool searches code chunks by semantic similarity.
type SemanticSearchTool struct {
	searcher VectorSearcher
}

func NewSemanticSearchTool(searcher VectorSearcher) *SemanticSearchTool {
	return &SemanticSearchTool{searcher: searcher}
}

func (t *SemanticSearchTool) Name() string        { return "semantic_search" }
func (t *SemanticSearchTool) Description() string { return "Search code by semantic meaning" }
func (t *SemanticSearchTool) Schema() string {
	return `query: string (required), top_k: int (optional, default 5)`
}

func (t *SemanticSearchTool) Execute(_ context.Context, args map[string]string) ToolResult {
	query := args["query"]
	if query == "" {
		return ToolResult{Output: "query is required", IsErr: true}
	}
	if t.searcher == nil {
		return ToolResult{Output: "vector index not available (run 'repo-probe index' first)", IsErr: true}
	}

	topK := 5
	if v := args["top_k"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			topK = n
		}
	}

	results := t.searcher.Search(query, topK)
	if len(results) == 0 {
		return ToolResult{Output: "no semantic matches found"}
	}

	var sb strings.Builder
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("[%d] %s (score: %.3f)\n%s\n\n", i+1, r.ID, r.Score, r.Text))
	}
	return ToolResult{Output: sb.String()}
}

// ─── SummarizeFileTool ────────────────────────────────────────────────────────

// SummarizeFileTool gets a high-level summary of a file using the LLM.
type SummarizeFileTool struct {
	repoRoot string
	provider llm.Provider
}

func NewSummarizeFileTool(repoRoot string, provider llm.Provider) *SummarizeFileTool {
	return &SummarizeFileTool{repoRoot: repoRoot, provider: provider}
}

func (t *SummarizeFileTool) Name() string        { return "summarize_file" }
func (t *SummarizeFileTool) Description() string { return "Get a high-level summary of what a file does" }
func (t *SummarizeFileTool) Schema() string      { return `path: string (required)` }

func (t *SummarizeFileTool) Execute(ctx context.Context, args map[string]string) ToolResult {
	path, err := safePath(t.repoRoot, args["path"])
	if err != nil {
		return ToolResult{Output: err.Error(), IsErr: true}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("cannot read %s: %v", args["path"], err), IsErr: true}
	}

	content := string(data)
	if len(content) > 6000 {
		content = content[:6000] + "\n[truncated]"
	}

	relPath, _ := filepath.Rel(t.repoRoot, path)
	prompt := fmt.Sprintf("Summarize what this file does in 3-5 sentences. Focus on its purpose, key types/functions, and how it fits into a larger system.\n\nFile: %s\n\n```\n%s\n```", relPath, content)

	resp, err := t.provider.Complete(ctx, llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
	})
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("summarization failed: %v", err), IsErr: true}
	}
	return ToolResult{Output: resp.Content}
}

// ─── Tool execution helper ────────────────────────────────────────────────────

// ExecuteTool finds and runs a tool by name with the given raw JSON args.
func ExecuteTool(ctx context.Context, reg *Registry, toolName, rawArgs string) ToolResult {
	tool := reg.Get(toolName)
	if tool == nil {
		return ToolResult{
			Output: fmt.Sprintf("unknown tool %q; available: %s", toolName, availableTools(reg)),
			IsErr:  true,
		}
	}

	args, err := parseArgs(rawArgs)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("invalid args for %s: %v", toolName, err), IsErr: true}
	}

	return tool.Execute(ctx, args)
}

func availableTools(reg *Registry) string {
	names := make([]string, 0)
	for _, t := range reg.All() {
		names = append(names, t.Name())
	}
	return strings.Join(names, ", ")
}
