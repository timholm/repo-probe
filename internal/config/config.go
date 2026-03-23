package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config holds all runtime configuration for repo-probe.
type Config struct {
	Repo   RepoConfig   `json:"repo"`
	LLM    LLMConfig    `json:"llm"`
	Index  IndexConfig  `json:"index"`
	Server ServerConfig `json:"server"`
	Agent  AgentConfig  `json:"agent"`
	Eval   EvalConfig   `json:"eval"`
}

type RepoConfig struct {
	Root    string   `json:"root"`
	Exclude []string `json:"exclude"`
}

type LLMConfig struct {
	Provider           string        `json:"provider"` // "ollama" | "openai"
	BaseURL            string        `json:"base_url"`
	Model              string        `json:"model"`
	EmbedModel         string        `json:"embed_model"`
	APIKey             string        `json:"api_key"`
	MaxTokens          int           `json:"max_tokens"`
	ContextWindowTokens int          `json:"context_window_tokens"`
	Temperature        float64       `json:"temperature"`
	Timeout            time.Duration `json:"timeout"`
}

type IndexConfig struct {
	CacheDir   string `json:"cache_dir"`
	Dimensions int    `json:"dimensions"`
}

type ServerConfig struct {
	Addr         string        `json:"addr"`
	ReadTimeout  time.Duration `json:"read_timeout"`
	WriteTimeout time.Duration `json:"write_timeout"`
}

type AgentConfig struct {
	MaxSteps int `json:"max_steps"`
}

type EvalConfig struct {
	DatasetPath string `json:"dataset_path"`
	OutputPath  string `json:"output_path"`
	TimeoutSecs int    `json:"timeout_secs"`
}

// Default returns a Config populated with sensible defaults.
func Default() *Config {
	return &Config{
		Repo: RepoConfig{
			Root:    ".",
			Exclude: DefaultExcludePatterns,
		},
		LLM: LLMConfig{
			Provider:            DefaultProvider,
			BaseURL:             DefaultOllamaURL,
			Model:               DefaultModel,
			EmbedModel:          DefaultEmbedModel,
			MaxTokens:           DefaultMaxTokens,
			ContextWindowTokens: DefaultContextWindowTokens,
			Temperature:         DefaultTemperature,
			Timeout:             DefaultLLMTimeout,
		},
		Index: IndexConfig{
			CacheDir:   DefaultCacheDir,
			Dimensions: DefaultEmbedDimensions,
		},
		Server: ServerConfig{
			Addr:         DefaultServerAddr,
			ReadTimeout:  DefaultReadTimeout,
			WriteTimeout: DefaultWriteTimeout,
		},
		Agent: AgentConfig{
			MaxSteps: DefaultMaxSteps,
		},
		Eval: EvalConfig{
			TimeoutSecs: 120,
		},
	}
}

// Load reads a JSON config file and merges it over defaults.
func Load(path string) (*Config, error) {
	cfg := Default()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config %s: %w", path, err)
	}
	applyEnv(cfg)
	return cfg, nil
}

// LoadFromEnv returns a config built entirely from environment variables,
// merged over defaults.
func LoadFromEnv() *Config {
	cfg := Default()
	applyEnv(cfg)
	return cfg
}

// applyEnv overrides cfg fields from REPO_PROBE_* environment variables.
func applyEnv(cfg *Config) {
	if v := os.Getenv("REPO_PROBE_PROVIDER"); v != "" {
		cfg.LLM.Provider = v
	}
	if v := os.Getenv("REPO_PROBE_BASE_URL"); v != "" {
		cfg.LLM.BaseURL = v
	}
	if v := os.Getenv("REPO_PROBE_MODEL"); v != "" {
		cfg.LLM.Model = v
	}
	if v := os.Getenv("REPO_PROBE_EMBED_MODEL"); v != "" {
		cfg.LLM.EmbedModel = v
	}
	if v := os.Getenv("REPO_PROBE_API_KEY"); v != "" {
		cfg.LLM.APIKey = v
	}
	if v := os.Getenv("REPO_PROBE_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.LLM.MaxTokens = n
		}
	}
	if v := os.Getenv("REPO_PROBE_SERVER_ADDR"); v != "" {
		cfg.Server.Addr = v
	}
	if v := os.Getenv("REPO_PROBE_REPO_ROOT"); v != "" {
		cfg.Repo.Root = v
	}
	if v := os.Getenv("REPO_PROBE_MAX_STEPS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Agent.MaxSteps = n
		}
	}
}

// Validate checks that the config is internally consistent.
func Validate(cfg *Config) error {
	if cfg.LLM.Provider == "" {
		return fmt.Errorf("llm.provider is required")
	}
	if cfg.LLM.Provider != "ollama" && cfg.LLM.Provider != "openai" {
		return fmt.Errorf("llm.provider must be 'ollama' or 'openai', got %q", cfg.LLM.Provider)
	}
	if cfg.LLM.BaseURL == "" {
		return fmt.Errorf("llm.base_url is required")
	}
	if cfg.LLM.Model == "" {
		return fmt.Errorf("llm.model is required")
	}
	if cfg.Agent.MaxSteps < 1 {
		return fmt.Errorf("agent.max_steps must be >= 1")
	}
	if cfg.LLM.Provider == "openai" && cfg.LLM.APIKey == "" {
		// Allow empty key for local OpenAI-compatible endpoints like vLLM.
		// Not a hard error.
	}
	return nil
}

// ResolveRepoRoot returns the absolute path of the repo root.
func (c *Config) ResolveRepoRoot() (string, error) {
	root := c.Repo.Root
	if root == "" {
		root = "."
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolving repo root %q: %w", root, err)
	}
	return abs, nil
}

// IsExcluded reports whether a path component matches any exclude pattern.
func (c *Config) IsExcluded(name string) bool {
	for _, pat := range c.Repo.Exclude {
		if matched, _ := filepath.Match(pat, name); matched {
			return true
		}
		if strings.Contains(name, pat) {
			return true
		}
	}
	return false
}
