package config

import "time"

const (
	DefaultOllamaURL           = "http://localhost:11434"
	DefaultModel               = "qwen2.5-coder:7b"
	DefaultMaxSteps            = 20
	DefaultServerAddr          = ":8080"
	DefaultMaxTokens           = 4096
	DefaultContextWindowTokens = 8192
	DefaultTemperature         = 0.1
	DefaultProvider            = "ollama"
	DefaultCacheDir            = ".repo-probe-cache"
	DefaultEmbedModel          = "nomic-embed-text"
	DefaultEmbedDimensions     = 768
)

var DefaultExcludePatterns = []string{
	".git",
	"vendor",
	"node_modules",
	"*.pb.go",
	"*.min.js",
	"*.min.css",
	"dist",
	"build",
	".repo-probe-cache",
}

var DefaultReadTimeout = 30 * time.Second
var DefaultWriteTimeout = 120 * time.Second
var DefaultLLMTimeout = 300 * time.Second
