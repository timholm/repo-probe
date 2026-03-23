package llm

import (
	"context"
	"fmt"

	"github.com/factory/repo-probe/internal/config"
)

// Message is a single turn in a conversation.
type Message struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// CompletionRequest is a request to generate a completion.
type CompletionRequest struct {
	Messages    []Message
	Temperature float64
	Stream      bool
}

// CompletionResponse is the result of a completion.
type CompletionResponse struct {
	Content      string
	FinishReason string
	Usage        Usage
}

// Usage tracks token consumption.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
}

// Provider is the interface implemented by all LLM backends.
type Provider interface {
	// Complete generates a completion for the given messages.
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)

	// Stream generates a streaming completion, returning a channel of token chunks.
	Stream(ctx context.Context, req CompletionRequest) (<-chan string, error)

	// Embed returns an embedding vector for the given text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// ModelName returns the model identifier being used.
	ModelName() string
}

// NewProvider creates the appropriate Provider from a config.
func NewProvider(cfg *config.LLMConfig) (Provider, error) {
	switch cfg.Provider {
	case "ollama":
		return NewOllamaProvider(
			cfg.BaseURL,
			cfg.Model,
			cfg.EmbedModel,
			cfg.MaxTokens,
			cfg.Timeout,
		), nil
	case "openai":
		return NewOpenAIProvider(
			cfg.BaseURL,
			cfg.Model,
			cfg.EmbedModel,
			cfg.APIKey,
			cfg.MaxTokens,
			cfg.Timeout,
		), nil
	default:
		return nil, fmt.Errorf("unknown LLM provider %q; supported: ollama, openai", cfg.Provider)
	}
}

// CollectStream reads all chunks from a stream channel and concatenates them.
func CollectStream(ch <-chan string) string {
	var sb []byte
	for chunk := range ch {
		sb = append(sb, chunk...)
	}
	return string(sb)
}
