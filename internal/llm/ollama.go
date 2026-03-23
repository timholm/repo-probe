package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaProvider implements Provider using the Ollama native REST API.
type OllamaProvider struct {
	baseURL    string
	model      string
	embedModel string
	maxTokens  int
	timeout    time.Duration
	client     *http.Client
}

// NewOllamaProvider creates a new Ollama provider.
func NewOllamaProvider(baseURL, model, embedModel string, maxTokens int, timeout time.Duration) *OllamaProvider {
	return &OllamaProvider{
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      model,
		embedModel: embedModel,
		maxTokens:  maxTokens,
		timeout:    timeout,
		client:     &http.Client{Timeout: timeout},
	}
}

func (p *OllamaProvider) ModelName() string { return p.model }

// ollamaChatRequest is the request body for POST /api/chat.
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  ollamaOptions   `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	NumPredict  int     `json:"num_predict,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

// ollamaChatResponse is the response body from POST /api/chat (non-streaming).
type ollamaChatResponse struct {
	Model   string        `json:"model"`
	Message ollamaMessage `json:"message"`
	Done    bool          `json:"done"`
}

// ollamaEmbedRequest is the request body for POST /api/embeddings.
type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Complete sends a non-streaming chat completion request to Ollama.
func (p *OllamaProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	msgs := make([]ollamaMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = ollamaMessage{Role: m.Role, Content: m.Content}
	}

	body := ollamaChatRequest{
		Model:    p.model,
		Messages: msgs,
		Stream:   false,
		Options: ollamaOptions{
			NumPredict:  p.maxTokens,
			Temperature: req.Temperature,
		},
	}

	var resp ollamaChatResponse
	if err := p.doJSON(ctx, "POST", "/api/chat", body, &resp); err != nil {
		return nil, fmt.Errorf("ollama chat: %w", err)
	}

	return &CompletionResponse{
		Content:      resp.Message.Content,
		FinishReason: "stop",
		Usage: Usage{
			PromptTokens:     CountTokens(joinMessages(req.Messages)),
			CompletionTokens: CountTokens(resp.Message.Content),
		},
	}, nil
}

// Stream sends a streaming chat completion request to Ollama.
// Returns a channel that emits token chunks and is closed when done.
func (p *OllamaProvider) Stream(ctx context.Context, req CompletionRequest) (<-chan string, error) {
	msgs := make([]ollamaMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = ollamaMessage{Role: m.Role, Content: m.Content}
	}

	body := ollamaChatRequest{
		Model:    p.model,
		Messages: msgs,
		Stream:   true,
		Options: ollamaOptions{
			NumPredict:  p.maxTokens,
			Temperature: req.Temperature,
		},
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/api/chat", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama stream request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	ch := make(chan string, 64)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var chunk ollamaChatResponse
			if err := json.Unmarshal(line, &chunk); err != nil {
				continue
			}
			if chunk.Message.Content != "" {
				select {
				case ch <- chunk.Message.Content:
				case <-ctx.Done():
					return
				}
			}
			if chunk.Done {
				return
			}
		}
	}()

	return ch, nil
}

// Embed returns an embedding vector for the given text.
func (p *OllamaProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	body := ollamaEmbedRequest{
		Model:  p.embedModel,
		Prompt: text,
	}
	var resp ollamaEmbedResponse
	if err := p.doJSON(ctx, "POST", "/api/embeddings", body, &resp); err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	return resp.Embedding, nil
}

// doJSON sends a JSON request and decodes the JSON response.
func (p *OllamaProvider) doJSON(ctx context.Context, method, path string, body, result interface{}) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshaling request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(1<<uint(attempt-1)) * time.Second):
			}
			// Reset body for retry.
			req.Body = io.NopCloser(bytes.NewReader(data))
		}

		resp, err := p.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			lastErr = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
			if resp.StatusCode < 500 {
				return lastErr // don't retry 4xx
			}
			continue
		}

		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decoding response: %w", err)
		}
		return nil
	}

	return fmt.Errorf("after 3 attempts: %w", lastErr)
}

func joinMessages(msgs []Message) string {
	var parts []string
	for _, m := range msgs {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n")
}
