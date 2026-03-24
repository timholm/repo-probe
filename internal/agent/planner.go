package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/timholm/repo-probe/internal/llm"
)

// Plan holds the decomposed subgoals for a query.
type Plan struct {
	Query    string
	Subgoals []string
}

// Planner decomposes complex queries into actionable subgoals.
type Planner struct {
	provider llm.Provider
}

// NewPlanner creates a new Planner backed by the given LLM provider.
func NewPlanner(provider llm.Provider) *Planner {
	return &Planner{provider: provider}
}

// Decompose asks the LLM to break a query into subgoals.
// On any error or unparseable response, returns a single-subgoal plan.
func (p *Planner) Decompose(ctx context.Context, query string) *Plan {
	if len(strings.TrimSpace(query)) < 20 {
		// Short queries don't need decomposition.
		return &Plan{Query: query, Subgoals: []string{query}}
	}

	prompt := llm.PlannerPrompt(query)
	resp, err := p.provider.Complete(ctx, llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
	})
	if err != nil {
		return p.fallback(query)
	}

	plan, err := parsePlan(query, resp.Content)
	if err != nil {
		return p.fallback(query)
	}
	return plan
}

type planResponse struct {
	Subgoals []string `json:"subgoals"`
}

// parsePlan extracts subgoals from the LLM's JSON response.
func parsePlan(query, raw string) (*Plan, error) {
	// Find the JSON object in the response (model might add preamble).
	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("no JSON found in response")
	}

	var pr planResponse
	if err := json.Unmarshal([]byte(raw[start:end+1]), &pr); err != nil {
		return nil, fmt.Errorf("parsing plan JSON: %w", err)
	}
	if len(pr.Subgoals) == 0 {
		return nil, fmt.Errorf("empty subgoals list")
	}

	// Clamp to 5 subgoals.
	if len(pr.Subgoals) > 5 {
		pr.Subgoals = pr.Subgoals[:5]
	}

	return &Plan{Query: query, Subgoals: pr.Subgoals}, nil
}

func (p *Planner) fallback(query string) *Plan {
	return &Plan{Query: query, Subgoals: []string{query}}
}
