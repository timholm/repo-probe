package agent

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/factory/repo-probe/internal/llm"
)

// StepKind classifies a step in the ReAct trajectory.
type StepKind string

const (
	StepThought  StepKind = "thought"
	StepAction   StepKind = "action"
	StepObserve  StepKind = "observe"
	StepAnswer   StepKind = "answer"
	StepCompress StepKind = "compress"
)

// Step is a single entry in the agent's reasoning trajectory.
type Step struct {
	Kind       StepKind   `json:"kind"`
	Content    string     `json:"content"`
	ToolName   string     `json:"tool_name,omitempty"`
	ToolArgs   string     `json:"tool_args,omitempty"`
	ToolError  bool       `json:"tool_error,omitempty"`
	Timestamp  time.Time  `json:"timestamp"`
	TokensUsed int        `json:"tokens_used,omitempty"`
}

// Trajectory records the full reasoning chain for a query.
type Trajectory struct {
	ID        string    `json:"id"`
	Query     string    `json:"query"`
	RepoRoot  string    `json:"repo_root"`
	Steps     []Step    `json:"steps"`
	Answer    string    `json:"answer"`
	CreatedAt time.Time `json:"created_at"`
	Duration  time.Duration `json:"duration_ns"`
}

// NewTrajectory creates a new empty trajectory.
func NewTrajectory(id, query, repoRoot string) *Trajectory {
	return &Trajectory{
		ID:        id,
		Query:     query,
		RepoRoot:  repoRoot,
		CreatedAt: time.Now(),
	}
}

// Append adds a step to the trajectory.
func (t *Trajectory) Append(s Step) {
	if s.Timestamp.IsZero() {
		s.Timestamp = time.Now()
	}
	t.Steps = append(t.Steps, s)
}

// StepCount returns the number of steps.
func (t *Trajectory) StepCount() int { return len(t.Steps) }

// TokensUsed returns the total tokens used across all steps.
func (t *Trajectory) TokensUsed() int {
	total := 0
	for _, s := range t.Steps {
		total += s.TokensUsed
	}
	return total
}

// LastObservation returns the content of the most recent observe step.
func (t *Trajectory) LastObservation() string {
	for i := len(t.Steps) - 1; i >= 0; i-- {
		if t.Steps[i].Kind == StepObserve {
			return t.Steps[i].Content
		}
	}
	return ""
}

// ForPrompt renders the trajectory as a string for injection into the LLM prompt.
// It drops oldest steps to stay within maxTokens. When steps are dropped,
// a summary placeholder is prepended.
func (t *Trajectory) ForPrompt(maxTokens int) string {
	if len(t.Steps) == 0 {
		return ""
	}

	steps := t.Steps
	rendered := renderSteps(steps)

	// If within budget, return as-is.
	if llm.CountTokens(rendered) <= maxTokens {
		return rendered
	}

	// Drop oldest steps until we fit, keeping at least the last 3.
	for len(steps) > 3 && llm.CountTokens(renderSteps(steps)) > maxTokens {
		steps = steps[1:]
	}

	prefix := "[Earlier steps omitted due to context length]\n\n"
	return prefix + renderSteps(steps)
}

func renderSteps(steps []Step) string {
	var sb strings.Builder
	for _, s := range steps {
		switch s.Kind {
		case StepThought:
			sb.WriteString(fmt.Sprintf("Thought: %s\n", s.Content))
		case StepAction:
			sb.WriteString(fmt.Sprintf("Action: %s %s\n", s.ToolName, s.ToolArgs))
		case StepObserve:
			if s.ToolError {
				sb.WriteString(fmt.Sprintf("Observation: [Error] %s\n", s.Content))
			} else {
				sb.WriteString(fmt.Sprintf("Observation: %s\n", s.Content))
			}
		case StepAnswer:
			sb.WriteString(fmt.Sprintf("Answer: %s\n", s.Content))
		case StepCompress:
			sb.WriteString(fmt.Sprintf("[Compressed: %s]\n", s.Content))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// Marshal returns the trajectory as JSON bytes.
func (t *Trajectory) Marshal() ([]byte, error) {
	return json.Marshal(t)
}

// Observations returns all observation contents in order.
func (t *Trajectory) Observations() []string {
	var obs []string
	for _, s := range t.Steps {
		if s.Kind == StepObserve && !s.ToolError {
			obs = append(obs, s.Content)
		}
	}
	return obs
}
