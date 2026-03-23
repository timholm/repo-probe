package llm

import (
	"fmt"
	"strings"
)

// ToolSchema describes a tool for injection into the system prompt.
type ToolSchema struct {
	Name        string
	Description string
	ArgsSchema  string // human-readable argument list
}

// SystemPrompt returns the static system prompt with the tool list embedded.
func SystemPrompt(tools []ToolSchema) string {
	var sb strings.Builder
	sb.WriteString(`You are repo-probe, an expert code analysis agent that answers questions about software repositories.

You have access to the following tools to explore the codebase:

`)
	for _, t := range tools {
		sb.WriteString(fmt.Sprintf("  %s(%s)\n    %s\n\n", t.Name, t.ArgsSchema, t.Description))
	}
	sb.WriteString(`
## Response Format

You MUST respond using EXACTLY this format, one step at a time:

Thought: <your reasoning about what to look for next>
Action: <tool_name> {"arg1": "val1", "arg2": "val2"}

OR when you have gathered enough information:

Answer: <your complete, accurate answer to the question>

## Rules

- Always start with a Thought explaining your reasoning
- Use tools to gather evidence before answering
- Reference specific file paths and line numbers in your answer
- If a tool returns an error, try an alternative approach
- Be concise but complete in your answer
- NEVER guess - if you don't know, use tools to find out
`)
	return sb.String()
}

// PlannerPrompt returns a prompt asking the LLM to decompose a query into subgoals.
func PlannerPrompt(query string) string {
	return fmt.Sprintf(`You are a code analysis planning agent. Break down the following question into 1-5 concrete subgoals that will help answer it completely.

Question: %s

Respond with ONLY a JSON object in this exact format:
{"subgoals": ["subgoal 1", "subgoal 2", ...]}

Keep subgoals specific and actionable. For simple questions, one subgoal is fine.`, query)
}

// CompressionPrompt returns a prompt to summarize a list of observations.
func CompressionPrompt(observations []string) string {
	var sb strings.Builder
	sb.WriteString("Summarize the following code exploration findings into 3-5 concise sentences. Preserve all specific file names, function names, and line numbers mentioned.\n\nFindings:\n")
	for i, obs := range observations {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, obs))
	}
	sb.WriteString("\nSummary:")
	return sb.String()
}

// SynthesisPrompt returns a prompt asking the LLM to synthesize an answer from findings.
func SynthesisPrompt(query string, findings string) string {
	return fmt.Sprintf(`Based on the following findings from exploring the codebase, provide a complete and accurate answer to this question:

Question: %s

Findings:
%s

Answer the question directly and specifically. Reference file paths and line numbers where relevant.`, query, findings)
}

// ReActMessage builds the user turn that drives the ReAct loop.
// It includes the current question, plan, and a reminder of the expected format.
func ReActMessage(query string, subgoals []string, stepNum, maxSteps int) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Question: %s\n\n", query))
	if len(subgoals) > 0 {
		sb.WriteString("Objectives to address:\n")
		for i, sg := range subgoals {
			sb.WriteString(fmt.Sprintf("  %d. %s\n", i+1, sg))
		}
		sb.WriteString("\n")
	}
	sb.WriteString(fmt.Sprintf("Step %d of %d. Respond with Thought+Action or Answer:\n", stepNum, maxSteps))
	return sb.String()
}

// CountTokens estimates the token count of a string.
// Uses a simple heuristic: ~4 chars per token (reasonable for code/English).
func CountTokens(s string) int {
	if len(s) == 0 {
		return 0
	}
	return (len(s) + 3) / 4
}

// TruncateToTokens truncates a string to approximately maxTokens tokens.
func TruncateToTokens(s string, maxTokens int) string {
	maxChars := maxTokens * 4
	if len(s) <= maxChars {
		return s
	}
	return s[:maxChars] + "\n[... truncated ...]"
}
