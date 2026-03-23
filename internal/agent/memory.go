package agent

import (
	"fmt"
	"strings"
	"sync"
)

// MemoryEntry is a single keyed finding stored in working memory.
type MemoryEntry struct {
	Key       string
	Value     string
	Source    string  // tool name or file path that produced this
	Relevance float32 // higher = more relevant to the query
}

// Memory is a bounded working memory for the agent during a single query.
// It stores key findings so they can be included in the synthesis prompt
// even after context compression.
type Memory struct {
	mu      sync.RWMutex
	entries []MemoryEntry
	maxSize int
	query   string // used for relevance scoring
}

// NewMemory creates a new Memory with the given capacity.
func NewMemory(maxSize int, query string) *Memory {
	if maxSize < 10 {
		maxSize = 10
	}
	return &Memory{
		maxSize: maxSize,
		query:   strings.ToLower(query),
	}
}

// Store adds or updates an entry in memory.
// If memory is full, the least-relevant entry is evicted.
func (m *Memory) Store(key, value, source string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	rel := m.score(key + " " + value)

	// Update existing entry with same key.
	for i, e := range m.entries {
		if e.Key == key {
			m.entries[i].Value = value
			m.entries[i].Source = source
			m.entries[i].Relevance = rel
			return
		}
	}

	entry := MemoryEntry{
		Key:       key,
		Value:     value,
		Source:    source,
		Relevance: rel,
	}

	if len(m.entries) < m.maxSize {
		m.entries = append(m.entries, entry)
		return
	}

	// Evict least-relevant entry.
	minIdx := 0
	for i, e := range m.entries {
		if e.Relevance < m.entries[minIdx].Relevance {
			minIdx = i
		}
	}
	if rel > m.entries[minIdx].Relevance {
		m.entries[minIdx] = entry
	}
}

// Snapshot returns the top-10 entries formatted as a text block.
func (m *Memory) Snapshot() string {
	m.mu.RLock()
	entries := make([]MemoryEntry, len(m.entries))
	copy(entries, m.entries)
	m.mu.RUnlock()

	if len(entries) == 0 {
		return ""
	}

	// Sort by relevance descending (simple insertion sort for small N).
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].Relevance > entries[j-1].Relevance; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}

	limit := 10
	if len(entries) < limit {
		limit = len(entries)
	}

	var sb strings.Builder
	sb.WriteString("## Key Findings\n")
	for _, e := range entries[:limit] {
		sb.WriteString(fmt.Sprintf("- [%s] %s: %s\n", e.Source, e.Key, truncate(e.Value, 200)))
	}
	return sb.String()
}

// AllValues returns all entry values in no particular order.
func (m *Memory) AllValues() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	vals := make([]string, len(m.entries))
	for i, e := range m.entries {
		vals[i] = e.Value
	}
	return vals
}

// Len returns the number of entries.
func (m *Memory) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// score computes a simple relevance score based on query term overlap.
func (m *Memory) score(text string) float32 {
	if m.query == "" {
		return 0.5
	}
	text = strings.ToLower(text)
	queryWords := strings.Fields(m.query)
	if len(queryWords) == 0 {
		return 0.5
	}
	matches := 0
	for _, w := range queryWords {
		if len(w) < 3 {
			continue
		}
		if strings.Contains(text, w) {
			matches++
		}
	}
	return float32(matches) / float32(len(queryWords))
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
