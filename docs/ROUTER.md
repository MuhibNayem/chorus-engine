# Semantic Router

The `SemanticTaskRouter` provides embedding-based intent classification with confidence thresholds and hybrid fallback. It lives in `src/harness/router.ts`.

## Why Semantic Routing?

Keyword-based routing (regex) fails on:
- Paraphrasing: "What's the newest React version?" vs "latest React"
- Ambiguity: "Check the auth code" (review? debug? research?)
- Compound intents: "Find and fix the bug in auth"

Embedding-based classification maps the query into the same vector space as route prototypes, enabling fuzzy matching.

## Usage

```typescript
import { SemanticTaskRouter, routeTaskSemantic } from "chorus-engine/harness";

// One-shot classification
const route = await routeTaskSemantic({ text: "Debug the login flow" });

// Reusable router with custom threshold
const router = new SemanticTaskRouter({
  confidenceThreshold: 0.75, // default: 0.5
});

const result = await router.route({ text: "Debug the login flow" });
console.log(result);
// {
//   kind: "debug",
//   confidence: 0.91,
//   method: "semantic",
//   lane: "analyze",
//   path: "agent",
//   requiresResearch: false,
//   canParallelize: false,
//   usesCheapTriage: false
// }
```

## Route Labels

| Label | Description | Lane | Path | Research? |
|-------|-------------|------|------|-----------|
| `answer_only` | Simple Q&A | `chat` | `agent` | No |
| `single_file_edit` | One-file change | `edit` | `agent` | No |
| `multi_file_edit` | Cross-file change | `edit` | `harness` | No |
| `research` | Needs external info | `research` | `harness` | Yes |
| `debug` | Diagnostic task | `analyze` | `agent` | No |
| `project_phase` | Entire-project work | `phase` | `harness` | No |

## Confidence Thresholds

Different routes have different tolerance for misclassification:

| Route | Threshold | Rationale |
|-------|-----------|-----------|
| `answer_only` | 0.60 | Low risk; can always ask clarifying question |
| `research` | 0.70 | Medium risk; wasted API call if wrong |
| `debug` | 0.75 | High risk; wrong diagnosis wastes time |
| `project_phase` | 0.80 | Very high risk; expensive operation |
| `OOD` | 0.50 | Catch-all; prefer false positive |

## Multi-Label Scoring

Get confidence for all routes:

```typescript
const scores = await router.score({ text: "Find and fix the bug" });
// [
//   { label: "debug", confidence: 0.85 },
//   { label: "research", confidence: 0.42 },
//   { label: "single_file_edit", confidence: 0.31 },
//   ...
// ]
```

Useful for:
- **Ambiguity detection**: Top-2 gap < 0.2 → ask clarifying question
- **Multi-intent routing**: Combine top-2 routes (e.g., research + edit)
- **Threshold tuning**: Collect production scores, adjust τ

## Fallback Strategy

When semantic confidence is below threshold:

1. Try regex keyword matching
2. If regex matches, use regex result with `method: "fallback"`
3. If no match, route to `OOD` with `method: "fallback"`

This ensures the router **never blocks** — worst case, it escalates to a human or generalist agent.

## Performance

| Metric | Value |
|--------|-------|
| Latency | ~50ms (local embedding) |
| Accuracy | ~94% (research benchmark) |
| Cost | ~60% less than LLM-based classification |

## Implementation Notes

- Uses `SkillEmbedder` from `src/skills/embedder.ts` for embedding generation
- Cosine similarity computed in-memory
- Route prototypes are static strings (no training required)
- Thread-safe: `route()` and `score()` are stateless reads
