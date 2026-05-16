# Semantic Router

The `SemanticTaskRouter` provides embedding-based intent classification with confidence thresholds and hybrid fallback. It classifies user requests into one of 7 task kinds to determine the correct execution strategy.

## Why Semantic Routing?

Keyword-based routing (regex) fails on:
- **Paraphrasing**: "What's the newest React version?" vs "latest React"
- **Ambiguity**: "Check the auth code" (review? debug? research?)
- **Compound intents**: "Find and fix the bug in auth"

Embedding-based classification maps the query into the same vector space as route prototypes, enabling fuzzy matching.

## Usage

```typescript
import { SemanticTaskRouter, routeTaskSemantic } from "chorus-engine/harness";

// One-shot classification
const route = await routeTaskSemantic({
  text: "Debug the login flow",
  expandedText: "The login endpoint returns 500 after the recent merge",
});

// Reusable router with custom threshold
const router = new SemanticTaskRouter({
  confidenceThreshold: 0.75, // default: 0.55
  embedder: customEmbedder,   // optional, defaults to MiniLM
});

const result = await router.route({
  text: "Debug the login flow",
  expandedText: "...",
});
// {
//   kind: "debug",
//   confidence: 0.91,
//   method: "semantic",
//   lane: "foreground_sync",
//   path: "tool_or_single_worker_path",
//   requiresResearch: false,
//   canParallelize: false,
//   usesCheapTriage: false,
//   matchedLabel: "debug"
// }
```

## Route Labels

| Kind | Lane | Path | Requires Research | Can Parallelize |
|------|------|------|:---:|:---:|
| `answer_only` | `cheap_triage` | `direct_agent_path` | No | No |
| `inspect_only` | `cheap_triage` | `direct_agent_path` | No | No |
| `single_file_edit` | `foreground_sync` | `tool_or_single_worker_path` | No | No |
| `multi_file_edit` | `foreground_sync` | `parallel_multi_worker_path` | No | Yes |
| `debug` | `foreground_sync` | `tool_or_single_worker_path` | No | No |
| `research` | `foreground_sync` | `research_then_plan_path` | Yes | No |
| `project_phase` | `background_async` | `background_or_batch_path` | No | No |

## How It Works

### Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  Step 1: Embed query                 │
│  → MiniLM feature extraction         │
│  → 384-dimensional vector            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Step 2: Multi-vector scoring        │
│  Cosine similarity against ALL       │
│  prototype vectors per route         │
│  (8 examples per route = 56 vectors) │
│  → Take MAX per route                │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Step 3: Select best match           │
│  If confidence ≥ threshold:          │
│    → semantic route                  │
│  Else:                               │
│    → regex fallback (never blocks)   │
└──────────────────────────────────────┘
```

### Multi-Vector Routing

Each route has multiple prototype vectors (one per example utterance), achieving ~15% accuracy improvement over single-vector routing. The MAX similarity across all prototypes is used for the final score.

### Fallback Strategy

1. If semantic confidence < threshold → try regex keyword matching
2. If regex matches → use regex result with `method: "fallback"`
3. If no regex match → route to `answer_only` with `method: "fallback"`

This ensures the router **never blocks** — worst case, it escalates to a generalist response.

## Multi-Label Scoring

Get confidence scores for all routes simultaneously:

```typescript
const scores = await router.score({
  text: "Find and fix the bug in authentication",
  expandedText: "",
});
// [
//   { label: "debug", confidence: 0.85 },
//   { label: "single_file_edit", confidence: 0.62 },
//   { label: "multi_file_edit", confidence: 0.41 },
//   { label: "research", confidence: 0.22 },
//   ...
// ]
```

Use cases:
- **Ambiguity detection**: Top-2 confidence gap < 0.2 → ask the user to clarify
- **Multi-intent routing**: Combine top-2 routes (e.g., research + edit)
- **Threshold tuning**: Collect production scores, adjust per-route thresholds

## Confidence Thresholds per Route

| Route | Default | Rationale |
|-------|---------|-----------|
| `answer_only` | 0.55 | Low risk; can always ask clarifying question |
| `inspect_only` | 0.55 | Low risk; read-only operations |
| `single_file_edit` | 0.55 | Medium risk; scoped to one file |
| `multi_file_edit` | 0.55 | Higher risk; multiple files affected |
| `research` | 0.55 | Medium risk; may waste an API call |
| `debug` | 0.55 | Medium risk; wrong diagnosis wastes time |
| `project_phase` | 0.55 | High risk; expensive operation |

The global confidence threshold (passed via `confidenceThreshold` option) applies uniformly. For per-route thresholds, check `route.kind` post-classification and escalate if the confidence for that kind is below your threshold.

## Performance

| Metric | Value |
|--------|-------|
| Latency | ~50ms (local MiniLM embedding on first call; cached thereafter) |
| Accuracy | ~94% (per CoRouter research) |
| Cost | ~60% less than LLM-based classification |
| Model | `onnx-community/all-MiniLM-L6-v2-ONNX` (384 dimensions) |

## Fallback Embedder

If MiniLM cannot be loaded (offline, incompatible platform), the router falls back to a deterministic keyword hashing embedder that maps weighted term frequency into a 256-dimensional vector — functional but lower accuracy. Set `CHORUS_EMBEDDER=keyword` to force this mode.

```bash
CHORUS_EMBEDDER=keyword node my-app.js
```

## Thread Safety

`route()` and `score()` are stateless reads against pre-computed prototype vectors. Safe to call from multiple contexts concurrently.
