# Chorus Engine Architecture

## Overview

`chorus-engine` is a headless agent execution library — zero UI dependencies, all I/O via async generators and typed event streams. It separates the orchestration engine from presentation so any consumer (CLI, HTTP API, Slack bot, CI runner) can drive it.

## Design Principles

1. **Headless**: No UI dependencies. All I/O is via async generators and event callbacks.
2. **Type-safe**: `strict: true`, discriminated unions everywhere. Every boundary is typed.
3. **Resilient**: Retry with exponential backoff + jitter, per-chunk stream timeouts, checkpoint recovery, graceful degradation.
4. **Observable**: OpenTelemetry-compatible tracing at every lifecycle boundary.
5. **Extensible**: Plugin architecture for LLM providers, tools, middleware, checkpointers, embedders.

## Module Map

```
src/
├── index.ts                # Public API surface (171 exports)
├── agent/
│   ├── loop.ts             # Core ReAct loop (521 lines, streaming generator)
│   ├── loop-utils.ts       # Shared primitives (consumeStream, toolDefsFromTools, executeToolCall)
│   ├── types.ts            # AgentEvent (30+ variants), LoopOptions, Checkpointer, HandoffSignal
│   ├── middleware.ts        # 6 hooks, 4 built-in middleware classes
│   ├── checkpointer.ts     # JsonFileCheckpointer
│   ├── durable-checkpointer.ts  # Event-sourced checkpointing (sync/async/exit modes)
│   ├── postgres-checkpointer.ts # PostgreSQL + JSONB checkpointer
│   ├── hitl.ts             # HitlGate with timeout, disposal, session-scoped approval
│   ├── retry.ts            # RetryPolicy, DEFAULT_RETRY_POLICY, RATE_LIMIT_RETRY_POLICY
│   ├── btw.ts              # BtwQueue for async side-channel messages
│   ├── memory-tools.ts     # createMemoryTools, createSharedMemoryTools
│   └── memory-store.ts     # Embedding-backed memory store
│
├── swarm/
│   ├── orchestrator.ts     # runSwarm — handoff/supervisor execution
│   ├── graph-executor.ts   # runSwarmGraph — DAG wave execution
│   ├── supervisor.ts       # buildSupervisorSwarm
│   ├── group-chat.ts       # runGroupChat — multi-agent debate
│   ├── handoff.ts          # Context-filtered agent handoff
│   ├── session.ts          # Shared state, artifact tools
│   ├── circuit-breaker.ts  # Cost/token/round limit enforcement
│   ├── cost-router.ts      # Budget-aware model selection
│   ├── worktree.ts         # Git worktree isolation for parallel agents
│   ├── validator.ts        # Output validation
│   ├── trace.ts            # SwarmTracer
│   ├── report.ts           # Swarm execution report generation
│   └── presets/            # 4 pre-configured swarm templates
│
├── graph/
│   ├── state-graph.ts      # StateGraph builder + CompiledGraph runtime (838 lines)
│   ├── channel.ts          # Reducer semantics (lastValue, append, sum, setUnion, mapMerge)
│   ├── create-agent.ts     # createAgent — LangGraph-style ReAct agent builder
│   ├── postgres-saver.ts   # PostgresSaver implementing Checkpointer + GraphCheckpointer
│   ├── rest-server.ts      # LangGraph Platform-compatible REST API
│   └── types.ts            # Channel, NodeFn, GraphEvent, GraphInterrupt, Command, Send
│
├── harness/
│   ├── orchestrator.ts     # prepareTaskExecution + prepareTaskExecutionAsync
│   ├── semanticRouter.ts   # Embedding-based intent classification (7 route kinds)
│   ├── router.ts           # Regex-based task classification (fallback)
│   ├── protocol.ts         # buildExecutionProtocol — lifecycle stages per task kind
│   ├── contextAssembler.ts # Deterministic hash-based context bundle versioning
│   ├── workerEngine.ts     # executeWorkers — parallel/pipeline modes with concurrency
│   ├── workerPrompts.ts    # Role-specific system prompts
│   ├── workerPool.ts       # Worker pool management
│   ├── repoIntelligence.ts # Auto-detect package manager, languages, test frameworks
│   ├── projectMemory.ts    # Persistent memory of tasks, decisions, known issues
│   ├── verifier.ts         # verifyTaskCompletion — criteria-based validation
│   ├── observability.ts    # Atomic metrics persistence
│   ├── approvalLog.ts      # NDJSON audit log of HITL decisions
│   └── storage.ts          # Harness run persistence
│
├── skills/
│   ├── harness.ts          # SkillHarness — orchestrator for ASR
│   ├── registry.ts         # SkillRegistry — loading, indexing, metrics, curation
│   ├── router.ts           # routeSkillsForTurn — cosine similarity + budget knapsack
│   ├── embedder.ts         # MiniLM local embeddings + keyword fallback
│   ├── semanticIndex.ts    # Persistent vector index with model versioning
│   ├── loader.ts           # SKILL.md YAML frontmatter parser + loader
│   ├── synthesizer.ts      # TrajectorySynthesizer — LCS alignment + pattern extraction
│   ├── annealer.ts         # k-medoids clustering for consensus trajectory selection
│   ├── executor.ts         # executeSkill — workflow + swarm execution
│   ├── budget.ts           # Token budget allocation (greedy knapsack)
│   ├── swarmAdapter.ts     # Skill swarm config → SwarmConfig adapter
│   ├── middleware.ts        # SkillMiddleware — bridges ASR into agent loop
│   └── types.ts            # SkillDef, PatternDef, ToolTrajectory, SkillMetrics
│
├── guardrails/
│   ├── index.ts            # runGuardrails, shouldHalt, BuiltInGuardrails
│   ├── tiered.ts           # Tiered guardrail execution (input/output/tool)
│   ├── adaptive.ts         # Adaptive guardrail thresholds
│   ├── interceptor.ts      # Tool call interception
│   ├── redaction.ts        # PII + secret redaction
│   └── ner.ts              # Named entity recognition for sensitive data
│
├── telemetry/
│   ├── exporter.ts         # OTLP JSON exporter with batching + retry
│   ├── inprocess.ts        # In-memory span collection for testing
│   ├── redaction.ts        # Span attribute redaction
│   ├── bridge.ts           # Multi-process telemetry bridge
│   └── types.ts            # OTelSpan, SpanKind
│
├── llm/
│   ├── provider.ts         # LLMProvider interface
│   ├── registry.ts         # createProvider, getDefaultProvider
│   ├── config.ts           # Provider name types
│   ├── contextWindows.ts   # Model context window sizes
│   ├── pricing.ts          # Per-model token pricing
│   ├── reasoningParser.ts  # Extract reasoning_content from model output
│   ├── retry.ts            # LLM-specific retry logic
│   ├── ollamaProvider.ts   # Ollama provider implementation
│   └── vllmProvider.ts     # vLLM provider implementation
│
├── tools/
│   ├── filesystem.ts       # Read, write, edit, ls, glob, grep
│   ├── shell.ts            # run_command with safety audit
│   ├── git.ts              # status, diff, log, branch, commit
│   ├── web-search.ts       # Serper + Google CSE
│   ├── safety.ts           # assessCommandSafety, auditCommand
│   ├── todos.ts            # write_todos structured task tracking
│   └── tool.ts             # Tool base class
│
├── mcp/
│   ├── client.ts           # getMcpTools — aggregated MCP tools
│   ├── server.ts           # MCP server wrappers
│   ├── config.ts           # Server configuration (stdio/SSE/HTTP)
│   ├── auth.ts             # OAuth support
│   └── manage.ts           # Server management (add, remove, list, update)
│
├── a2a/
│   ├── adapter.ts          # createSwarmA2AServer
│   ├── server.ts           # JSON-RPC 2.0 HTTP server
│   ├── client.ts           # A2A client for external agents
│   └── types.ts            # AgentCard, etc.
│
├── subagents/
│   ├── delegateTool.ts     # createDelegateTool
│   ├── runtime.ts          # Sub-agent execution runtime
│   ├── planner.ts          # Planner sub-agent
│   ├── builder.ts          # Builder sub-agent
│   └── vapt.ts             # VAPT sub-agent
│
├── context/
│   ├── compaction.ts       # shouldCompact, compactMessages
│   ├── tokenizer.ts        # countTokens, countMessagesTokens
│   └── cache.ts            # Context cache
│
├── evals/
│   ├── runner.ts           # runEvalSuite
│   ├── scorer.ts           # Scoring strategies
│   ├── storage.ts          # Eval run persistence
│   └── types.ts            # EvalSuite, EvalRun, EvalVerdict
│
├── memory/
│   ├── index.ts            # Memory system public API
│   └── compression.ts      # Memory compression strategies
│
├── settings/
│   ├── storage.ts          # configureEngine, EngineConfig
│   └── providers.ts        # Provider configuration persistence
│
├── session/
│   ├── manager.ts          # SessionManager — debounced persistence
│   ├── storage.ts          # JSON file-based session storage
│   └── types.ts            # SessionData, SessionMetadata
│
├── channels/
│   ├── broadcaster.ts      # EventBroadcaster — SSE/WS fan-out hub
│   ├── server.ts           # ChannelServer — HTTP SSE endpoint
│   └── types.ts            # ChannelEvent, ChannelSession
│
├── agents/
│   ├── generator.ts        # generateAgentDef — LLM-powered agent definition generation
│   ├── loader.ts           # Load agent JSON definitions from disk
│   ├── storage.ts          # Save/delete agent definitions
│   ├── resolver.ts         # Resolve tool names → live AgentTool instances
│   └── types.ts            # AgentDef
│
└── prompts/
    └── system.ts           # SYSTEM_PROMPT, PLANNER_PROMPT, VAPT_PROMPT, BUILDER_PROMPT
```

## Agent Loop — Core Execution Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  runAgentLoop() — async generator yields AgentEvent[]   │
│                                                         │
│  1. Load checkpoint → restore or start fresh             │
│  2. Drain BtwQueue → inject side-channel messages        │
│  3. Run input guardrails → halt on violation             │
│  4. beforeRound middleware (prioritized, parallel)       │
│  5. Rebuild tools + system prompt (dynamic routing)     │
│  6. maybeCompact middleware (first-wins)                 │
│  7. consumeStream() with retry + per-chunk timeout      │
│     ├── token events → yield to consumer                │
│     ├── thinking events → yield to consumer             │
│     ├── stream-done → extract response                  │
│     └── stream-error → retry (pre-token) or fatal       │
│  8. Merge assistant message into history                │
│  9. If no tool calls:                                   │
│     ├── Validate against outputSchema (Zod)             │
│     ├── Run output guardrails                           │
│     ├── Save checkpoint                                 │
│     └── Yield done event                                │
│  10. If tool calls:                                     │
│     ├── HITL gate → pause if policy requires            │
│     ├── beforeTool middleware (cancel + substitute)     │
│     ├── Tool guardrails → halt on violation             │
│     ├── executeToolCall() with retry                    │
│     ├── afterTool middleware (transform results)        │
│     ├── Catch HandoffSignal → yield handoff + done     │
│     └── Append tool results to history                  │
│  11. afterRound middleware (prioritized, parallel)      │
│  12. Save checkpoint → next round                       │
└─────────────────────────────────────────────────────────┘
```

## StateGraph — DAG Execution Flow

```
input: Partial<State>
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CompiledGraph.stream()                                  │
│                                                         │
│  1. Build initial state from channels + input           │
│  2. Load checkpoint → restore state if exists           │
│  3. Compute initial runnable nodes (from START edges)   │
│                                                         │
│  For each wave:                                         │
│  4. Check recursion limit, graph timeout, abort         │
│  5. State fingerprint → detect infinite loops           │
│  6. Execute all runnable nodes in parallel:             │
│     ├── beforeNode hooks                               │
│     ├── Node function with per-node timeout             │
│     ├── Catch GraphInterrupt → yield interrupt + save   │
│     └── afterNode hooks                                │
│  7. Yield node_end events with durations                │
│  8. Extract Send objects from outputs → fan-out         │
│  9. Merge outputs into state via channel reducers       │
│  10. Check finish points → yield end if reached         │
│  11. Compute next runnable from static + conditional    │
│     edges + Send targets                               │
│  12. Save checkpoint → repeat until no runnable nodes   │
│                                                         │
│  13. Deadlock detection: finish points unreachable?     │
│  14. Yield end with final state                         │
└─────────────────────────────────────────────────────────┘
```

## Swarm — Multi-Agent Execution Flow

```
SwarmConfig
    │
    ├── executionModel: "handoff" (default)
    │   │
    │   ├── Create session + shared state
    │   ├── WHILE activeAgent ≠ null:
    │   │   ├── Check circuit breaker
    │   │   ├── Build agent context (filtered)
    │   │   ├── runAgentLoop() for active agent
    │   │   ├── Process handoff tool calls → switch agent
    │   │   └── Update shared artifacts
    │   └── Validate output → yield swarm-done
    │
    ├── executionModel: "graph"
    │   │
    │   ├── Build adjacency from agent dependsOn
    │   ├── Compute waves topologically
    │   ├── FOR each wave:
    │   │   ├── Spawn agents in parallel (isolated worktrees)
    │   │   ├── Collect artifacts
    │   │   └── Circuit-break on missing required artifacts
    │   └── Merge artifacts → yield swarm-done
    │
    ├── executionModel: "supervisor"
    │   │
    │   ├── Supervisor agent runs first
    │   ├── Dynamically routes to specialist agents
    │   ├── Filters context to prevent token bloat
    │   └── Synthesizes final output
    │
    └── executionModel: "group-chat"
        │
        ├── All agents see full conversation
        ├── Each responds in turn
        ├── Merge strategy: vote | concatenate | first-success
        └── Yield final merged response
```

## Harness — Task Classification & Worker Routing

```
User Message
    │
    ▼
┌──────────────────────────────────────────┐
│  SemanticTaskRouter (primary)            │
│  1. Embed query → 384d vector           │
│  2. Cosine similarity vs 7 route        │
│     prototypes (multi-vector)            │
│  3. If confidence ≥ threshold → route   │
│  4. Else → regex fallback               │
│                                          │
│  Output: TaskRoute                       │
│  { kind, lane, path, confidence }       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  buildExecutionProtocol(route)           │
│  → lifecycle stages per task kind        │
│  → suggested checks (npm test, build)    │
│  → delegation policy                     │
│  → final response contract               │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  executeWorkers()                        │
│                                          │
│  Parallel mode:                          │
│    workers run concurrently (bounded)    │
│    each writes to shared context         │
│                                          │
│  Pipeline mode:                          │
│    researcher → planner → coder → ...    │
│    each receives accumulated context     │
└──────────────────────────────────────────┘
```

## Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| Network (LLM) | Retry 3× with exponential backoff; fatal after token emission (consistency) |
| Stream | Per-chunk timeout (`Promise.race`); abort signal propagation |
| Tool | Tool-level error → yield as tool result, not crash; retry via `withRetry` |
| HITL | Timeout → reject with `HitlGateTimeoutError`; dispose → reject with `HitlGateDisposedError` |
| Checkpointer | Best-effort; never block agent loop |
| Telemetry | Best-effort; never block agent loop |
| Guardrails | Configurable halt severity (info/warning/critical); runAll mode |
| Circuit Breaker | Abort all in-flight requests + checkpoint state + yield circuit-break event |

## Thread Safety

- **Agent Loop**: Single-threaded per instance. Do not share across concurrent loops.
- **Swarm**: Router is stateless; agent instances are per-invocation.
- **Harness**: Workers run concurrently but each has isolated context. Shared state via `SharedWorkerContext`.
- **Checkpointer**: Atomic writes (tmp + rename) prevent corruption from concurrent writes.
- **StateGraph**: Nodes execute in parallel within a wave; channel reducers ensure deterministic merge.

## Extension Points

| Extension | Interface | Example |
|-----------|-----------|---------|
| LLM Provider | `LLMProvider` (generate, stream, streamWithTools, health) | `examples/03-custom-provider.ts` |
| Tool | `AgentTool` (name, schema, invoke) | Any tool definition |
| Middleware | `AgentMiddleware` (6 hooks + priority) | `examples/05-middleware-pipeline.ts` |
| Checkpointer | `Checkpointer` (save, load, list, fork, delete) | `PostgresSaver`, custom |
| GraphCheckpointer | `GraphCheckpointer` (extends Checkpointer) | `PostgresSaver` |
| Skill Embedder | `SkillEmbedder` (embed, modelId, dimensions) | `MiniLMEmbedder`, custom |
| Telemetry Exporter | `OTelSpan` export | OTLP, custom collector |
| Shared Context | `SharedWorkerContext` (get, set, has, entries) | In-memory, Redis, etc. |
