/**
 * StateGraph — Typed-State Graph Builder & Compiled Runtime
 *
 * Builds on LangGraph's Pregel architecture:
 *   - Nodes read state and return partial updates
 *   - Channels merge updates deterministically
 *   - Edges route execution flow
 *   - Parallel wave execution for independent nodes
 *   - Checkpointing after each wave for durability
 *   - Interrupts pause at node boundaries for human input
 *
 * Usage:
 *   const graph = new StateGraph({
 *     messages: append<ChatMessage>(),
 *     answer: lastValue<string>(),
 *   });
 *
 *   graph.addNode("agent", async (state) => { ... });
 *   graph.addNode("tools", async (state) => { ... });
 *   graph.addEdge(START, "agent");
 *   graph.addConditionalEdges("agent", (state) =>
 *     state.toolCalls?.length ? "tools" : END
 *   );
 *   graph.addEdge("tools", "agent");
 *
 *   const app = graph.compile({ checkpointer, recursionLimit: 25 });
 *   const result = await app.invoke({ messages: [{ role: "user", content: "hi" }] });
 */

import type {
  Channel,
  NodeFn,
  NodeConfig,
  StaticEdge,
  ConditionalEdge,
  CompileOptions,
  RunConfig,
  GraphEvent,
  GraphCheckpoint,
  GraphCheckpointer,
  Command,
  Send,
} from "./types.js";
import { START, END, GraphInterrupt } from "./types.js";
import { lastValue } from "./channel.js";

// ═══════════════════════════════════════════════════════════════════════════════
// StateGraph Builder
// ═══════════════════════════════════════════════════════════════════════════════

export class StateGraph<State extends Record<string, unknown>> {
  private channels: { [K in keyof State]: Channel<State[K]> };
  private nodes = new Map<string, NodeFn<State>>();
  private edges: StaticEdge[] = [];
  private conditionalEdges: ConditionalEdge<State>[] = [];
  private entryPoint?: string;
  private finishPoints = new Set<string>();
  private nodeBeforeHooks = new Map<string, ((state: State) => Promise<void> | void)[]>();
  private nodeAfterHooks = new Map<string, ((state: State, output: Partial<State>) => Promise<void> | void)[]>();

  constructor(channels: Record<string, Channel<any>>) {
    this.channels = channels as { [K in keyof State]: Channel<State[K]> };
  }

  /** Add a node to the graph. */
  addNode(name: string, fn: NodeFn<State>): this {
    if (name === START || name === END) {
      throw new Error(`Node name cannot be "${START}" or "${END}"`);
    }
    this.nodes.set(name, fn);
    return this;
  }

  /** Add a static edge from source to target. */
  addEdge(source: string, target: string): this {
    this.edges.push({ source, target });
    return this;
  }

  /** Add conditional edges from source — router returns target node name(s). */
  addConditionalEdges(
    source: string,
    router: (state: State) => string | string[],
  ): this {
    this.conditionalEdges.push({ source, router });
    return this;
  }

  /** Set the entry point node (where execution begins). */
  setEntryPoint(name: string): this {
    this.entryPoint = name;
    return this;
  }

  /** Mark a node as a finish point (graph ends when reaching here). */
  setFinishPoint(name: string): this {
    this.finishPoints.add(name);
    return this;
  }

  /** Add a hook that runs before a specific node. */
  beforeNode(name: string, hook: (state: State) => Promise<void> | void): this {
    const list = this.nodeBeforeHooks.get(name) ?? [];
    list.push(hook);
    this.nodeBeforeHooks.set(name, list);
    return this;
  }

  /** Add a hook that runs after a specific node. */
  afterNode(name: string, hook: (state: State, output: Partial<State>) => Promise<void> | void): this {
    const list = this.nodeAfterHooks.get(name) ?? [];
    list.push(hook);
    this.nodeAfterHooks.set(name, list);
    return this;
  }

  /** Validate and compile the graph into an executable runtime. */
  compile(options: CompileOptions = {}): CompiledGraph<State> {
    this.validate();
    return new CompiledGraph(this, options);
  }

  private validate(): void {
    if (!this.entryPoint) {
      throw new Error("StateGraph: entry point not set. Call setEntryPoint().");
    }
    if (!this.nodes.has(this.entryPoint)) {
      throw new Error(`StateGraph: entry point "${this.entryPoint}" is not a registered node.`);
    }
    for (const edge of this.edges) {
      if (edge.source !== START && !this.nodes.has(edge.source)) {
        throw new Error(`StateGraph: edge source "${edge.source}" is not a registered node.`);
      }
      if (edge.target !== END && !this.nodes.has(edge.target)) {
        throw new Error(`StateGraph: edge target "${edge.target}" is not a registered node.`);
      }
    }
    for (const ce of this.conditionalEdges) {
      if (!this.nodes.has(ce.source)) {
        throw new Error(`StateGraph: conditional edge source "${ce.source}" is not a registered node.`);
      }
    }
  }

  /** @internal */
  _getChannels(): { [K in keyof State]: Channel<State[K]> } {
    return this.channels;
  }

  /** @internal */
  _getNodes(): Map<string, NodeFn<State>> {
    return this.nodes;
  }

  /** @internal */
  _getEdges(): StaticEdge[] {
    return this.edges;
  }

  /** @internal */
  _getConditionalEdges(): ConditionalEdge<State>[] {
    return this.conditionalEdges;
  }

  /** @internal */
  _getEntryPoint(): string | undefined {
    return this.entryPoint;
  }

  /** @internal */
  _getFinishPoints(): Set<string> {
    return this.finishPoints;
  }

  /** @internal */
  _getBeforeHooks(): Map<string, ((state: State) => Promise<void> | void)[]> {
    return this.nodeBeforeHooks;
  }

  /** @internal */
  _getAfterHooks(): Map<string, ((state: State, output: Partial<State>) => Promise<void> | void)[]> {
    return this.nodeAfterHooks;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CompiledGraph — Executable Runtime
// ═══════════════════════════════════════════════════════════════════════════════

export class CompiledGraph<State extends Record<string, unknown>> {
  private graph: StateGraph<State>;
  private options: Required<Omit<CompileOptions, "checkpointer">> & { checkpointer?: GraphCheckpointer };
  private channels: { [K in keyof State]: Channel<State[K]> };

  constructor(graph: StateGraph<State>, options: CompileOptions) {
    this.graph = graph;
    this.options = {
      checkpointer: options.checkpointer as GraphCheckpointer | undefined,
      recursionLimit: options.recursionLimit ?? 25,
      debug: options.debug ?? false,
      detectCycles: options.detectCycles ?? true,
      nodeTimeoutMs: options.nodeTimeoutMs ?? 60_000,
    };
    this.channels = graph._getChannels();

    if (this.options.detectCycles) {
      const cycle = this.detectCycles();
      if (cycle) {
        throw new Error(
          `StateGraph: cycle detected at compile time: ${cycle.join(" → ")}. ` +
            `Break the cycle with conditional edges to END or set detectCycles: false to bypass.`,
        );
      }
    }
  }

  /** Execute the graph with input values, return final state. */
  async invoke(input: Partial<State>, config: RunConfig = {}): Promise<State> {
    let finalState: State | undefined;
    for await (const event of this.stream(input, config)) {
      if (event.type === "end") {
        finalState = event.state as State;
      }
    }
    if (!finalState) {
      throw new Error("Graph completed without an end event.");
    }
    return finalState;
  }

  /** Stream graph execution events. */
  async *stream(input: Partial<State>, config: RunConfig = {}): AsyncGenerator<GraphEvent> {
    const threadId = config.threadId ?? `thread_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const metadata = config.metadata ?? {};
    const signal = config.signal;
    const streamMode = config.streamMode ?? "values";
    const nodeTimeoutMs = config.nodeTimeoutMs ?? this.options.nodeTimeoutMs;
    const graphTimeoutMs = config.graphTimeoutMs ?? 300_000;
    const detectStateLoops = config.detectStateLoops ?? true;
    const graphStartMs = Date.now();

    function graphTimedOut(): boolean {
      return Date.now() - graphStartMs > graphTimeoutMs;
    }

    // Initialize state from channels
    let state = this.buildInitialState(input);
    const completedNodes = new Set<string>();
    let waveCount = 0;

    // State-loop detection: fingerprint → waveCount
    const stateFingerprints = new Map<string, number>();

    yield { type: "start", threadId, input: input as Record<string, unknown> };

    // Try to resume from checkpoint
    const cp = await this.loadCheckpoint(threadId);
    if (cp) {
      state = { ...state, ...cp.state } as State;
      for (const n of cp.completedNodes) completedNodes.add(n);
      waveCount = cp.waveCount;
      yield { type: "state", threadId, values: { ...state } };
    }

    // Determine initial runnable nodes
    let runnable = this.getInitialRunnable(completedNodes);

    try {
      while (runnable.length > 0) {
        if (signal?.aborted) {
          throw new Error("Graph execution aborted.");
        }

        if (graphTimedOut()) {
          yield { type: "timeout", threadId, kind: "graph", limitMs: graphTimeoutMs };
          throw new Error(`Graph exceeded total timeout of ${graphTimeoutMs}ms.`);
        }

        if (waveCount >= this.options.recursionLimit) {
          throw new Error(`Graph exceeded recursion limit of ${this.options.recursionLimit} waves.`);
        }

        waveCount++;

        // ── State-loop detection ────────────────────────────────────────────
        if (detectStateLoops) {
          const fp = this.fingerprintState(state, completedNodes);
          if (stateFingerprints.has(fp)) {
            const prevWave = stateFingerprints.get(fp)!;
            yield {
              type: "cycle_detected",
              threadId,
              message: `Exact state repetition detected between wave ${prevWave} and wave ${waveCount}. ` +
                `This indicates an infinite loop.`,
            };
            throw new Error(
              `Infinite loop detected: state at wave ${waveCount} is identical to wave ${prevWave}. ` +
                `Use detectStateLoops: false to disable, or add a terminating condition.`,
            );
          }
          stateFingerprints.set(fp, waveCount);
        }

        // Yield node_start for each node in the wave
        for (const node of runnable) {
          yield { type: "node_start", node, threadId, state: { ...state } };
        }

        // Execute all runnable nodes in parallel with per-node timeout
        const nodeResults = await Promise.all(
          runnable.map(async (node) => {
            const nodeStartMs = Date.now();

            // Build the node execution promise
            const execPromise = (async () => {
              // Before hooks
              const beforeHooks = this.graph._getBeforeHooks().get(node);
              if (beforeHooks) {
                for (const hook of beforeHooks) await hook(state);
              }

              const fn = this.graph._getNodes().get(node)!;
              let output: Partial<State>;
              try {
                const result = fn(state, { threadId, metadata, signal });
                output = await Promise.resolve(result);
              } catch (error) {
                if (error instanceof GraphInterrupt) {
                  throw error;
                }
                const msg = error instanceof Error ? error.message : String(error);
                return { node, output: undefined as unknown as Partial<State>, durationMs: Date.now() - nodeStartMs, error: msg };
              }

              // After hooks
              const afterHooks = this.graph._getAfterHooks().get(node);
              if (afterHooks) {
                for (const hook of afterHooks) await hook(state, output);
              }

              const durationMs = Date.now() - nodeStartMs;
              return { node, output, durationMs };
            })();

            // Race against per-node timeout
            const timeoutPromise = new Promise<never>((_, reject) => {
              const timer = setTimeout(() => {
                reject(new Error(`Node "${node}" exceeded timeout of ${nodeTimeoutMs}ms.`));
              }, nodeTimeoutMs);
              // Clean up timer if signal fires first
              signal?.addEventListener("abort", () => {
                clearTimeout(timer);
                reject(new Error("Graph execution aborted."));
              }, { once: true });
            });

            try {
              return await Promise.race([execPromise, timeoutPromise]);
            } catch (error) {
              if (error instanceof GraphInterrupt) throw error;
              const msg = error instanceof Error ? error.message : String(error);
              return { node, output: undefined as unknown as Partial<State>, durationMs: Date.now() - nodeStartMs, error: msg };
            }
          }),
        );

        // Yield node_end and error events after parallel execution
        for (const result of nodeResults) {
          if ("error" in result) {
            yield { type: "error", threadId, node: result.node, error: result.error! };
            throw new Error(result.error);
          }
          yield { type: "node_end", node: result.node, threadId, output: result.output as Record<string, unknown>, durationMs: result.durationMs };
        }

        // Process outputs: extract Send objects and merge regular outputs
        const sends: Send[] = [];
        for (const { node, output } of nodeResults) {
          const { sends: nodeSends, cleanOutput } = this.extractSends(output);
          sends.push(...nodeSends);
          state = this.mergeOutputs(state, cleanOutput);
          completedNodes.add(node);

          // Check if this is a finish point
          if (this.graph._getFinishPoints().has(node)) {
            yield { type: "end", threadId, state: { ...state } };
            await this.saveCheckpoint(threadId, state, completedNodes, [], waveCount, metadata);
            return;
          }
        }

        if (streamMode === "values") {
          yield { type: "state", threadId, values: { ...state } };
        }

        // Compute next runnable from edges + Send objects
        const edgeRunnable = this.getNextRunnable(state, completedNodes, runnable);
        const sendRunnable = sends.map((s) => s.node);

        // Merge Send args into state
        for (const send of sends) {
          state = this.mergeOutputs(state, send.arg as Partial<State>);
        }

        runnable = [...edgeRunnable, ...sendRunnable];
        await this.saveCheckpoint(threadId, state, completedNodes, runnable, waveCount, metadata);
      }

      // ── Deadlock detection ────────────────────────────────────────────────
      // If no nodes are runnable but we haven't hit an end event, it's a deadlock
      // (unless there were no finish points defined — then it's a natural termination)
      if (this.graph._getFinishPoints().size > 0 && !Array.from(this.graph._getFinishPoints()).some((n) => completedNodes.has(n))) {
        yield {
          type: "deadlock",
          threadId,
          completedNodes: Array.from(completedNodes),
        };
        throw new Error(
          `Graph deadlock: no runnable nodes remain, but no finish point was reached. ` +
            `Completed nodes: ${Array.from(completedNodes).join(", ")}. ` +
            `Ensure all paths lead to a finish point or END.`,
        );
      }

      // No more runnable nodes — graph is complete
      yield { type: "end", threadId, state: { ...state } };
    } catch (error) {
      if (error instanceof GraphInterrupt) {
        yield { type: "interrupt", node: error.node, threadId, message: error.message, state: error.stateSnapshot };
        await this.saveCheckpoint(threadId, state, completedNodes, [error.node], waveCount, metadata);
        return;
      }
      throw error;
    }
  }

  /** Resume a graph after an interrupt. */
  async *resume(threadId: string, command: Command<State>, config: RunConfig = {}): AsyncGenerator<GraphEvent> {
    const cp = await this.loadCheckpoint(threadId);
    if (!cp) {
      throw new Error(`No checkpoint found for thread "${threadId}".`);
    }

    let state = { ...cp.state, ...command.update } as State;
    const completedNodes = new Set(cp.completedNodes);
    const waveCount = cp.waveCount;
    const nodeTimeoutMs = config.nodeTimeoutMs ?? this.options.nodeTimeoutMs;
    const graphTimeoutMs = config.graphTimeoutMs ?? 300_000;
    const graphStartMs = Date.now();

    function graphTimedOut(): boolean {
      return Date.now() - graphStartMs > graphTimeoutMs;
    }

    // Resume from the interrupted node or specified node
    const resumeNode = command.resumeNode ?? cp.nextNodes[0];
    if (!resumeNode) {
      throw new Error("No node to resume from.");
    }

    yield { type: "start", threadId, input: {} };
    yield { type: "state", threadId, values: { ...state } };

    // Run the resumed node
    const fn = this.graph._getNodes().get(resumeNode);
    if (!fn) {
      throw new Error(`Resume node "${resumeNode}" not found.`);
    }

    const startMs = Date.now();
    yield { type: "node_start", node: resumeNode, threadId, state: { ...state } };

    const nodePromise = Promise.resolve(fn(state, { threadId, metadata: command.metadata, signal: config.signal }));
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`Node "${resumeNode}" exceeded timeout of ${nodeTimeoutMs}ms.`)), nodeTimeoutMs);
    });

    let result: Partial<State>;
    try {
      result = await Promise.race([nodePromise, timeoutPromise]);
    } catch (error) {
      if (error instanceof GraphInterrupt) throw error;
      const msg = error instanceof Error ? error.message : String(error);
      yield { type: "error", threadId, node: resumeNode, error: msg };
      throw new Error(msg);
    }

    state = this.mergeOutputs(state, result);
    completedNodes.add(resumeNode);

    yield { type: "node_end", node: resumeNode, threadId, output: result as Record<string, unknown>, durationMs: Date.now() - startMs };

    if (this.graph._getFinishPoints().has(resumeNode)) {
      yield { type: "state", threadId, values: { ...state } };
      yield { type: "end", threadId, state: { ...state } };
      await this.saveCheckpoint(threadId, state, completedNodes, [], waveCount, command.metadata);
      return;
    }

    yield { type: "state", threadId, values: { ...state } };

    // Continue with remaining waves
    let runnable = this.getNextRunnable(state, completedNodes, [resumeNode]);
    while (runnable.length > 0) {
      if (config.signal?.aborted) throw new Error("Graph execution aborted.");
      if (graphTimedOut()) {
        yield { type: "timeout", threadId, kind: "graph", limitMs: graphTimeoutMs };
        throw new Error(`Graph exceeded total timeout of ${graphTimeoutMs}ms.`);
      }

      for (const node of runnable) {
        yield { type: "node_start", node, threadId, state: { ...state } };
      }

      const nodeResults = await Promise.all(
        runnable.map(async (node) => {
          const nodeStartMs = Date.now();
          const fn = this.graph._getNodes().get(node)!;
          const execPromise = Promise.resolve(fn(state, { threadId, metadata: command.metadata, signal: config.signal }));
          const timeoutPromise = new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error(`Node "${node}" exceeded timeout of ${nodeTimeoutMs}ms.`)), nodeTimeoutMs);
          });
          try {
            const output = await Promise.race([execPromise, timeoutPromise]);
            return { node, output, durationMs: Date.now() - nodeStartMs };
          } catch (error) {
            if (error instanceof GraphInterrupt) throw error;
            const msg = error instanceof Error ? error.message : String(error);
            return { node, output: undefined as unknown as Partial<State>, durationMs: Date.now() - nodeStartMs, error: msg };
          }
        }),
      );

      for (const result of nodeResults) {
        if ("error" in result) {
          yield { type: "error", threadId, node: result.node, error: result.error! };
          throw new Error(result.error);
        }
        yield { type: "node_end", node: result.node, threadId, output: result.output as Record<string, unknown>, durationMs: result.durationMs };
      }

      for (const { node, output } of nodeResults) {
        state = this.mergeOutputs(state, output);
        completedNodes.add(node);
        if (this.graph._getFinishPoints().has(node)) {
          yield { type: "end", threadId, state: { ...state } };
          await this.saveCheckpoint(threadId, state, completedNodes, [], waveCount, command.metadata);
          return;
        }
      }

      yield { type: "state", threadId, values: { ...state } };
      const nextRunnable = this.getNextRunnable(state, completedNodes, runnable);
      await this.saveCheckpoint(threadId, state, completedNodes, nextRunnable, waveCount, command.metadata);
      runnable = nextRunnable;
    }

    yield { type: "end", threadId, state: { ...state } };
  }

  /** Get the current state for a thread. */
  async getState(threadId: string): Promise<State | null> {
    const cp = await this.loadCheckpoint(threadId);
    return cp ? (cp.state as State) : null;
  }

  /**
   * Wrap this compiled graph as a node function for embedding in a parent graph.
   *
   * The subgraph receives the parent state (optionally mapped via `inputMapper`),
   * runs to completion, and its final state is merged back into the parent state
   * (optionally mapped via `outputMapper`).
   */
  asNode<ParentState extends Record<string, unknown>>(
    inputMapper?: (parentState: ParentState) => Partial<State>,
    outputMapper?: (subgraphState: State) => Partial<ParentState>,
  ): (parentState: ParentState, nodeConfig?: NodeConfig) => Promise<Partial<ParentState>> {
    return async (parentState, nodeConfig) => {
      const input = inputMapper ? inputMapper(parentState) : (parentState as unknown as Partial<State>);
      const result = await this.invoke(input, { threadId: nodeConfig?.threadId, signal: nodeConfig?.signal });
      return outputMapper ? outputMapper(result) : (result as unknown as Partial<ParentState>);
    };
  }

  /** Update state values for a thread (does not run nodes). */
  async updateState(threadId: string, values: Partial<State>): Promise<void> {
    const cp = await this.loadCheckpoint(threadId);
    if (!cp) {
      throw new Error(`No checkpoint found for thread "${threadId}".`);
    }
    const newState = this.mergeOutputs(cp.state as State, values);
    await this.saveCheckpoint(
      threadId,
      newState,
      new Set(cp.completedNodes),
      cp.nextNodes,
      cp.waveCount,
      cp.metadata,
    );
  }

  // ── Cycle Detection (compile-time) ──────────────────────────────────────────

  private detectCycles(): string[] | null {
    const adj = new Map<string, string[]>();
    const nodes = Array.from(this.graph._getNodes().keys());
    for (const n of nodes) adj.set(n, []);

    // Static edges
    for (const edge of this.graph._getEdges()) {
      if (edge.source !== START && edge.target !== END) {
        adj.get(edge.source)?.push(edge.target);
      }
    }
    // Conditional edges: we conservatively assume all possible targets
    for (const ce of this.graph._getConditionalEdges()) {
      // We can't know runtime targets, so we skip cycle detection for
      // conditional edges unless they unconditionally return to themselves
      // For safety, we only flag cycles through static edges.
    }

    // DFS cycle detection
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map<string, number>();
    const parent = new Map<string, string | null>();
    for (const n of nodes) color.set(n, WHITE);

    const dfs = (u: string): string[] | null => {
      color.set(u, GRAY);
      for (const v of adj.get(u) ?? []) {
        if (color.get(v) === GRAY) {
          // Found cycle — reconstruct path
          const cycle: string[] = [v];
          let cur = u;
          while (cur !== v) {
            cycle.push(cur);
            cur = parent.get(cur) ?? v;
          }
          cycle.push(v);
          return cycle.reverse();
        }
        if (color.get(v) === WHITE) {
          parent.set(v, u);
          const result = dfs(v);
          if (result) return result;
        }
      }
      color.set(u, BLACK);
      return null;
    };

    for (const n of nodes) {
      if (color.get(n) === WHITE) {
        parent.set(n, null);
        const result = dfs(n);
        if (result) return result;
      }
    }
    return null;
  }

  // ── State fingerprint for loop detection ────────────────────────────────────

  private fingerprintState(state: State, completedNodes: Set<string>): string {
    // Deterministic JSON hash of state + completed nodes
    const sortedCompleted = Array.from(completedNodes).sort().join(",");
    const sortedState = Object.keys(state)
      .sort()
      .map((k) => {
        const v = (state as Record<string, unknown>)[k];
        let serialized: string;
        if (v instanceof Set) {
          serialized = JSON.stringify([...v].sort());
        } else {
          serialized = JSON.stringify(v);
        }
        return `${k}=${serialized}`;
      })
      .join("|");
    return `${sortedState}@[${sortedCompleted}]`;
  }

  // ── Send extraction ─────────────────────────────────────────────────────────

  private extractSends(output: Partial<State> | undefined): { sends: Send[]; cleanOutput: Partial<State> } {
    if (!output) return { sends: [], cleanOutput: {} };

    // Check if output itself is a Send
    if (this.isSend(output)) {
      return { sends: [output], cleanOutput: {} };
    }

    // Check if any value in output is a Send or array of Sends
    const sends: Send[] = [];
    const cleanOutput = { ...output } as Record<string, unknown>;

    for (const [key, value] of Object.entries(cleanOutput)) {
      if (this.isSend(value)) {
        sends.push(value);
        delete cleanOutput[key];
      } else if (Array.isArray(value) && value.length > 0 && value.every((v) => this.isSend(v))) {
        sends.push(...value);
        delete cleanOutput[key];
      }
    }

    return { sends, cleanOutput: cleanOutput as Partial<State> };
  }

  private isSend(value: unknown): value is Send {
    return (
      value !== null &&
      typeof value === "object" &&
      "node" in value &&
      typeof (value as Record<string, unknown>).node === "string" &&
      "arg" in value
    );
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  private buildInitialState(input: Partial<State>): State {
    const state = {} as Record<string, unknown>;
    for (const [key, channel] of Object.entries(this.channels)) {
      state[key] = channel.default();
    }
    // Merge input values
    for (const [key, value] of Object.entries(input)) {
      if (key in this.channels) {
        const ch = this.channels[key as keyof State];
        state[key] = ch.update(state[key] as State[keyof State], value as State[keyof State]);
      } else {
        state[key] = value;
      }
    }
    return state as State;
  }

  private mergeOutputs(state: State, outputs: Partial<State>): State {
    const merged = { ...state } as Record<string, unknown>;
    for (const [key, value] of Object.entries(outputs)) {
      if (key in this.channels && value !== undefined) {
        const ch = this.channels[key as keyof State];
        merged[key] = ch.update(merged[key] as State[keyof State], value as State[keyof State]);
      } else if (value !== undefined) {
        merged[key] = value;
      }
    }
    return merged as State;
  }

  private getInitialRunnable(completed: Set<string>): string[] {
    const entry = this.graph._getEntryPoint()!;
    // Find nodes directly connected from START
    const fromStart = this.graph
      ._getEdges()
      .filter((e) => e.source === START)
      .map((e) => e.target);
    if (fromStart.length > 0) {
      return fromStart.filter((n) => !completed.has(n));
    }
    // Fallback: entry point itself
    return completed.has(entry) ? [] : [entry];
  }

  private getNextRunnable(state: State, completed: Set<string>, justRan: string[]): string[] {
    const next = new Set<string>();

    for (const node of justRan) {
      // Static edges
      for (const edge of this.graph._getEdges()) {
        if (edge.source === node && edge.target !== END) {
          next.add(edge.target);
        }
      }
      // Conditional edges
      for (const ce of this.graph._getConditionalEdges()) {
        if (ce.source === node) {
          const targets = ce.router(state);
          const arr = Array.isArray(targets) ? targets : [targets];
          for (const t of arr) {
            if (t !== END) next.add(t);
          }
        }
      }
    }

    // Allow nodes to be re-queued (cycles are handled by recursionLimit).
    // Only deduplicate within the same wave.
    return Array.from(next);
  }

  private async saveCheckpoint(
    threadId: string,
    state: State,
    completedNodes: Set<string>,
    nextNodes: string[],
    waveCount: number,
    metadata?: Record<string, unknown>,
  ): Promise<void> {
    if (!this.options.checkpointer) return;

    const list = await this.listCheckpoints(threadId);
    const parentId = list.length > 0 ? list[list.length - 1].checkpointId : undefined;

    const cp: GraphCheckpoint = {
      threadId,
      checkpointId: `${threadId}_cp_${waveCount}_${Date.now()}`,
      parentCheckpointId: parentId,
      state: { ...state },
      completedNodes: Array.from(completedNodes),
      nextNodes,
      waveCount,
      createdAt: Date.now(),
      metadata,
    };

    if ("saveGraphCheckpoint" in this.options.checkpointer) {
      await (this.options.checkpointer as GraphCheckpointer).saveGraphCheckpoint(cp);
    }
  }

  private async loadCheckpoint(threadId: string): Promise<GraphCheckpoint | null> {
    if (!this.options.checkpointer) return null;
    if ("loadGraphCheckpoint" in this.options.checkpointer) {
      return (this.options.checkpointer as GraphCheckpointer).loadGraphCheckpoint(threadId);
    }
    return null;
  }

  private async listCheckpoints(threadId: string): Promise<GraphCheckpoint[]> {
    if (!this.options.checkpointer) return [];
    if ("listGraphCheckpoints" in this.options.checkpointer) {
      return (this.options.checkpointer as GraphCheckpointer).listGraphCheckpoints(threadId);
    }
    return [];
  }
}
