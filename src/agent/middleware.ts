import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { shouldCompact, compactMessages } from "../context/compaction.js";
import { countMessagesTokens } from "../context/tokenizer.js";
import { getContextWindow } from "../llm/contextWindows.js";
import type { ChatMessage } from "../llm/provider.js";
import type { AgentTool } from "./types.js";
import { SkillMiddleware } from "../skills/middleware.js";

function chorusHome(): string {
  return process.env.CHORUS_HOME_DIR ?? path.join(os.homedir(), ".chorus");
}

export interface RoundContext {
  round: number;
  threadId: string;
  model: string;
  history: readonly ChatMessage[];
  toolCallsThisRound: number;
}

export interface CompactResult {
  replacement: ChatMessage[];
  removedMessages: number;
  savedTokens: number;
}

export interface ToolResultContext {
  id: string;
  name: string;
  result: string;
  durationMs: number;
}

export interface BeforeToolContext {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface AgentMiddleware {
  /** Execution priority for parallel hooks. Lower numbers run first. Default: 0. */
  priority?: number;
  beforeRound?(ctx: RoundContext): Promise<void>;
  /**
   * Called before a tool executes. Return `{ cancel: true, result }` to skip
   * execution and inject a synthetic result into the conversation.
   */
  beforeTool?(ctx: BeforeToolContext): Promise<void | { cancel: true; result: string }>;
  afterTool?(ctx: ToolResultContext): Promise<string | undefined>;
  maybeCompact?(history: readonly ChatMessage[], opts: { model: string; systemPrompt: string }): Promise<CompactResult | null>;
  afterRound?(ctx: RoundContext): Promise<void>;
  extraTools?(): AgentTool[];
  extraSystemPrompt?(): string;
  setTools?(toolsByName: Map<string, AgentTool>): void;
}

// ─── SummarizationMiddleware ──────────────────────────────────────────────────

const SUMMARIZATION_THRESHOLD = 0.85;

export interface SummarizationOptions {
  /** Fraction of the context window at which compaction kicks in (default: 0.85). */
  threshold?: number;
}

export class SummarizationMiddleware implements AgentMiddleware {
  private readonly threshold: number;

  constructor(private readonly threadId: string, opts?: SummarizationOptions) {
    this.threshold = opts?.threshold ?? SUMMARIZATION_THRESHOLD;
  }

  async maybeCompact(
    history: readonly ChatMessage[],
    opts: { model: string; systemPrompt: string },
  ): Promise<CompactResult | null> {
    const contextWindow = getContextWindow(opts.model);
    const threshold = Math.floor(contextWindow * this.threshold);
    const needsCompact = await shouldCompact(history, opts.systemPrompt, threshold);
    if (!needsCompact) return null;

    const originalCount = countMessagesTokens(history, opts.systemPrompt);
    this.saveHistorySnapshot(history);

    const result = await compactMessages(history, opts.systemPrompt, threshold);
    const replacement = result.messages as ChatMessage[];

    return {
      replacement,
      removedMessages: history.length - replacement.length,
      savedTokens: result.originalCount - result.compressedCount,
    };
  }

  private saveHistorySnapshot(history: readonly ChatMessage[]): void {
    try {
      const dir = path.join(chorusHome(), "history");
      fs.mkdirSync(dir, { recursive: true });
      const filePath = path.join(dir, `${this.threadId}.md`);
      const lines = history.map((m) => {
        const header = `### ${m.role}`;
        const body = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
        return `${header}\n\n${body}`;
      });
      fs.writeFileSync(filePath, lines.join("\n\n---\n\n"), "utf-8");
    } catch {
      // never block the loop on a snapshot write failure
    }
  }
}

// ─── ObservabilityMiddleware ──────────────────────────────────────────────────

interface ObsRecord {
  ts: number;
  round: number;
  type: string;
  [key: string]: unknown;
}

export class ObservabilityMiddleware implements AgentMiddleware {
  private readonly runId: string;
  private readonly logPath: string;
  private buffer: ObsRecord[] = [];

  constructor(threadId: string) {
    this.runId = `${threadId}-${Date.now()}`;
    const dir = path.join(chorusHome(), "runs");
    try {
      fs.mkdirSync(dir, { recursive: true });
    } catch {
      // ignore
    }
    this.logPath = path.join(dir, `${this.runId}.jsonl`);
  }

  async beforeRound(ctx: RoundContext): Promise<void> {
    this.buffer.push({ ts: Date.now(), round: ctx.round, type: "round-start" });
  }

  async afterTool(ctx: ToolResultContext): Promise<undefined> {
    this.buffer.push({
      ts: Date.now(),
      round: 0,
      type: "tool-done",
      id: ctx.id,
      name: ctx.name,
      durationMs: ctx.durationMs,
      resultLength: ctx.result.length,
    });
    return undefined;
  }

  async afterRound(ctx: RoundContext): Promise<void> {
    this.buffer.push({
      ts: Date.now(),
      round: ctx.round,
      type: "round-end",
      toolCallsThisRound: ctx.toolCallsThisRound,
    });
    this.flush();
  }

  private flush(): void {
    if (this.buffer.length === 0) return;
    const lines = this.buffer.map((r) => JSON.stringify(r)).join("\n") + "\n";
    this.buffer = [];
    try {
      fs.appendFileSync(this.logPath, lines, "utf-8");
    } catch {
      // never crash on log write failure
    }
  }
}

// ─── LargeOutputOffloadMiddleware ─────────────────────────────────────────────

const OFFLOAD_THRESHOLD_BYTES = 8 * 1024; // 8 KB

export interface LargeOutputOffloadOptions {
  /** Byte threshold at which tool output is offloaded to disk (default: 8192). */
  thresholdBytes?: number;
}

export class LargeOutputOffloadMiddleware implements AgentMiddleware {
  private readonly thresholdBytes: number;

  constructor(opts?: LargeOutputOffloadOptions) {
    this.thresholdBytes = opts?.thresholdBytes ?? OFFLOAD_THRESHOLD_BYTES;
  }

  async afterTool(ctx: ToolResultContext): Promise<string | undefined> {
    const byteLength = Buffer.byteLength(ctx.result, "utf-8");
    if (byteLength <= this.thresholdBytes) return undefined;

    try {
      const dir = path.join(chorusHome(), "tool-outputs");
      fs.mkdirSync(dir, { recursive: true });
      const filePath = path.join(dir, `${ctx.id}.txt`);
      fs.writeFileSync(filePath, ctx.result, "utf-8");
      const kb = (byteLength / 1024).toFixed(1);
      return `[Large output offloaded to ${filePath} (${kb} KB)]\n\n${ctx.result.slice(0, 500)}…`;
    } catch {
      return undefined;
    }
  }
}

// ─── TodoMiddleware ───────────────────────────────────────────────────────────

export class TodoMiddleware implements AgentMiddleware {
  private readonly filePath: string;

  constructor() {
    this.filePath = path.join(chorusHome(), "todo.md");
  }

  extraSystemPrompt(): string {
    return `You have access to a persistent todo list stored at ${this.filePath}. Use todo_read and todo_write to manage tasks across sessions.`;
  }

  extraTools(): AgentTool[] {
    return [
      {
        name: "todo_read",
        description: "Read the current todo list",
        async invoke() {
          try {
            return fs.readFileSync(path.join(chorusHome(), "todo.md"), "utf-8") || "(empty)";
          } catch {
            return "(empty)";
          }
        },
      },
      {
        name: "todo_write",
        description: "Overwrite the todo list with new content",
        async invoke(input: unknown) {
          const { content } = input as { content: string };
          const fp = path.join(chorusHome(), "todo.md");
          const tmp = `${fp}.tmp`;
          fs.writeFileSync(tmp, content ?? "", "utf-8");
          fs.renameSync(tmp, fp);
          return "ok";
        },
      },
    ];
  }
}

// ─── Default middleware factory ───────────────────────────────────────────────

export function createDefaultMiddleware(
  threadId: string,
  opts: { contextWindow?: number; skillDirs?: string[] } = {},
): AgentMiddleware[] {
  const middleware: AgentMiddleware[] = [
    new SummarizationMiddleware(threadId),
    new ObservabilityMiddleware(threadId),
    new LargeOutputOffloadMiddleware(),
  ];

  if (opts.contextWindow && opts.contextWindow > 0) {
    middleware.push(new SkillMiddleware(opts.contextWindow, opts.skillDirs));
  }

  return middleware;
}
