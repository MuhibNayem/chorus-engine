/**
 * MCP Server — exposes Chorus agents/swarm as a Model Context Protocol server.
 *
 * This allows Chorus to be plugged into IDEs (Cursor, Claude Desktop, etc.)
 * and other MCP clients. The server exposes:
 *   • Tools: All Chorus tools (filesystem, git, shell, web search, etc.)
 *   • Resources: Files in the workspace
 *   • Prompts: System prompts for each configured agent/subagent
 *
 * Usage:
 *   const server = new ChorusMcpServer({ port: 3001, tools, prompts });
 *   await server.start();
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import type { AgentTool } from "../agent/types.js";
import type { IncomingMessage, ServerResponse } from "http";
import * as http from "http";

export interface ChorusMcpServerConfig {
  /** Server name advertised to MCP clients */
  name?: string;
  /** Server version */
  version?: string;
  /** Transport type */
  transport?: "stdio" | "sse";
  /** Port for SSE transport (default: 3001) */
  port?: number;
  /** Host for SSE transport (default: 127.0.0.1) */
  host?: string;
  /** Tools to expose */
  tools?: AgentTool[];
  /** Prompts to expose: name → { description, template } */
  prompts?: Record<string, { description: string; template: string }>;
}

export class ChorusMcpServer {
  private server: Server;
  private transport: StdioServerTransport | SSEServerTransport | undefined;
  private httpServer: http.Server | undefined;
  private config: ChorusMcpServerConfig;

  constructor(config: ChorusMcpServerConfig = {}) {
    this.config = config;
    this.server = new Server(
      { name: config.name ?? "chorus-engine", version: config.version ?? "0.1.0" },
      {
        capabilities: {
          tools: {},
          resources: {},
          prompts: {},
        },
      },
    );
    this.registerHandlers();
  }

  private registerHandlers(): void {
    const tools = this.config.tools ?? [];
    const prompts = this.config.prompts ?? {};

    // ─── Tools ───────────────────────────────────────────────────────────────
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: tools
        .filter((t): t is AgentTool & { name: string } => typeof t.name === "string" && t.name.length > 0)
        .map((t) => ({
          name: t.name,
          description: t.description ?? "",
          inputSchema:
            t.schema && typeof t.schema === "object" && !Array.isArray(t.schema)
              ? (t.schema as Record<string, unknown>)
              : { type: "object" as const, properties: {} },
        })),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      const tool = tools.find((t) => t.name === name);
      if (!tool) {
        throw new Error(`Tool not found: ${name}`);
      }
      try {
        const result = await tool.invoke(args ?? {});
        return {
          content: [{ type: "text" as const, text: typeof result === "string" ? result : JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          content: [{ type: "text" as const, text: `Error: ${message}` }],
          isError: true,
        };
      }
    });

    // ─── Resources (workspace files) ─────────────────────────────────────────
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [], // Chorus does not expose a static resource list by default
    }));

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;
      throw new Error(`Resource not found: ${uri}`);
    });

    // ─── Prompts ─────────────────────────────────────────────────────────────
    this.server.setRequestHandler(ListPromptsRequestSchema, async () => ({
      prompts: Object.entries(prompts).map(([name, p]) => ({
        name,
        description: p.description,
      })),
    }));

    this.server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      const prompt = prompts[name];
      if (!prompt) {
        throw new Error(`Prompt not found: ${name}`);
      }
      let text = prompt.template;
      if (args) {
        for (const [key, value] of Object.entries(args)) {
          text = text.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), String(value));
        }
      }
      return {
        messages: [
          {
            role: "user" as const,
            content: { type: "text" as const, text },
          },
        ],
      };
    });
  }

  async start(): Promise<void> {
    const transportType = this.config.transport ?? "stdio";

    if (transportType === "stdio") {
      this.transport = new StdioServerTransport();
      await this.server.connect(this.transport);
      return;
    }

    // SSE transport
    const port = this.config.port ?? 3001;
    const host = this.config.host ?? "127.0.0.1";

    this.httpServer = http.createServer((req: IncomingMessage, res: ServerResponse) => {
      if (req.url === "/sse") {
        this.transport = new SSEServerTransport("/messages", res);
        void this.server.connect(this.transport);
        return;
      }
      if (req.url === "/messages" && req.method === "POST") {
        if (this.transport instanceof SSEServerTransport) {
          void this.transport.handlePostMessage(req, res);
        } else {
          res.writeHead(503);
          res.end("SSE transport not initialized");
        }
        return;
      }
      res.writeHead(404);
      res.end("Not found");
    });

    return new Promise((resolve) => {
      this.httpServer!.listen(port, host, () => resolve());
    });
  }

  async stop(): Promise<void> {
    await this.server.close();
    if (this.httpServer) {
      await new Promise<void>((resolve, reject) => {
        this.httpServer!.close((err) => (err ? reject(err) : resolve()));
      });
    }
  }
}
