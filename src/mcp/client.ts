import { createHash } from "crypto";
import { spawn } from "child_process";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import type { Transport, FetchLike } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { AgentTool } from "../agent/types.js";
import { loadMcpServers, type McpServerConfig } from "./config.js";
import { buildAuthProvider, getAuthStatus, PersistentClientCredentialsProvider } from "./auth.js";

type McpToolDef = Awaited<ReturnType<Client["listTools"]>>["tools"][number];
type McpResourceDef = Awaited<ReturnType<Client["listResources"]>>["resources"][number];
type McpPromptDef = Awaited<ReturnType<Client["listPrompts"]>>["prompts"][number];

type ConnectionState = "disconnected" | "connecting" | "connected" | "error" | "reconnecting";

type ManagedConnection = {
  config: McpServerConfig;
  key: string;
  client: Client;
  transport: Transport;
  connectedAt: number;
  lastHealthCheck: number;
  tools: McpToolDef[];
  resources: McpResourceDef[];
  prompts: McpPromptDef[];
  state: ConnectionState;
  error?: string;
  consecutiveFailures: number;
};

export type McpTool = AgentTool & {
  mcpServerName: string;
  mcpToolName?: string;
  mcpReadOnly?: boolean;
};

export type McpServerStatus = {
  name: string;
  source: McpServerConfig["source"];
  command: string;
  connected: boolean;
  state: ConnectionState;
  toolCount: number;
  resourceCount: number;
  promptCount: number;
  error?: string;
  authType: string;
  needsAuth: boolean;
};

const connections = new Map<string, ManagedConnection>();
const DEFAULT_MAX_OUTPUT_TOKENS = 25_000;
const MAX_CONSECUTIVE_FAILURES = 3;
const HEALTH_CHECK_INTERVAL_MS = 30_000;
let healthCheckTimer: ReturnType<typeof setInterval> | undefined;

function connectionKey(config: McpServerConfig): string {
  return createHash("sha256")
    .update(JSON.stringify({
      name: config.name,
      command: config.command,
      args: config.args ?? [],
      env: config.env ?? {},
      cwd: config.cwd ?? "",
      type: config.type ?? "stdio",
      url: config.url ?? "",
      headers: config.headers ?? {},
      bearerTokenEnv: config.bearerTokenEnv ?? "",
      auth: config.auth ?? {},
      headersHelper: config.headersHelper ?? "",
      envFile: config.envFile ?? "",
    }))
    .digest("hex");
}

function sanitizeToolPart(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 48);
}

function namespacedToolName(serverName: string, toolName: string): string {
  return `mcp__${sanitizeToolPart(serverName)}__${sanitizeToolPart(toolName)}`;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function connectServer(config: McpServerConfig): Promise<ManagedConnection> {
  const key = connectionKey(config);
  const existing = connections.get(config.name);

  if (existing?.key === key && existing.state === "connected") {
    return existing;
  }

  if (existing) {
    if (existing.key !== key) {
      await closeConnection(existing);
      connections.delete(config.name);
    } else if (existing.state === "connecting") {
      return existing;
    }
  }

  const managed: ManagedConnection = {
    config,
    key,
    client: new Client(
      { name: "chorus-cli", version: "0.1.7" },
      { capabilities: { roots: { listChanged: true } } },
    ),
    transport: undefined as unknown as Transport,
    connectedAt: 0,
    lastHealthCheck: 0,
    tools: [],
    resources: [],
    prompts: [],
    state: "connecting",
    consecutiveFailures: 0,
  };

  try {
    managed.transport = await createTransport(config);
    const timeoutMs = config.timeoutMs ?? Number(process.env.MCP_TIMEOUT ?? 10_000);

    await withTimeout(managed.client.connect(managed.transport), timeoutMs, `MCP server "${config.name}" startup`);

    const [toolResult, resourceResult, promptResult] = await Promise.allSettled([
      withTimeout(managed.client.listTools(), timeoutMs, `MCP server "${config.name}" tools/list`),
      withTimeout(managed.client.listResources(), timeoutMs, `MCP server "${config.name}" resources/list`),
      withTimeout(managed.client.listPrompts(), timeoutMs, `MCP server "${config.name}" prompts/list`),
    ]);

    managed.tools = toolResult.status === "fulfilled" ? toolResult.value.tools : [];
    managed.resources = resourceResult.status === "fulfilled" ? resourceResult.value.resources : [];
    managed.prompts = promptResult.status === "fulfilled" ? promptResult.value.prompts : [];
    managed.connectedAt = Date.now();
    managed.lastHealthCheck = Date.now();
    managed.state = "connected";
    managed.consecutiveFailures = 0;
    managed.error = toolResult.status === "rejected" ? String(toolResult.reason) : undefined;

    connections.set(config.name, managed);
    startHealthChecks();
    return managed;
  } catch (error) {
    managed.state = "error";
    managed.consecutiveFailures = (existing?.consecutiveFailures ?? 0) + 1;
    managed.error = error instanceof Error ? error.message : String(error);
    managed.connectedAt = Date.now();
    connections.set(config.name, managed);
    return managed;
  }
}

async function reconnectServer(connection: ManagedConnection): Promise<void> {
  if (connection.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
    connection.state = "error";
    connection.error = `Max reconnection attempts (${MAX_CONSECUTIVE_FAILURES}) exceeded`;
    return;
  }

  connection.state = "reconnecting";
  try {
    await connection.client.close();
  } catch { /* ignore */ }

  try {
    connection.transport = await createTransport(connection.config);
    const timeoutMs = connection.config.timeoutMs ?? Number(process.env.MCP_TIMEOUT ?? 10_000);
    await withTimeout(connection.client.connect(connection.transport), timeoutMs, `MCP server "${connection.config.name}" reconnect`);
    await withTimeout(connection.client.listTools(), timeoutMs, `MCP server "${connection.config.name}" tools refresh`);

    connection.state = "connected";
    connection.consecutiveFailures = 0;
    connection.connectedAt = Date.now();
    connection.lastHealthCheck = Date.now();
    connection.error = undefined;
  } catch (error) {
    connection.state = "error";
    connection.consecutiveFailures += 1;
    connection.error = error instanceof Error ? error.message : String(error);
  }
}

async function healthCheck(connection: ManagedConnection): Promise<void> {
  if (connection.state !== "connected" && connection.state !== "error") return;
  connection.lastHealthCheck = Date.now();

  try {
    const timeoutMs = Math.min(connection.config.timeoutMs ?? 10_000, 5_000);
    await withTimeout(connection.client.ping({}), timeoutMs, `MCP server "${connection.config.name}" health check`);
    if (connection.state !== "connected") {
      connection.state = "connected";
      connection.consecutiveFailures = 0;
      connection.error = undefined;
    }
  } catch {
    if (connection.state === "connected") {
      connection.state = "error";
      connection.consecutiveFailures += 1;
      connection.error = "Health check failed - server may be unreachable";
    }
    if (connection.consecutiveFailures < MAX_CONSECUTIVE_FAILURES) {
      await reconnectServer(connection);
    }
  }
}

function startHealthChecks(): void {
  if (healthCheckTimer) return;
  healthCheckTimer = setInterval(() => {
    for (const connection of connections.values()) {
      healthCheck(connection);
    }
  }, HEALTH_CHECK_INTERVAL_MS);
  healthCheckTimer.unref();
}

function stopHealthChecks(): void {
  if (healthCheckTimer) {
    clearInterval(healthCheckTimer);
    healthCheckTimer = undefined;
  }
}

async function closeConnection(connection: ManagedConnection): Promise<void> {
  try {
    await connection.client.close();
  } catch {
    try {
      await connection.transport.close();
    } catch {
      // ignore shutdown failures
    }
  }
}
function parseHelperHeaders(output: string, serverName: string): Record<string, string> {
  const parsed = JSON.parse(output) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`MCP headers helper for "${serverName}" must print a JSON object`);
  }

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(parsed)) {
    if (typeof value !== "string") {
      throw new Error(`MCP headers helper for "${serverName}" returned non-string header "${key}"`);
    }
    headers[key] = value;
  }
  return headers;
}

async function runHeadersHelper(config: McpServerConfig): Promise<Record<string, string>> {
  const helper = config.headersHelper;
  if (!helper) return {};
  const command = typeof helper === "string" ? helper : helper.command;
  const args = typeof helper === "string" ? [] : helper.args ?? [];
  const timeoutMs = typeof helper === "string" ? config.timeoutMs ?? 10_000 : helper.timeoutMs ?? config.timeoutMs ?? 10_000;

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: typeof helper === "string" ? config.cwd : helper.cwd ?? config.cwd,
      env: {
        ...process.env,
        ...(typeof helper === "string" ? {} : helper.env ?? {}),
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`MCP headers helper for "${config.name}" timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => { stdout += String(chunk); });
    child.stderr.on("data", (chunk) => { stderr += String(chunk); });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`MCP headers helper for "${config.name}" exited ${code}: ${stderr.trim()}`));
        return;
      }
      try {
        resolve(parseHelperHeaders(stdout, config.name));
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function buildHeaders(config: McpServerConfig): Promise<Record<string, string>> {
  const headers: Record<string, string> = { ...(config.headers ?? {}) };
  if (config.bearerTokenEnv) {
    const token = process.env[config.bearerTokenEnv];
    if (token && !headers.Authorization && !headers.authorization) {
      headers.Authorization = `Bearer ${token}`;
    }
  }
  Object.assign(headers, await runHeadersHelper(config));
  return headers;
}

async function createTransport(config: McpServerConfig): Promise<Transport> {
  const type = config.type ?? "stdio";
  if (type === "stdio") {
    return new StdioClientTransport({
      command: config.command!,
      args: config.args ?? [],
      cwd: config.cwd,
      env: { ...getDefaultEnvironment(), ...(config.env ?? {}) },
      stderr: "pipe",
    });
  }

  const headers = await buildHeaders(config);
  const authProvider = buildAuthProvider(config);
  const requestInit: RequestInit = Object.keys(headers).length > 0 ? { headers } : {};

  if (type === "sse") {
    const fetchWithHeaders: FetchLike = (url, init) => fetch(url, {
      ...init,
      headers: { ...headers, ...(init?.headers as Record<string, string> | undefined ?? {}) },
    });
    return new SSEClientTransport(new URL(config.url!), {
      authProvider,
      requestInit,
      eventSourceInit: Object.keys(headers).length > 0 ? { fetch: fetchWithHeaders } : undefined,
      fetch: fetchWithHeaders,
    });
  }

  return new StreamableHTTPClientTransport(new URL(config.url!), { authProvider, requestInit });
}

function outputTokenLimit(config: McpServerConfig): number {
  const envLimit = process.env.CHORUS_MCP_MAX_OUTPUT_TOKENS ?? process.env.MAX_MCP_OUTPUT_TOKENS;
  const parsed = envLimit ? Number(envLimit) : undefined;
  return config.maxOutputTokens ?? (Number.isFinite(parsed) && parsed! > 0 ? parsed! : DEFAULT_MAX_OUTPUT_TOKENS);
}

function capMcpOutput(output: string, config: McpServerConfig): string {
  const maxTokens = outputTokenLimit(config);
  const maxChars = Math.max(1_000, maxTokens * 4);
  if (output.length <= maxChars) return output;
  return `${output.slice(0, maxChars)}\n\n[Chorus truncated MCP output at ${maxTokens.toLocaleString()} tokens. Set maxOutputTokens or CHORUS_MCP_MAX_OUTPUT_TOKENS to change this.]`;
}

function formatCallToolResult(result: Awaited<ReturnType<Client["callTool"]>>, config: McpServerConfig): string {
  if ("toolResult" in result) {
    const output = typeof result.toolResult === "string" ? result.toolResult : JSON.stringify(result.toolResult, null, 2);
    return capMcpOutput(output, config);
  }

  const sections: string[] = [];
  if (result.structuredContent) {
    sections.push(`Structured content:\n${JSON.stringify(result.structuredContent, null, 2)}`);
  }

  for (const item of result.content ?? []) {
    if (item.type === "text") {
      sections.push(item.text);
    } else if (item.type === "resource") {
      const resource = item.resource;
      sections.push("text" in resource
        ? `[Resource ${resource.uri}]\n${resource.text}`
        : `[Binary resource ${resource.uri} ${resource.mimeType ?? ""}]`);
    } else if (item.type === "resource_link") {
      sections.push(`[Resource link ${item.uri}] ${item.name}${item.description ? ` - ${item.description}` : ""}`);
    } else if (item.type === "image" || item.type === "audio") {
      sections.push(`[${item.type} content ${item.mimeType}, ${item.data.length} base64 chars]`);
    }
  }

  const output = sections.join("\n\n").trim() || "(empty MCP tool result)";
  return capMcpOutput(result.isError ? `MCP tool returned an error:\n${output}` : output, config);
}

function createToolWrapper(connection: ManagedConnection, tool: McpToolDef): McpTool {
  return {
    name: namespacedToolName(connection.config.name, tool.name),
    description: `[MCP:${connection.config.name}] ${tool.description ?? tool.title ?? tool.name}`,
    schema: tool.inputSchema,
    mcpServerName: connection.config.name,
    mcpToolName: tool.name,
    mcpReadOnly: tool.annotations?.readOnlyHint === true && tool.annotations?.destructiveHint !== true,
    async invoke(input: unknown) {
      const result = await connection.client.callTool({
        name: tool.name,
        arguments: input && typeof input === "object" ? input as Record<string, unknown> : {},
      });
      return formatCallToolResult(result, connection.config);
    },
  };
}

function createPromptTools(connection: ManagedConnection): McpTool[] {
  const server = connection.config.name;
  return [
    {
      name: namespacedToolName(server, "list_prompts"),
      description: `[MCP:${server}] List prompt templates exposed by this MCP server.`,
      schema: { type: "object", properties: {}, additionalProperties: false },
      mcpServerName: server,
      mcpReadOnly: true,
      async invoke() {
        const result = await connection.client.listPrompts();
        return capMcpOutput(
          result.prompts
            .map((p) => `${p.name}  ${p.description ?? ""}`)
            .join("\n") || "(no MCP prompts)",
          connection.config,
        );
      },
    },
    {
      name: namespacedToolName(server, "get_prompt"),
      description: `[MCP:${server}] Get a prompt template by name from this MCP server.`,
      schema: {
        type: "object",
        properties: {
          name: { type: "string", description: "Prompt name to retrieve." },
          arguments: {
            type: "object",
            description: "Optional key-value arguments to substitute into the prompt template.",
            additionalProperties: true,
          },
        },
        required: ["name"],
        additionalProperties: false,
      },
      mcpServerName: server,
      mcpReadOnly: true,
      async invoke(input: unknown) {
        const params = input as { name?: string; arguments?: Record<string, string> };
        if (!params.name) throw new Error("Missing required MCP prompt name.");
        const result = await connection.client.getPrompt({ name: params.name, arguments: params.arguments });
        const messages = result.messages
          .map((m) => {
            if (m.content.type === "text") return m.content.text;
            if (m.content.type === "image") return `[Image ${m.content.mimeType}]`;
            if (m.content.type === "resource") {
              const r = m.content.resource;
              return "text" in r ? `[Resource ${r.uri}]\n${r.text}` : `[Resource ${r.uri}]`;
            }
            return "";
          })
          .filter(Boolean)
          .join("\n\n");
        return capMcpOutput(messages, connection.config);
      },
    },
  ];
}

function createResourceTools(connection: ManagedConnection): McpTool[] {
  const server = connection.config.name;
  return [
    {
      name: namespacedToolName(server, "list_resources"),
      description: `[MCP:${server}] List resources exposed by this MCP server.`,
      schema: { type: "object", properties: {}, additionalProperties: false },
      mcpServerName: server,
      mcpReadOnly: true,
      async invoke() {
        const result = await connection.client.listResources();
        return capMcpOutput(result.resources
          .map((r) => `${r.uri}  ${r.name}${r.description ? ` - ${r.description}` : ""}`)
          .join("\n") || "(no MCP resources)", connection.config);
      },
    },
    {
      name: namespacedToolName(server, "read_resource"),
      description: `[MCP:${server}] Read a resource by URI from this MCP server.`,
      schema: {
        type: "object",
        properties: { uri: { type: "string", description: "Resource URI to read." } },
        required: ["uri"],
        additionalProperties: false,
      },
      mcpServerName: server,
      mcpReadOnly: true,
      async invoke(input: unknown) {
        const uri = (input as { uri?: string } | undefined)?.uri;
        if (!uri) throw new Error("Missing required MCP resource URI.");
        const result = await connection.client.readResource({ uri });
        return capMcpOutput(result.contents.map((content) => (
          "text" in content
            ? `[${content.uri}]\n${content.text}`
            : `[${content.uri}] binary ${content.mimeType ?? "application/octet-stream"} ${content.blob.length} base64 chars`
        )).join("\n\n"), connection.config);
      },
    },
  ];
}

export async function getMcpTools(): Promise<McpTool[]> {
  const configs = loadMcpServers();
  if (configs.length === 0) return [];

  const connections = await Promise.all(configs.map(connectServer));
  return connections.flatMap((connection) => {
    if (connection.error) return [];
    return [
      ...connection.tools.map((tool) => createToolWrapper(connection, tool)),
      ...createResourceTools(connection),
      ...createPromptTools(connection),
    ];
  });
}

export async function getMcpStatus(): Promise<McpServerStatus[]> {
  const configs = loadMcpServers();
  await Promise.allSettled(configs.map((c) => connectServer(c)));
  return configs.map((config) => {
    const connection = connections.get(config.name);
    const auth = getAuthStatus(config);
    return {
      name: config.name,
      source: config.source,
      command: formatServerEndpoint(config),
      connected: connection?.state === "connected",
      state: connection?.state ?? "disconnected",
      toolCount: connection?.tools.length ?? 0,
      resourceCount: connection?.resources.length ?? 0,
      promptCount: connection?.prompts.length ?? 0,
      error: connection?.error,
      authType: auth.type,
      needsAuth: auth.needsAuth,
    };
  });
}

function formatServerEndpoint(config: McpServerConfig): string {
  if ((config.type ?? "stdio") === "stdio") {
    return [config.command, ...(config.args ?? [])].filter(Boolean).join(" ");
  }
  return `${config.type}: ${config.url}`;
}

export async function reloadMcpConnections(): Promise<McpServerStatus[]> {
  const current = [...connections.values()];
  connections.clear();
  stopHealthChecks();
  await Promise.all(current.map(closeConnection));
  return getMcpStatus();
}

export async function closeMcpConnections(): Promise<void> {
  const current = [...connections.values()];
  connections.clear();
  stopHealthChecks();
  await Promise.all(current.map(closeConnection));
}
