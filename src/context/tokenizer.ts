import { get_encoding, Tiktoken } from "tiktoken";

let encoder: Tiktoken | null = null;

function getEncoder(): Tiktoken {
  if (!encoder) {
    encoder = get_encoding("cl100k_base");
  }
  return encoder;
}

export function countTokens(text: string): number {
  const enc = getEncoder();
  return enc.encode(text).length;
}

export function countMessagesTokens(
  messages: ReadonlyArray<{ role: string; content: string; reasoning_content?: string }>,
  systemPrompt: string
): number {
  const serialized = [
    systemPrompt,
    ...messages.map((msg) => {
      let text = `${msg.role}: ${msg.content}`;
      if (msg.reasoning_content) {
        text += `\n[reasoning]: ${msg.reasoning_content}`;
      }
      return text;
    }),
  ].join("\n");
  return getEncoder().encode(serialized).length;
}

export function tokensToDisplay(tokens: number): string {
  if (tokens < 1000) return `${tokens}`;
  if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${(tokens / 1000000).toFixed(1)}M`;
}
