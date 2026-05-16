/**
 * State Channels — Reducer Semantics for Typed State Fields
 *
 * Each channel defines how a state field is initialized and how updates are merged.
 * This enables parallel node execution: multiple nodes can write to the same field,
 * and the channel's reducer deterministically merges the results.
 *
 * Channel types:
 *   - LastValue: last write wins (default for most fields)
 *   - Append: accumulate values in an array
 *   - BinaryOperator: custom merge function (e.g., sum, max, set union)
 */

import type { Channel } from "./types.js";

/** Last write wins. Use for scalar fields that nodes overwrite. */
export function lastValue<T>(): Channel<T | undefined> {
  return {
    default: () => undefined,
    update: (_current, update) => update,
  };
}

/** Append updates to an array. Use for message lists, event logs. */
export function append<T>(): Channel<T[]> {
  return {
    default: () => [],
    update: (current, update) => {
      if (Array.isArray(update)) {
        return [...current, ...update];
      }
      return [...current, update as T];
    },
  };
}

/** Prepend updates to an array (newest first). */
export function prepend<T>(): Channel<T[]> {
  return {
    default: () => [],
    update: (current, update) => {
      if (Array.isArray(update)) {
        return [...update, ...current];
      }
      return [update as T, ...current];
    },
  };
}

/** Custom binary operator merge. Use for counters, sets, maps. */
export function binaryOperator<T>(
  op: (a: T, b: T) => T,
  defaultValue: T,
): Channel<T> {
  return {
    default: () => defaultValue,
    update: (current, update) => op(current, update),
  };
}

/** Sum channel for numeric counters. */
export function sum(defaultValue = 0): Channel<number> {
  return binaryOperator((a, b) => a + b, defaultValue);
}

/** Set union channel. */
export function setUnion<T>(): Channel<Set<T>> {
  return {
    default: () => new Set(),
    update: (current, update) => {
      const next = new Set(current);
      if (update instanceof Set) {
        for (const item of update) next.add(item);
      } else {
        next.add(update as T);
      }
      return next;
    },
  };
}

/** Map merge channel — shallow merge of Record objects. */
export function mapMerge<T>(): Channel<Record<string, T>> {
  return {
    default: () => ({}),
    update: (current, update) => ({ ...current, ...update }),
  };
}

/** Create a channel with a literal default value (last-write-wins). */
export function withDefault<T>(defaultValue: T): Channel<T> {
  return {
    default: () => defaultValue,
    update: (_current, update) => update,
  };
}
