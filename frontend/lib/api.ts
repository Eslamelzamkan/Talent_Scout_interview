"use client";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";
const DEFAULT_WS_BASE = process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8001/api";

function normalizeBase(base: string): string {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

function readError(response: Response, fallback: string): Promise<string> {
  return response
    .text()
    .then((value) => value || fallback)
    .catch(() => fallback);
}

export function apiUrl(path: string): string {
  return `${normalizeBase(DEFAULT_API_BASE)}${normalizePath(path)}`;
}

export function wsUrl(path: string): string {
  return `${normalizeBase(DEFAULT_WS_BASE)}${normalizePath(path)}`;
}

export async function apiFetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = init?.body
    ? { "Content-Type": "application/json", ...(init.headers ?? {}) }
    : init?.headers;
  const response = await fetch(apiUrl(path), {
    cache: "no-store",
    ...init,
    headers,
  });
  if (!response.ok) {
    throw new Error(await readError(response, `Request failed with ${response.status}`));
  }
  return response.json() as Promise<T>;
}
