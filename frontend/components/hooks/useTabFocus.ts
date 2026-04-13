"use client";

import type React from "react";
import { useEffect } from "react";

import { apiUrl } from "@/lib/api";

export function useTabFocus(sessionId: string, wsRef: React.RefObject<WebSocket | null>) {
  useEffect(() => {
    const handler = () => {
      if (document.visibilityState !== "hidden") {
        return;
      }
      const timestamp = new Date().toISOString();
      wsRef.current?.send(
        JSON.stringify({
          event: "integrity_flag",
          session_id: sessionId,
          flag_type: "tab_hidden",
          timestamp,
        }),
      );
      fetch(apiUrl(`/sessions/${sessionId}/integrity_flag`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ flag_type: "tab_hidden", timestamp }),
      }).catch(() => undefined);
    };

    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, [sessionId, wsRef]);
}
