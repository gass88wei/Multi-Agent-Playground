async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let detail = errorText;
    try {
      const parsed = JSON.parse(errorText);
      if (typeof parsed?.detail === "string" && parsed.detail.trim()) {
        detail = parsed.detail.trim();
      }
    } catch {
      // noop: keep raw error text
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export function fetchTemplates() {
  return request("/api/workflow-templates");
}

export function fetchSkills() {
  return request("/api/skills");
}

export function createSkill(payload) {
  return request("/api/skills", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function syncSkills(payload) {
  return request("/api/skills/sync", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function installSkill(skillId) {
  return request(`/api/skills/${skillId}/install`, {
    method: "POST",
  });
}

export function fetchAgents() {
  return request("/api/agents");
}

export function createAgent(payload) {
  return request("/api/agents", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateAgent(agentId, payload) {
  return request(`/api/agents/${agentId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function fetchWorkflows() {
  return request("/api/workflows");
}

export function createWorkflow(payload) {
  return request("/api/workflows", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateWorkflow(workflowId, payload) {
  return request(`/api/workflows/${workflowId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function fetchWorkflowGraph(workflowId) {
  return request(`/api/workflows/${workflowId}/graph`);
}

export function runWorkflow(payload) {
  return request("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

function parseSseFrame(frame) {
  const lines = frame.split(/\r?\n/);
  let eventName = "message";
  let dataText = "";
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataText += `${line.slice(5).trimStart()}\n`;
    }
  }
  if (!dataText) return null;
  const raw = dataText.trim();
  try {
    return { event: eventName, data: JSON.parse(raw) };
  } catch {
    return { event: eventName, data: raw };
  }
}

export async function runWorkflowStream(
  payload,
  { onTrace, onFinal, onError, onEnd, signal } = {},
) {
  const response = await fetch("/api/runs/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Streaming body is not available in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let splitIndex = buffer.indexOf("\n\n");
    while (splitIndex >= 0) {
      const frame = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      const parsed = parseSseFrame(frame);
      if (parsed) {
        if (parsed.event === "trace") onTrace?.(parsed.data);
        if (parsed.event === "final") onFinal?.(parsed.data);
        if (parsed.event === "error") onError?.(parsed.data);
      }
      splitIndex = buffer.indexOf("\n\n");
    }
  }

  onEnd?.();
}
