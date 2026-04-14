from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, TypedDict

from fastapi import HTTPException

from ..runtime import llm_gateway
from ..settings_bridge import settings
from ..schemas import (
    AgentDefinition,
    RunArtifacts,
    TraceEvent,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    WorkflowRunResponse,
)
from ..store import InMemoryPlaygroundStore


ROUTER_NODE = "first_owner_router"
GROUP_NODE = "peer_pool"
FINALIZER_NODE = "finalize"


class AgentAction(TypedDict, total=False):
    action: str
    target_agent_id: str
    task_title: str
    message: str
    raw_response: str


class PeerState(TypedDict, total=False):
    user_input: str
    current_task_title: str
    current_owner_id: str
    current_owner_name: str
    route_reason: str
    reports: list[str]
    hop_count: int
    max_hops: int
    assistant_message: str
    terminal_status: str


class ToolOutcome(TypedDict, total=False):
    blocked: bool
    failed: bool
    ok: bool
    message: str


PEER_HANDOFF_ACTION_EXAMPLES = (
    "## Examples\n"
    "### Handoff\n"
    '{"action":"handoff","target_agent_id":"agent_designer","task_title":"Create UI/UX design based on the completed PRD","message":"I completed the PRD with scope, core features, and acceptance criteria. Please produce the UI/UX design next."}\n\n'
    "### Complete\n"
    '{"action":"complete","message":"I updated the project files, added the missing interaction logic, and verified the calculator now responds correctly."}\n\n'
    "### Block\n"
    '{"action":"block","message":"The required API key is missing, so I cannot call the deployment tool. Please provide the key or let another peer handle a non-deployment path."}\n\n'
    "### Bad\n"
    '- {"action":"block","message":"TOOL_EXECUTION_NO_FINAL_ANSWER ..."}\n'
    '- {"action":"complete","message":"I found the cause. Next I will fix it."}\n'
)

INTERNAL_RUNTIME_MARKERS = (
    "TOOL_EXECUTION_NO_FINAL_ANSWER",
    "TOOL_EXECUTION_BLOCKED",
    "TOOL_UNAVAILABLE",
    "Tool-enabled execution completed",
    "This result should not be treated as task completion.",
    "This result should be retried, continued by the planner, or handed to another step.",
)

INTERNAL_RUNTIME_LINE_PREFIXES = (
    "Selected tools:",
    "Verified evidence:",
    "Tool:",
    "Skill:",
    "Attempts:",
    "Reason:",
    "Error code:",
)


def _build_peer_handoff_worker_prompt(
    *,
    user_input: str,
    current_task_title: str,
    hop_count: int,
    max_hops: int,
    peer_directory: str,
    reports_block: str,
) -> str:
    return (
        "# Peer Handoff\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not use markdown outside the JSON object.\n\n"
        "## Allowed actions\n"
        '- {"action":"handoff","target_agent_id":"<peer-id>","task_title":"<next task>","message":"<handoff reason>"}\n'
        '- {"action":"review","target_agent_id":"<peer-id>","task_title":"<review task>","message":"<review reason>"}\n'
        '- {"action":"complete","message":"<task result>"}\n'
        '- {"action":"respond_user","message":"<final user-facing answer>"}\n'
        '- {"action":"block","message":"<real blocker only>"}\n\n'
        "## Core rules\n"
        "- Use `complete` only for work already executed.\n"
        "- If future work remains, do not use `complete`.\n"
        "- If another peer is clearly better for the next step, use `handoff`.\n"
        "- If no clearly better peer exists, continue working instead of handing off.\n"
        "- Use `block` only for a real blocker.\n"
        "- Never target yourself.\n\n"
        "## Message rules\n"
        "- `message` must contain business context only.\n"
        "- Never copy runtime markers such as TOOL_EXECUTION_NO_FINAL_ANSWER or TOOL_EXECUTION_BLOCKED.\n"
        "- If using `handoff`, make `task_title` and `message` specific enough for the next peer to execute.\n"
        "- Mention concrete paths/files/outputs only when they matter.\n\n"
        f"{PEER_HANDOFF_ACTION_EXAMPLES}\n"
        "## Context\n"
        f"Original user request:\n{user_input}\n\n"
        f"Completed collaboration log:\n{reports_block}\n\n"
        f"Peer directory:\n{peer_directory}\n\n"
        f"Current task:\n{current_task_title}\n\n"
        f"Current hop budget: {hop_count}/{max_hops}"
    )


PEER_HANDOFF_FINAL_RESPONSE_INSTRUCTION = (
    "Return ONLY one JSON object that matches one allowed action exactly. "
    "Do not use markdown. Do not output prose outside JSON. "
    "Never copy internal runtime markers into message, including TOOL_EXECUTION_NO_FINAL_ANSWER, TOOL_EXECUTION_BLOCKED, or Tool-enabled execution completed. "
    "The message must contain business context, not runtime diagnostics. "
    "If tool execution did not yield enough useful information, choose handoff or review when another peer can continue. "
    'If no clearly better peer exists for the remaining work, continue executing instead of handing off. '
    'Use "complete" only for work that has already been executed. '
    'If the message implies future work, remaining implementation, or a next step for another peer, do not use "complete". '
    'If downstream work remains, prefer "handoff". '
    'Use "block" only for a real blocker, not for incomplete execution or runtime summary text. '
    'Allowed actions: {"action":"handoff","target_agent_id":"<peer-id>","task_title":"<next task>","message":"<handoff reason>"}, '
    '{"action":"review","target_agent_id":"<peer-id>","task_title":"<review task>","message":"<review reason>"}, '
    '{"action":"complete","message":"<task result>"}, '
    '{"action":"respond_user","message":"<final user-facing answer>"}, '
    '{"action":"block","message":"<real blocker only>"}.'
)


def event(
    event_type: str,
    title: str,
    detail: str,
    **payload: object,
) -> TraceEvent:
    return TraceEvent(type=event_type, title=title, detail=detail, payload=payload)


def _estimate_max_hops(agent_count: int, user_input: str) -> int:
    base = max(5, min(12, agent_count * 3 + 2))
    text = str(user_input or "").strip()
    if len(text) > 220:
        return min(14, base + 2)
    return base


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None

    fenced = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_action_name(raw: str) -> str:
    lowered = str(raw or "").strip().lower()
    if lowered in {"handoff", "delegate", "handoff_to"}:
        return "handoff"
    if lowered in {"review", "review_by", "request_review"}:
        return "review"
    if lowered in {"complete", "done", "finish"}:
        return "complete"
    if lowered in {"respond_user", "respond", "final", "answer_user"}:
        return "respond_user"
    if lowered in {"block", "blocked"}:
        return "block"
    return lowered


def _parse_agent_action(raw_response: str) -> AgentAction | None:
    payload = _extract_json_object(raw_response)
    if payload is None:
        return None

    action = _normalize_action_name(payload.get("action", ""))
    if action not in {"handoff", "review", "complete", "respond_user", "block"}:
        return None

    result: AgentAction = {
        "action": action,
        "raw_response": raw_response,
        "message": str(payload.get("message") or "").strip(),
    }
    target_agent_id = str(payload.get("target_agent_id") or "").strip()
    task_title = str(payload.get("task_title") or "").strip()
    if target_agent_id:
        result["target_agent_id"] = target_agent_id
    if task_title:
        result["task_title"] = task_title
    return result


def _contains_internal_runtime_text(text: str) -> bool:
    message = str(text or "").strip()
    if not message:
        return False
    return any(marker in message for marker in INTERNAL_RUNTIME_MARKERS)


def _sanitize_action_message(text: str) -> str:
    message = str(text or "").strip()
    if not message:
        return ""

    lines: list[str] = []
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(marker in line for marker in INTERNAL_RUNTIME_MARKERS):
            continue
        if any(line.startswith(prefix) for prefix in INTERNAL_RUNTIME_LINE_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _validate_agent_action(action: AgentAction) -> str | None:
    action_name = str(action.get("action") or "").strip()
    message = str(action.get("message") or "").strip()
    target_agent_id = str(action.get("target_agent_id") or "").strip()
    task_title = str(action.get("task_title") or "").strip()

    if action_name in {"handoff", "review"}:
        if not target_agent_id:
            return f"{action_name} requires target_agent_id"
        if not task_title:
            return f"{action_name} requires task_title"
        if not message:
            return f"{action_name} requires message"
    elif action_name in {"complete", "respond_user", "block"} and not message:
        return f"{action_name} requires message"

    if message and _contains_internal_runtime_text(message):
        return "message contains internal runtime text"
    return None


def _fallback_action(raw_response: str) -> AgentAction:
    text = _sanitize_action_message(raw_response)
    return {
        "action": "block",
        "message": text or "Agent returned an invalid workflow action payload.",
        "raw_response": raw_response,
    }


def _repair_agent_action(
    *,
    raw_response: str,
    worker: AgentDefinition,
    workers: list[AgentDefinition],
    user_input: str,
    current_task_title: str,
    reports: list[str],
    invalid_reason: str,
) -> AgentAction | None:
    if not llm_gateway.api_configured or llm_gateway.client is None:
        return None

    peer_lines = "\n".join(
        f"- id={peer.id}; name={peer.name}; description={peer.description}"
        for peer in workers
        if peer.id != worker.id
    ) or "(no peers)"
    completed_log = _reports_block(reports)

    prompt = (
        "You are a workflow action repair layer.\n"
        "Your only job is to convert the worker output into ONE valid JSON action.\n"
        "Do not do the task again. Do not add markdown. Do not output prose outside JSON.\n\n"
        "Allowed actions:\n"
        '- {"action":"handoff","target_agent_id":"<peer-id>","task_title":"<next task>","message":"<handoff reason>"}\n'
        '- {"action":"review","target_agent_id":"<peer-id>","task_title":"<review task>","message":"<review reason>"}\n'
        '- {"action":"complete","message":"<task result>"}\n'
        '- {"action":"respond_user","message":"<final user-facing answer>"}\n'
        '- {"action":"block","message":"<real blocker only>"}\n\n'
        "Repair rules:\n"
        "- Preserve the original meaning as much as possible.\n"
        "- Never copy internal runtime markers into message, including TOOL_EXECUTION_NO_FINAL_ANSWER or TOOL_EXECUTION_BLOCKED.\n"
        "- If the original output includes prose outside the JSON object, discard the extra prose and keep only one valid JSON object.\n"
        "- If a message contains internal runtime status text, strip it out and rewrite the action with clean business-facing wording.\n"
        "- If the output suggests another specialist should continue, prefer handoff.\n"
        "- If the output indicates a real blocker with no clear next peer, use block.\n"
        "- If the output contains a usable task result and no handoff is needed, use complete.\n"
        "- Never target the current worker.\n\n"
        f"Why the original output is invalid:\n{invalid_reason}\n\n"
        f"Current worker:\n- id={worker.id}; name={worker.name}; description={worker.description}\n\n"
        f"Available peers:\n{peer_lines}\n\n"
        f"Original user request:\n{user_input}\n\n"
        f"Current task title:\n{current_task_title}\n\n"
        f"Completed collaboration log:\n{completed_log}\n\n"
        "Raw worker output to repair:\n"
        f"{raw_response}"
    )

    try:
        response = llm_gateway.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:  # noqa: BLE001
        return None

    repaired = (response.choices[0].message.content or "").strip()
    action = _parse_agent_action(repaired)
    if action is None:
        return None
    validation_error = _validate_agent_action(action)
    if validation_error is not None:
        return None
    cleaned_message = _sanitize_action_message(str(action.get("message") or ""))
    if cleaned_message:
        action["message"] = cleaned_message
    action["raw_response"] = raw_response
    return action


def _peer_directory(workers: list[AgentDefinition], current_agent_id: str) -> str:
    lines: list[str] = []
    for worker in workers:
        role = "current_owner" if worker.id == current_agent_id else "peer"
        lines.append(
            f"- id={worker.id}; name={worker.name}; role={role}; description={worker.description}"
        )
    return "\n".join(lines)


def _reports_block(reports: list[str]) -> str:
    if not reports:
        return "(none yet)"
    return "\n\n".join(reports[-6:])


def build_peer_handoff_graph(
    workflow: WorkflowDefinition,
    agents: list[AgentDefinition],
) -> WorkflowGraph:
    if len(agents) < 2:
        raise HTTPException(status_code=400, detail="peer_handoff requires at least 2 agents.")

    nodes = [
        WorkflowNode(id="start", label="START", kind="start"),
        WorkflowNode(id=ROUTER_NODE, label="First Owner Router", kind="logic"),
        WorkflowNode(id=GROUP_NODE, label="Peer Collaboration Zone", kind="group"),
        *[
            WorkflowNode(id=agent.id, label=agent.name, kind="agent", parent_id=GROUP_NODE)
            for agent in agents
        ],
    ]
    if workflow.finalizer_enabled:
        nodes.append(WorkflowNode(id=FINALIZER_NODE, label="Finalizer", kind="final"))
    nodes.append(WorkflowNode(id="end", label="END", kind="end"))

    edges = [
        WorkflowEdge(source="start", target=ROUTER_NODE),
        WorkflowEdge(source=ROUTER_NODE, target=GROUP_NODE),
    ]
    if workflow.finalizer_enabled:
        edges.append(WorkflowEdge(source=GROUP_NODE, target=FINALIZER_NODE))
        edges.append(WorkflowEdge(source=FINALIZER_NODE, target="end"))
    else:
        edges.append(WorkflowEdge(source=GROUP_NODE, target="end"))
    return WorkflowGraph(nodes=nodes, edges=edges)


def run_peer_handoff(
    store: InMemoryPlaygroundStore,
    workflow: WorkflowDefinition,
    user_input: str,
    history: list[dict[str, str]] | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> WorkflowRunResponse:
    workers: list[AgentDefinition] = []
    for agent_id in workflow.specialist_agent_ids:
        agent = store.get_agent(agent_id)
        if agent is not None:
            workers.append(agent)

    if len(workers) < 2:
        raise HTTPException(status_code=400, detail="peer_handoff requires at least 2 valid agents.")

    worker_by_id = {worker.id: worker for worker in workers}

    def push(trace: list[TraceEvent], item: TraceEvent) -> None:
        trace.append(item)
        if on_event is not None:
            on_event(item)

    trace: list[TraceEvent] = []
    latest_tool_outcome: dict[str, ToolOutcome] = {}

    def make_tool_trace_hook(agent: AgentDefinition):
        def on_tool_trace(meta: dict[str, Any]) -> None:
            stage = str(meta.get("stage") or "")
            tool_name = str(meta.get("tool_name") or "tool")

            if stage == "tool_started":
                push(
                    trace,
                    event(
                        "state_updated",
                        "Tool Started",
                        f"{agent.name} is running {tool_name}.",
                        node_id=agent.id,
                        agent_id=agent.id,
                        tool_name=tool_name,
                        tool_call_id=meta.get("tool_call_id"),
                        input_keys=meta.get("input_keys", []),
                        skill_id=meta.get("skill_id"),
                        skill_name=meta.get("skill_name"),
                    ),
                )
                return

            if stage == "tool_retry":
                attempt = int(meta.get("attempt") or 1)
                max_attempts = int(meta.get("max_attempts") or attempt)
                reason = str(meta.get("reason") or "Transient failure, retrying.")
                push(
                    trace,
                    event(
                        "state_updated",
                        "Tool Retry",
                        f"{tool_name} attempt {attempt}/{max_attempts} failed: {reason[:120]}",
                        node_id=agent.id,
                        agent_id=agent.id,
                        tool_name=tool_name,
                        tool_call_id=meta.get("tool_call_id"),
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_ms=meta.get("delay_ms"),
                        skill_id=meta.get("skill_id"),
                        skill_name=meta.get("skill_name"),
                    ),
                )
                return

            if stage == "tool_blocked":
                reason = _sanitize_action_message(
                    str(meta.get("reason") or "Tool execution failed; continuing without this tool.")
                ) or "Tool execution failed; continuing without this tool."
                latest_tool_outcome[agent.id] = {
                    "blocked": True,
                    "failed": True,
                    "ok": False,
                    "message": reason,
                }
                push(
                    trace,
                    event(
                        "state_updated",
                        "Tool Unavailable",
                        reason[:220],
                        node_id=agent.id,
                        agent_id=agent.id,
                        tool_name=tool_name,
                        tool_call_id=meta.get("tool_call_id"),
                        skill_id=meta.get("skill_id"),
                        skill_name=meta.get("skill_name"),
                        missing_env_vars=meta.get("missing_env_vars", []),
                        missing_shell_dependencies=meta.get("missing_shell_dependencies", []),
                        missing_launchers=meta.get("missing_launchers", []),
                    ),
                )
                return

            if stage != "tool_finished":
                return

            ok = bool(meta.get("ok"))
            generated_files = meta.get("generated_files")
            files = generated_files if isinstance(generated_files, list) else []
            latest_tool_outcome[agent.id] = {
                "blocked": False,
                "failed": not ok,
                "ok": ok,
                "message": _sanitize_action_message(str(meta.get("error") or "").strip()),
            }
            detail = f"{agent.name} finished {tool_name} ({'success' if ok else 'failed'})."
            attempt_count = int(meta.get("attempt_count") or 1)
            max_attempts = int(meta.get("max_attempts") or 1)
            if max_attempts > 1:
                detail += f" Attempts {attempt_count}/{max_attempts}."
            if ok and files:
                detail += f" Generated {len(files)} file(s)."
            if (not ok) and meta.get("error"):
                detail += f" Error: {str(meta.get('error'))[:140]}"
            push(
                trace,
                event(
                    "state_updated",
                    "Tool Finished" if ok else "Tool Failed",
                    detail,
                    node_id=agent.id,
                    agent_id=agent.id,
                    tool_name=tool_name,
                    tool_call_id=meta.get("tool_call_id"),
                    ok=ok,
                    duration_ms=meta.get("duration_ms"),
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    output_dir=meta.get("output_dir"),
                    generated_files=files,
                ),
            )

            result_preview = str(meta.get("result_preview") or "").strip()
            if result_preview:
                push(
                    trace,
                    event(
                        "message_generated",
                        "Tool Output",
                        f"{tool_name} produced output.",
                        node_id=agent.id,
                        agent_id=agent.id,
                        tool_name=tool_name,
                        preview=result_preview[:180],
                    ),
                )

        return on_tool_trace

    push(
        trace,
        event(
            "run_started",
            "Run Started",
            f"Starting workflow: {workflow.name}",
            workflow_id=workflow.id,
            workflow_type=workflow.type,
        ),
    )

    graph = build_peer_handoff_graph(workflow, workers)

    max_hops = _estimate_max_hops(len(workers), user_input)
    state: PeerState = {
        "user_input": user_input,
        "current_task_title": user_input,
        "reports": [],
        "hop_count": 0,
        "max_hops": max_hops,
    }

    push(
        trace,
        event(
            "node_entered",
            "Enter First Owner Router",
            "Routing the request into the peer collaboration zone.",
            node_id=ROUTER_NODE,
        ),
    )
    routed_worker_id, route_reason = llm_gateway.route(user_input, workers)
    first_worker = worker_by_id[routed_worker_id]
    push(
        trace,
        event(
            "route_selected",
            "First Owner Selected",
            f"Router selected {first_worker.name} as the initial owner.",
            node_id=ROUTER_NODE,
            next_node_id=first_worker.id,
            reason=route_reason,
            focus_task=user_input,
        ),
    )
    push(
        trace,
        event(
            "node_exited",
            "Exit First Owner Router",
            "Initial owner routing completed.",
            node_id=ROUTER_NODE,
        ),
    )
    state["current_owner_id"] = first_worker.id
    state["current_owner_name"] = first_worker.name
    state["route_reason"] = route_reason

    last_worker = first_worker
    while True:
        worker = worker_by_id[str(state.get("current_owner_id") or first_worker.id)]
        last_worker = worker

        push(
            trace,
            event(
                "node_entered",
                "Enter Peer Agent",
                f"{worker.name} is deciding the next collaboration step.",
                node_id=worker.id,
                hop_count=state.get("hop_count", 0),
                task_title=state.get("current_task_title", user_input),
            ),
        )

        worker_input = _build_peer_handoff_worker_prompt(
            user_input=state["user_input"],
            current_task_title=str(state.get("current_task_title", state["user_input"])),
            hop_count=int(state.get("hop_count", 0)),
            max_hops=max_hops,
            peer_directory=_peer_directory(workers, worker.id),
            reports_block=_reports_block(list(state.get("reports", []))),
        )
        final_response_instruction = PEER_HANDOFF_FINAL_RESPONSE_INSTRUCTION
        raw_response = llm_gateway.run_agent(
            worker,
            worker_input,
            history=history,
            trace_hook=make_tool_trace_hook(worker),
            final_response_instruction=final_response_instruction,
            response_contract="action_json",
        )
        action = _parse_agent_action(raw_response)
        invalid_reason = _validate_agent_action(action) if action is not None else "output was not a single valid JSON action object"
        if invalid_reason is not None:
            repaired_action = _repair_agent_action(
                raw_response=raw_response,
                worker=worker,
                workers=workers,
                user_input=state["user_input"],
                current_task_title=str(state.get("current_task_title", state["user_input"])),
                reports=list(state.get("reports", [])),
                invalid_reason=invalid_reason,
            )
            if repaired_action is not None:
                action = repaired_action
                push(
                    trace,
                    event(
                        "state_updated",
                        "Action Repaired",
                        f"{worker.name}'s output was repaired into a valid workflow action.",
                        node_id=worker.id,
                        repaired_action=action.get("action"),
                    ),
                )
            else:
                action = _fallback_action(raw_response)
        action_name = str(action.get("action") or "complete")
        action_message = _sanitize_action_message(str(action.get("message") or "").strip())
        if not action_message:
            action_message = _sanitize_action_message(str(raw_response or "").strip())
        if not action_message:
            action_message = "Agent returned an invalid workflow action payload."
        action["message"] = action_message
        tool_outcome = latest_tool_outcome.get(worker.id, {})

        if tool_outcome.get("failed") and action_name in {"complete", "respond_user"}:
            failure_reason = _sanitize_action_message(str(tool_outcome.get("message") or "").strip())
            action_name = "block"
            action["action"] = "block"
            action_message = (
                f"Tool execution failed before completion. {failure_reason}".strip()
                if failure_reason
                else "Tool execution failed before completion."
            )
            action["message"] = action_message

        reports = list(state.get("reports", []))
        reports.append(
            f"{worker.name} [{action_name}] on '{state.get('current_task_title', user_input)}':\n{action_message}"
        )
        state["reports"] = reports

        push(
            trace,
            event(
                "message_generated",
                "Peer Action",
                f"{worker.name} proposed {action_name}.",
                node_id=worker.id,
                action=action_name,
                preview=action_message[:180],
                target_agent_id=action.get("target_agent_id"),
                task_title=action.get("task_title"),
            ),
        )
        push(
            trace,
            event(
                "node_exited",
                "Exit Peer Agent",
                f"{worker.name} returned a structured workflow action.",
                node_id=worker.id,
            ),
        )

        if action_name in {"handoff", "review"}:
            target_agent_id = str(action.get("target_agent_id") or "").strip()
            next_task_title = str(action.get("task_title") or "").strip() or state.get("current_task_title", user_input)
            target_worker = worker_by_id.get(target_agent_id)
            if target_worker is None or target_worker.id == worker.id:
                rewritten_task = next_task_title or state.get("current_task_title", user_input)
                reports = list(state.get("reports", []))
                reports.append(
                    f"Runtime note for {worker.name}:\n"
                    f"The proposed handoff target was invalid. Continue the remaining work yourself under this task:\n{rewritten_task}"
                )
                state["reports"] = reports
                push(
                    trace,
                    event(
                        "state_updated",
                        "Handoff Rewritten",
                        "Peer handoff target was invalid, so runtime kept the current agent and continued execution.",
                        node_id=worker.id,
                        target_agent_id=target_agent_id,
                        task_title=rewritten_task,
                    ),
                )
                state["current_task_title"] = rewritten_task
                state["hop_count"] = int(state.get("hop_count", 0))
                continue

            next_hop = int(state.get("hop_count", 0)) + 1
            if next_hop >= max_hops:
                push(
                    trace,
                    event(
                        "state_updated",
                        "Hop Limit Reached",
                        f"Reached max peer handoff budget ({max_hops}).",
                        node_id=worker.id,
                        hop_count=next_hop,
                        max_hops=max_hops,
                    ),
                )
                state["terminal_status"] = "max_hops"
                break

            push(
                trace,
                event(
                    "route_selected",
                    "Review Requested" if action_name == "review" else "Peer Handoff",
                    f"{worker.name} routed work to {target_worker.name}.",
                    node_id=worker.id,
                    next_node_id=target_worker.id,
                    reason=action_message[:180],
                    task_title=next_task_title,
                    hop_count=next_hop,
                ),
            )
            state["current_owner_id"] = target_worker.id
            state["current_owner_name"] = target_worker.name
            state["current_task_title"] = next_task_title
            state["hop_count"] = next_hop
            continue

        if action_name == "respond_user":
            state["assistant_message"] = action_message
            state["terminal_status"] = "respond_user"
            break

        if action_name == "block":
            state["terminal_status"] = "blocked"
            break

        state["terminal_status"] = "complete"
        break

    combined_report = "\n\n".join(list(state.get("reports", [])))
    assistant_message = str(state.get("assistant_message") or "").strip()

    if workflow.finalizer_enabled:
        push(
            trace,
            event(
                "node_entered",
                "Enter Finalizer",
                "Finalizer is composing the visible answer from peer collaboration reports.",
                node_id=FINALIZER_NODE,
            ),
        )
        specialist_answer = combined_report
        if assistant_message:
            specialist_answer = f"{combined_report}\n\nDirect user-ready answer:\n{assistant_message}".strip()
        assistant_message = llm_gateway.finalize(
            user_input=user_input,
            agent=last_worker,
            specialist_answer=specialist_answer or assistant_message or "No specialist output was produced.",
        )
        push(
            trace,
            event(
                "node_exited",
                "Exit Finalizer",
                "Finalizer completed.",
                node_id=FINALIZER_NODE,
            ),
        )
    elif not assistant_message:
        assistant_message = combined_report or "Workflow finished without a visible answer."

    push(
        trace,
        event(
            "run_finished",
            "Run Finished",
            "Workflow completed.",
            workflow_id=workflow.id,
            terminal_status=state.get("terminal_status", "complete"),
        ),
    )

    artifacts = RunArtifacts(
        route_agent_id=state.get("current_owner_id"),
        route_agent_name=state.get("current_owner_name"),
        route_reason=(
            f"First owner: {first_worker.name}. "
            f"Peer hops used: {state.get('hop_count', 0)}/{max_hops}. "
            f"Terminal status: {state.get('terminal_status', 'complete')}."
        ),
        specialist_answer=combined_report or None,
        final_answer=assistant_message,
    )
    return WorkflowRunResponse(
        workflow_id=workflow.id,
        user_input=user_input,
        assistant_message=assistant_message,
        trace=trace,
        graph=graph,
        artifacts=artifacts,
    )
