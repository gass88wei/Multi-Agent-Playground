from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph

from ..runtime import llm_gateway
from ..schemas import (
    AgentDefinition,
    RunArtifacts,
    TraceEvent,
    WorkflowDefinition,
    WorkflowGraph,
    WorkflowRunResponse,
)
from ..store import InMemoryPlaygroundStore
from .langgraph_adapter import workflow_graph_from_compiled


AGENT_NODE = "single_agent"
FINALIZER_NODE = "finalize"


class SingleAgentState(TypedDict, total=False):
    user_input: str
    specialist_answer: str
    assistant_message: str


def event(
    event_type: str,
    title: str,
    detail: str,
    **payload: object,
) -> TraceEvent:
    return TraceEvent(type=event_type, title=title, detail=detail, payload=payload)


def _compile_single_agent_app(
    workflow: WorkflowDefinition,
    agent: AgentDefinition,
    agent_node: Callable[[SingleAgentState], SingleAgentState],
    finalizer_node: Callable[[SingleAgentState], SingleAgentState] | None = None,
):
    builder = StateGraph(SingleAgentState)
    builder.add_node(
        AGENT_NODE,
        agent_node,
        metadata={"kind": "agent", "label": agent.name},
    )
    if workflow.finalizer_enabled and finalizer_node is not None:
        builder.add_node(
            FINALIZER_NODE,
            finalizer_node,
            metadata={"kind": "final", "label": "Finalizer"},
        )

    builder.add_edge(START, AGENT_NODE)
    if workflow.finalizer_enabled and finalizer_node is not None:
        builder.add_edge(AGENT_NODE, FINALIZER_NODE)
        builder.add_edge(FINALIZER_NODE, END)
    else:
        builder.add_edge(AGENT_NODE, END)

    return builder.compile()


def build_single_agent_graph(
    workflow: WorkflowDefinition,
    agents: list[AgentDefinition],
) -> WorkflowGraph:
    if not agents:
        raise HTTPException(status_code=400, detail="single_agent_chat requires at least 1 agent.")
    agent = agents[0]

    def noop_agent(_: SingleAgentState) -> SingleAgentState:
        return {}

    def noop_finalizer(_: SingleAgentState) -> SingleAgentState:
        return {}

    app = _compile_single_agent_app(
        workflow,
        agent,
        agent_node=noop_agent,
        finalizer_node=noop_finalizer if workflow.finalizer_enabled else None,
    )
    return workflow_graph_from_compiled(app)


def run_single_agent_chat(
    store: InMemoryPlaygroundStore,
    workflow: WorkflowDefinition,
    user_input: str,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> WorkflowRunResponse:
    agent: AgentDefinition | None = None
    for agent_id in workflow.specialist_agent_ids:
        resolved = store.get_agent(agent_id)
        if resolved is not None:
            agent = resolved
            break
    if agent is None:
        raise HTTPException(status_code=400, detail="single_agent_chat requires 1 valid agent.")

    def push(trace: list[TraceEvent], item: TraceEvent) -> None:
        trace.append(item)
        if on_event is not None:
            on_event(item)

    trace: list[TraceEvent] = []
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

    def agent_node(state: SingleAgentState) -> SingleAgentState:
        push(
            trace,
            event(
                "node_entered",
                "Enter Agent",
                f"{agent.name} is generating the response.",
                node_id=AGENT_NODE,
                agent_id=agent.id,
            ),
        )
        specialist_answer = llm_gateway.run_agent(agent, state["user_input"])
        push(
            trace,
            event(
                "message_generated",
                "Agent Output",
                f"{agent.name} generated an answer.",
                node_id=AGENT_NODE,
                agent_id=agent.id,
                preview=specialist_answer[:120],
            ),
        )
        push(
            trace,
            event(
                "node_exited",
                "Exit Agent",
                f"{agent.name} finished processing.",
                node_id=AGENT_NODE,
                agent_id=agent.id,
            ),
        )
        return {"specialist_answer": specialist_answer}

    def finalizer_node(state: SingleAgentState) -> SingleAgentState:
        push(
            trace,
            event(
                "node_entered",
                "Enter Finalizer",
                "Finalizer is composing the final answer.",
                node_id=FINALIZER_NODE,
            ),
        )
        assistant_message = llm_gateway.finalize(
            user_input=state["user_input"],
            agent=agent,
            specialist_answer=state["specialist_answer"],
        )
        push(
            trace,
            event(
                "node_exited",
                "Exit Finalizer",
                "Finalizer finished.",
                node_id=FINALIZER_NODE,
            ),
        )
        return {"assistant_message": assistant_message}

    app = _compile_single_agent_app(
        workflow,
        agent,
        agent_node=agent_node,
        finalizer_node=finalizer_node if workflow.finalizer_enabled else None,
    )
    graph = workflow_graph_from_compiled(app)
    final_state = app.invoke({"user_input": user_input})

    specialist_answer = str(final_state.get("specialist_answer", ""))
    if workflow.finalizer_enabled:
        assistant_message = str(final_state.get("assistant_message", specialist_answer))
    else:
        assistant_message = specialist_answer

    push(
        trace,
        event(
            "run_finished",
            "Run Finished",
            "Workflow completed.",
            workflow_id=workflow.id,
        ),
    )

    artifacts = RunArtifacts(
        route_agent_id=agent.id,
        route_agent_name=agent.name,
        route_reason="single_agent_chat",
        specialist_answer=specialist_answer or None,
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
