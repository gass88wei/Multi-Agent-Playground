from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .runtime import llm_gateway
from .schemas import (
    AgentDefinition,
    AgentDefinitionCreate,
    AgentDefinitionUpdate,
    SkillDefinition,
    SkillDefinitionCreate,
    SkillInstallResponse,
    SkillSyncRequest,
    SkillSyncResponse,
    TraceEvent,
    WorkflowDefinition,
    WorkflowDefinitionCreate,
    WorkflowDefinitionUpdate,
    WorkflowGraph,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from .skillhub_client import skillhub_client
from .store import store
from .workflows.planner_executor import build_planner_graph, run_planner_executor
from .workflows.router_specialists import build_router_graph, run_router_specialists
from .workflows.single_agent_chat import build_single_agent_graph, run_single_agent_chat
from .workflows.supervisor_dynamic import build_supervisor_graph, run_supervisor_dynamic


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/workflow-templates")
def list_workflow_templates():
    return store.get_templates()


@router.get("/skills", response_model=list[SkillDefinition])
def list_skills() -> list[SkillDefinition]:
    skills = store.list_skills()
    for skill in skills:
        skill.runtime_preflight = llm_gateway.build_skill_preflight(skill)
    return skills


@router.post("/skills", response_model=SkillDefinition)
def create_skill(payload: SkillDefinitionCreate) -> SkillDefinition:
    return store.create_skill(payload)


@router.post("/skills/{skill_id}/install", response_model=SkillInstallResponse)
def install_skill(skill_id: str) -> SkillInstallResponse:
    skill = store.get_skill(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found.")

    provider = str(skill.source_provider or "").strip().lower()
    source_skill_id = str(skill.source_skill_id or "").strip() or None

    if provider == "skillhub":
        if not source_skill_id:
            raise HTTPException(status_code=400, detail="SkillHub skill missing source_skill_id.")
        try:
            remote = skillhub_client.fetch_skill_package(source_skill_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        package_files = remote.package_files or {}
        if not package_files:
            raise HTTPException(
                status_code=409,
                detail=(
                    "SkillHub returned metadata only (no package files). "
                    "This skill cannot be truly installed yet."
                ),
            )

        installed = store.install_skill_package(
            skill_id=skill.id,
            name=remote.name or skill.name,
            description=remote.description or skill.description,
            instruction=remote.instruction or skill.instruction,
            tool=remote.tool,
            package_files=package_files,
        )
        if installed is None:
            raise HTTPException(status_code=404, detail="Skill not found.")

        return SkillInstallResponse(
            skill_id=installed.id,
            skill_name=installed.name,
            source_provider=installed.source_provider,
            source_skill_id=installed.source_skill_id,
            downloaded_files=len(package_files),
            tool_enabled=bool(installed.tool),
            message=f"Skill package downloaded: {len(package_files)} files.",
        )

    # Local/manual skill: package already exists in local store.
    return SkillInstallResponse(
        skill_id=skill.id,
        skill_name=skill.name,
        source_provider=skill.source_provider,
        source_skill_id=skill.source_skill_id,
        downloaded_files=0,
        tool_enabled=bool(skill.tool),
        message="Local skill is ready.",
    )


@router.post("/skills/sync", response_model=SkillSyncResponse)
def sync_skills(payload: SkillSyncRequest) -> SkillSyncResponse:
    if payload.provider != "skillhub":
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")

    query = (payload.query or "").strip() or "search"

    try:
        remote_skills = skillhub_client.fetch_skills(query=query, limit=payload.limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    imported, updated = store.upsert_marketplace_skills(
        source_provider="skillhub",
        skills=[
            {
                "source_skill_id": skill.source_skill_id,
                "name": skill.name,
                "description": skill.description,
                "instruction": skill.instruction,
                "tool": skill.tool,
                "package_files": skill.package_files,
            }
            for skill in remote_skills
        ],
    )

    return SkillSyncResponse(
        provider="skillhub",
        query=query,
        fetched=len(remote_skills),
        imported=imported,
        updated=updated,
    )


def _validate_skill_ids(skill_ids: list[str]) -> None:
    missing_ids = [skill_id for skill_id in skill_ids if store.get_skill(skill_id) is None]
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"These skill IDs do not exist: {missing_ids}")


@router.get("/agents", response_model=list[AgentDefinition])
def list_agents() -> list[AgentDefinition]:
    return store.list_agents()


@router.post("/agents", response_model=AgentDefinition)
def create_agent(payload: AgentDefinitionCreate) -> AgentDefinition:
    _validate_skill_ids(payload.skill_ids)
    return store.create_agent(payload)


@router.put("/agents/{agent_id}", response_model=AgentDefinition)
def update_agent(agent_id: str, payload: AgentDefinitionUpdate) -> AgentDefinition:
    _validate_skill_ids(payload.skill_ids)
    updated = store.update_agent(agent_id, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return updated


@router.get("/workflows", response_model=list[WorkflowDefinition])
def list_workflows() -> list[WorkflowDefinition]:
    return store.list_workflows()


def _required_agent_count(workflow_type: str) -> int:
    for template in store.get_templates():
        if template.type == workflow_type:
            return template.required_agent_count
    return 2


@router.post("/workflows", response_model=WorkflowDefinition)
def create_workflow(payload: WorkflowDefinitionCreate) -> WorkflowDefinition:
    missing_ids = [agent_id for agent_id in payload.specialist_agent_ids if store.get_agent(agent_id) is None]
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"These agent IDs do not exist: {missing_ids}")

    required_count = _required_agent_count(payload.type)
    if len(payload.specialist_agent_ids) < required_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{payload.type} requires at least {required_count} agents, "
                f"but got {len(payload.specialist_agent_ids)}."
            ),
        )
    return store.create_workflow(payload)


@router.put("/workflows/{workflow_id}", response_model=WorkflowDefinition)
def update_workflow(workflow_id: str, payload: WorkflowDefinitionUpdate) -> WorkflowDefinition:
    missing_ids = [agent_id for agent_id in payload.specialist_agent_ids if store.get_agent(agent_id) is None]
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"These agent IDs do not exist: {missing_ids}")

    required_count = _required_agent_count(payload.type)
    if len(payload.specialist_agent_ids) < required_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{payload.type} requires at least {required_count} agents, "
                f"but got {len(payload.specialist_agent_ids)}."
            ),
        )

    updated = store.update_workflow(workflow_id, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return updated


def _resolve_agents(workflow: WorkflowDefinition) -> list[AgentDefinition]:
    agents = [store.get_agent(agent_id) for agent_id in workflow.specialist_agent_ids]
    return [agent for agent in agents if agent is not None]


@router.get("/workflows/{workflow_id}/graph", response_model=WorkflowGraph)
def get_workflow_graph(workflow_id: str) -> WorkflowGraph:
    workflow = store.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    agents = _resolve_agents(workflow)
    if workflow.type == "router_specialists":
        return build_router_graph(workflow, agents)
    if workflow.type == "planner_executor":
        return build_planner_graph(workflow, agents)
    if workflow.type == "supervisor_dynamic":
        return build_supervisor_graph(workflow, agents)
    if workflow.type == "single_agent_chat":
        return build_single_agent_graph(workflow, agents)

    raise HTTPException(status_code=400, detail=f"Unsupported workflow type: {workflow.type}")


def _dispatch_run(
    workflow: WorkflowDefinition,
    user_input: str,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> WorkflowRunResponse:
    if workflow.type == "router_specialists":
        return run_router_specialists(store, workflow, user_input, on_event=on_event)
    if workflow.type == "planner_executor":
        return run_planner_executor(store, workflow, user_input, on_event=on_event)
    if workflow.type == "supervisor_dynamic":
        return run_supervisor_dynamic(store, workflow, user_input, on_event=on_event)
    if workflow.type == "single_agent_chat":
        return run_single_agent_chat(store, workflow, user_input, on_event=on_event)
    raise HTTPException(status_code=400, detail=f"Unsupported workflow type: {workflow.type}")


@router.post("/runs", response_model=WorkflowRunResponse)
def run_workflow(payload: WorkflowRunRequest) -> WorkflowRunResponse:
    workflow = store.get_workflow(payload.workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return _dispatch_run(workflow, payload.user_input)


@router.post("/runs/stream")
def run_workflow_stream(payload: WorkflowRunRequest) -> StreamingResponse:
    workflow = store.get_workflow(payload.workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    stream_queue: queue.Queue[tuple[str, dict | None]] = queue.Queue()

    def on_trace(event: TraceEvent) -> None:
        stream_queue.put(("trace", event.model_dump()))

    def worker() -> None:
        try:
            result = _dispatch_run(workflow, payload.user_input, on_event=on_trace)
            stream_queue.put(("final", result.model_dump()))
        except Exception as error:  # noqa: BLE001
            stream_queue.put(("error", {"message": str(error)}))
        finally:
            stream_queue.put(("end", None))

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            event_name, body = stream_queue.get()
            if event_name == "end":
                yield "event: end\ndata: {}\n\n"
                break
            yield f"event: {event_name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
