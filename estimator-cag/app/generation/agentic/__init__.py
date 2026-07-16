"""Paquete agéntico: Actor-Critic-Boss (S05), agente one-shot (pre-S12) y flujo híbrido (S12).

Conviven dos approaches a propósito. El one-shot (`run_agent`) le da al agente el proyecto
entero y se conserva como BASELINE de comparación: menos control y peores resultados. El de
producción es el flujo híbrido — el determinista de S10 con el agente solo en las dos fases
donde aporta: `propose_structure` y `run_task_hours_recovery_agent`.
"""

from app.domain.agent_trace import AgentStep, AgentTrace
from app.generation.agentic.agent import build_system_prompt, partition_output, run_agent
from app.generation.agentic.agent_loop import run_loop
from app.generation.agentic.agent_schemas import (
    AgentEstimate,
    AgentResult,
    ComponentEstimate,
)
from app.generation.agentic.boss import BossService
from app.generation.agentic.critic import CriticService
from app.generation.agentic.flagging import flag_task_estimates
from app.generation.agentic.recovery_agent import run_task_hours_recovery_agent
from app.generation.agentic.structure_agent import propose_structure
from app.generation.agentic.structured_service import estimate_task_hours_agentic

__all__ = [
    "AgentEstimate",
    "AgentResult",
    "AgentStep",
    "AgentTrace",
    "BossService",
    "ComponentEstimate",
    "CriticService",
    "build_system_prompt",
    "estimate_task_hours_agentic",
    "flag_task_estimates",
    "partition_output",
    "propose_structure",
    "run_agent",
    "run_loop",
    "run_task_hours_recovery_agent",
]
