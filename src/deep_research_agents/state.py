"""Graph state definitions and structured-output schemas."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured outputs (also doubles as tool schemas for the supervisor)
###################
class ConductResearch(BaseModel):
    """Call this tool to delegate research on one specific topic."""

    research_topic: str = Field(
        description="A single research topic, described in detail (at least a paragraph)."
    )


class ResearchComplete(BaseModel):
    """Call this tool to signal that research is complete."""


class ClarifyWithUser(BaseModel):
    """Whether the user's request needs clarification before research starts."""

    need_clarification: bool
    clarifying_points: list[str] = Field(
        default_factory=list,
        description=(
            "All distinct aspects of the request that are ambiguous or under-specified. "
            "Each item is one short question. Leave empty if need_clarification is False."
        ),
    )
    combined_question: str = Field(
        description=(
            "A single, natural-sounding message that weaves all clarifying_points together. "
            "Ask everything at once so the user only has to reply once. "
            "Leave empty if need_clarification is False."
        ),
    )
    verification: str = Field(
        description="Message confirming research will now start. Used when need_clarification is False."
    )


class ResearchQuestion(BaseModel):
    """A research brief distilled from the conversation."""

    research_brief: str


###################
# Reducer
###################
def override_reducer(current_value, new_value):
    """Append by default; fully replace when given {"type": "override", "value": [...]}."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


###################
# Graph states
###################
class AgentInputState(MessagesState):
    """Public input contract of the top-level graph: just messages."""


class AgentState(MessagesState):
    """Full state of the top-level graph."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    notes: Annotated[list[str], override_reducer] = []
    final_report: str


class SupervisorState(TypedDict):
    """State of the supervisor subgraph."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int


class ResearcherState(TypedDict):
    """State of an individual researcher subgraph instance."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str


class ResearcherOutputState(BaseModel):
    """What a researcher subgraph returns to its caller."""

    compressed_research: str
