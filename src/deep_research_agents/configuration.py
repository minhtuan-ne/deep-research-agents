"""Configuration for the deep research agent.

A single Pydantic model holds every tunable knob, and knows how to build
itself from either a LangGraph RunnableConfig (per-request `configurable`
dict) or environment variables.
"""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(str, Enum):
    """Which search backend the researcher should use."""

    TAVILY = "tavily"
    NONE = "none"


class Configuration(BaseModel):
    """Top-level configuration for the agent."""

    # General
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=3)

    # Research loop limits
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY)
    max_researcher_iterations: int = Field(default=4)
    max_react_tool_calls: int = Field(default=6)

    # Models -- provider:model strings, passed straight into init_chat_model
    research_model: str = Field(default="openai:gpt-4.1-mini")
    research_model_max_tokens: int = Field(default=8000)
    compression_model: str = Field(default="openai:gpt-4.1-mini")
    compression_model_max_tokens: int = Field(default=6000)
    final_report_model: str = Field(default="openai:gpt-4.1")
    final_report_model_max_tokens: int = Field(default=8000)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Build a Configuration, preferring env vars over request-supplied values."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
