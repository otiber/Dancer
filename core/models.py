from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DisruptionType(str, Enum):
    PARAMETRIC = "parametric_fluctuation"
    STRUCTURAL = "structural_failure"
    SOVEREIGNTY = "sovereignty_conflict"


class ShippingMethod(str, Enum):
    STANDARD = "Standard_Shipping"
    EXPRESS_AIR = "Express_Air_Freight"
    CONSOLIDATED_RAIL = "Consolidated_Rail"
    LAND_TRANSPORT = "Land_Transport"
    THIRD_PARTY_3PL = "Third_Party_Logistics"
    ALTERNATIVE_PORT = "Alternative_Port_B"
    EXPEDITED_GROUND = "Expedited_Ground"


class Proposal(BaseModel):
    id: str
    method: ShippingMethod
    cost_normalized: float = Field(ge=0.0, le=1.0)
    time_days: float = Field(gt=0.0)
    description: str
    requires_new_agent: bool = False
    new_agent_id: Optional[str] = None


class UtilityVector(BaseModel):
    f_res: float = Field(ge=0.0, le=1.0, description="Resource availability")
    f_time: float = Field(ge=0.0, le=1.0, description="Deadline slack")
    f_pol: float = Field(ge=0.0, le=1.0, description="Policy/cost compliance")


class Bid(BaseModel):
    agent_id: str
    proposal_id: str
    utility_vector: UtilityVector
    utility_score: float
    accepted: bool
    rejection_reason: Optional[str] = None


class RunResult(BaseModel):
    scenario: str
    system: str
    iteration: int
    success: bool
    rounds: int
    messages_sent: int
    total_tokens: int
    final_utility: float
    latency_seconds: float
    # Committed-proposal provenance (audit F10): without these, constant or
    # degenerate LLM outputs are invisible in the results CSV.
    proposal_method: Optional[str] = None
    proposal_cost: Optional[float] = None
    proposal_time: Optional[float] = None
    proposal_new_agent: Optional[str] = None
    n_accepting: Optional[int] = None     # agents individually accepting (diagnostic)
    refusals: Optional[str] = None        # CNP/ζ-CNP refusal-reason summary
