from __future__ import annotations
import random
from typing import Optional
from core.models import Proposal, UtilityVector, Bid


class IPAgent:
    """Intelligent Process Agent (IPA) with local constraints and utility decomposition.

    Utility decomposition (DANCER eq. 1):
        L_k(p) = w_res * f_res  +  w_time * f_time  +  w_pol * f_pol

    f_res is committed once per negotiation run (via committed_f_res) so that
    an agent's availability is consistent across all negotiation rounds.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        max_cost: float = 1.0,
        min_utility: float = 0.5,
        max_delay_days: float = 10.0,
        availability: float = 1.0,
        weights: Optional[dict] = None,
        is_active: bool = True,
        committed_f_res: Optional[float] = None,  # committed once per run
    ):
        self.agent_id = agent_id
        self.role = role
        self.max_cost = max_cost
        self.min_utility = min_utility
        self.max_delay_days = max_delay_days
        self.availability = availability
        self.weights = weights or {"w_res": 0.4, "w_time": 0.4, "w_pol": 0.2}
        self.is_active = is_active
        # Commit f_res once so all rounds use the same resource score
        if committed_f_res is not None:
            self.committed_f_res = committed_f_res
        else:
            if not is_active:
                self.committed_f_res = 0.0
            elif random.random() < availability:
                self.committed_f_res = random.uniform(0.85, 1.0)
            else:
                self.committed_f_res = 0.0  # busy this run

    # ------------------------------------------------------------------
    def _f_res(self) -> float:
        """Resource availability score (committed at construction time)."""
        return self.committed_f_res

    def _f_time(self, proposal: Proposal) -> float:
        """Deadline slack score: 1.0 when well within deadline, 0.0 when far past."""
        if proposal.time_days <= self.max_delay_days:
            return max(0.0, 1.0 - proposal.time_days / (self.max_delay_days * 2.0))
        else:
            overshoot = (proposal.time_days - self.max_delay_days) / self.max_delay_days
            return max(0.0, 1.0 - overshoot)

    def _f_pol(self, proposal: Proposal) -> float:
        """Policy/cost compliance score: punishes proposals over budget."""
        if proposal.cost_normalized <= self.max_cost:
            return max(0.0, 1.0 - proposal.cost_normalized / (self.max_cost * 1.5))
        else:
            overage = (proposal.cost_normalized - self.max_cost) / (self.max_cost + 1e-9)
            return max(0.0, 1.0 - overage * 2.0)

    # ------------------------------------------------------------------
    def calculate_utility(self, proposal: Proposal) -> UtilityVector:
        return UtilityVector(
            f_res=self._f_res(),
            f_time=self._f_time(proposal),
            f_pol=self._f_pol(proposal),
        )

    def compute_local_utility(self, uv: UtilityVector) -> float:
        w = self.weights
        return w["w_res"] * uv.f_res + w["w_time"] * uv.f_time + w["w_pol"] * uv.f_pol

    def evaluate_proposal(self, proposal: Proposal) -> Bid:
        uv = self.calculate_utility(proposal)
        score = self.compute_local_utility(uv)
        accepted = score >= self.min_utility

        rejection_reason: Optional[str] = None
        if not accepted:
            if uv.f_pol < 0.25:
                rejection_reason = (
                    f"Cost constraint violated: proposal cost {proposal.cost_normalized:.2f} "
                    f"exceeds budget {self.max_cost:.2f}"
                )
            elif uv.f_time < 0.25:
                rejection_reason = (
                    f"Time constraint violated: {proposal.time_days:.1f} days "
                    f"exceeds max delay {self.max_delay_days:.1f} days"
                )
            elif uv.f_res < 0.25:
                rejection_reason = (
                    f"Resource unavailable: {self.agent_id} availability={self.availability:.0%}"
                )
            else:
                rejection_reason = (
                    f"Utility {score:.3f} below minimum threshold {self.min_utility:.2f}"
                )

        return Bid(
            agent_id=self.agent_id,
            proposal_id=proposal.id,
            utility_vector=uv,
            utility_score=score,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "max_cost": self.max_cost,
            "min_utility": self.min_utility,
            "max_delay_days": self.max_delay_days,
            "availability": self.availability,
            "weights": self.weights,
            "is_active": self.is_active,
            "committed_f_res": self.committed_f_res,   # preserves commitment across rounds
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IPAgent":
        # Filter keys to only those accepted by __init__
        import inspect
        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
