# state.py
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from datetime import datetime
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FraudResult(TypedDict):
    """Structure for a single fraud detection result."""
    claim_id: str
    rule_id: str
    severity: Severity
    confidence: float
    reason: str
    detection_method: str  # "rule_based", "statistical", "llm_pattern"
    timestamp: str

class Rule(TypedDict):
    """Structure for an extracted rule."""
    rule_id: str
    description: str
    conditions: List[str]
    severity: Severity
    source_document: str

class SessionContext(TypedDict):
    """Session-level context."""
    session_id: str
    user_id: str
    started_at: str
    department: str
    priority: str

# ═══════════════════════════════════════════════════════════════
# MAIN STATE - This is the communication medium between all agents
# ═══════════════════════════════════════════════════════════════
class FraudDetectionState(TypedDict):
    """
    Shared state that flows through all agents.
    Each agent reads what it needs and writes its outputs.
    """
    
    # ─────────────────────────────────────────────────────────────
    # INPUT DATA (Set at start, read by agents)
    # ─────────────────────────────────────────────────────────────
    pdf_paths: List[str]              # TRD documents to parse
    claims_data: List[Dict[str, Any]] # Claims to analyze
    
    # ─────────────────────────────────────────────────────────────
    # AGENT OUTPUTS (Written by one agent, read by others)
    # ─────────────────────────────────────────────────────────────
    # Written by: RulesAgent
    extracted_rules: List[Rule]
    rules_stored_in_vectordb: bool
    
    # Written by: FraudAgent
    detected_frauds: List[FraudResult]
    max_severity: Severity
    fraud_summary: Dict[str, Any]
    
    # Written by: ReviewAgent
    reviewed_frauds: List[FraudResult]
    human_feedback: Optional[str]
    
    # Written by: NotifyAgent
    notifications_sent: bool
    notification_channels: List[str]  # ["email", "slack", "dashboard"]
    
    # ─────────────────────────────────────────────────────────────
    # CONVERSATION & MEMORY (Appended by all agents)
    # ─────────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    
    # ─────────────────────────────────────────────────────────────
    # CONTEXT (Read by all agents for personalization)
    # ─────────────────────────────────────────────────────────────
    context: SessionContext
    
    # ─────────────────────────────────────────────────────────────
    # CONTROL FLOW (Used by router/edges)
    # ─────────────────────────────────────────────────────────────
    current_step: str
    needs_human_review: bool
    error: Optional[str]
    iteration_count: int


def create_initial_state(
    pdf_paths: List[str],
    claims_data: List[Dict],
    session_id: str,
    user_id: str = "analyst-001"
) -> FraudDetectionState:
    """Factory function to create initial state."""
    
    return FraudDetectionState(
        # Inputs
        pdf_paths=pdf_paths,
        claims_data=claims_data,
        
        # Agent outputs (empty initially)
        extracted_rules=[],
        rules_stored_in_vectordb=False,
        detected_frauds=[],
        max_severity=Severity.LOW,
        fraud_summary={},
        reviewed_frauds=[],
        human_feedback=None,
        notifications_sent=False,
        notification_channels=[],
        
        # Messages (empty initially)
        messages=[],
        
        # Context
        context=SessionContext(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now().isoformat(),
            department="Claims Review",
            priority="normal"
        ),
        
        # Control flow
        current_step="start",
        needs_human_review=False,
        error=None,
        iteration_count=0
    )