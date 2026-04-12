
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under BSD-style license.

"""
Data models for Hospital RL Environment
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Dict, Optional, Literal


# -----------------------------
# 🏥 Patient Model
# -----------------------------
class Patient(Observation):  # using Observation for nested compatibility
    id: str
    severity: int = Field(..., ge=1, le=10)
    wait_time: int = Field(default=0, ge=0)
    required_care: Literal["ICU", "GENERAL", "EMERGENCY"]
    status: Literal["WAITING", "ADMITTED", "DISCHARGED"] = "WAITING"


# -----------------------------
# 🛏 Bed Availability
# -----------------------------
class BedAvailability(Observation):
    ICU: int
    GENERAL: int
    EMERGENCY: int


# -----------------------------
# 📊 Resources
# -----------------------------
class Resources(Observation):
    ventilators: int = 0
    staff: int = 0


# -----------------------------
# 👁 Observation (STATE)
# -----------------------------
class HospitalObservation(Observation):
    available_beds: BedAvailability
    waiting_patients: List[Patient]
    admitted_patients: List[Patient]
    resources: Resources
    time_step: int


# -----------------------------
# 🎮 Action Model
# -----------------------------
class HospitalAction(Action):
    action_type: Literal[
        "ADMIT",
        "ASSIGN",
        "TRANSFER",
        "DISCHARGE",
        "REJECT"
    ]
    patient_id: Optional[str] = None
    target_bed: Optional[Literal["ICU", "GENERAL", "EMERGENCY"]] = None