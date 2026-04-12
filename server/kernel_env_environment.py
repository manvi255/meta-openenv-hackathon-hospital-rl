
from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HospitalAction, HospitalObservation, Patient, BedAvailability, Resources
except ImportError:
    from models import HospitalAction, HospitalObservation, Patient, BedAvailability, Resources


class KernelEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.reset()

    # -----------------------------
    # 🔄 RESET
    # -----------------------------
    def reset(self) -> HospitalObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.time_step = 0

        self.max_beds = {
            "ICU": 5,
            "GENERAL": 10,
            "EMERGENCY": 5,
        }

        self.available_beds = self.max_beds.copy()

        self.waiting_patients = []
        self.admitted_patients = []

        self.resources = Resources(ventilators=5, staff=10)

        return self._get_observation(reward=0.0, done=False)

    # -----------------------------
    # ⚡ STEP
    # -----------------------------
    def step(self, action: HospitalAction) -> HospitalObservation:
        self._state.step_count += 1
        self.time_step += 1

        reward = 0.0
        done = False

        # 1️⃣ Generate patients
        self._generate_patients()

        # 2️⃣ Apply action (IMPROVED REWARD)
        if action.action_type == "ADMIT":
            for p in self.waiting_patients:
                if p.id == action.patient_id:
                    reward += 0.5 + (p.severity * 0.1)
            reward += self._admit(action.patient_id)

        elif action.action_type == "ASSIGN":
            for p in self.admitted_patients:
                if p.id == action.patient_id:
                    if p.required_care == action.target_bed:
                        reward += 1.0 + (p.severity * 0.1)
                    else:
                        reward -= 1.0
            reward += self._assign(action.patient_id, action.target_bed)

        elif action.action_type == "DISCHARGE":
            for p in self.admitted_patients:
                if p.id == action.patient_id:
                    if p.severity <= 4:
                        reward += 1.5
                    else:
                        reward -= 2.0
            reward += self._discharge(action.patient_id)

        elif action.action_type == "REJECT":
            for p in self.waiting_patients:
                if p.id == action.patient_id:
                    if p.severity >= 8:
                        reward -= 3.0
                    else:
                        reward -= 0.5

        # 3️⃣ Waiting penalty (severity-aware)
        for p in self.waiting_patients:
            p.wait_time += 1
            reward -= 0.05 * p.wait_time * (1 + p.severity / 10)

        # 4️⃣ Utilization reward
        used_beds = sum(self.max_beds[b] - self.available_beds[b] for b in self.available_beds)
        total_beds = sum(self.max_beds.values())
        utilization = used_beds / total_beds
        reward += 0.5 * utilization

        # 5️⃣ Termination condition
        if self.time_step >= 20:
            done = True

        return self._get_observation(reward=reward, done=done)

    # -----------------------------
    # 🏥 PATIENT ARRIVAL
    # -----------------------------
    def _generate_patients(self):
        if random.random() < 0.7:
            patient = Patient(
                id=str(uuid4()),
                severity=random.randint(1, 10),
                wait_time=0,
                required_care=random.choice(["ICU", "GENERAL", "EMERGENCY"]),
                status="WAITING",
            )
            self.waiting_patients.append(patient)

    # -----------------------------
    # 🏥 ADMIT
    # -----------------------------
    def _admit(self, patient_id):
        for p in self.waiting_patients:
            if p.id == patient_id:
                p.status = "ADMITTED"
                self.admitted_patients.append(p)
                self.waiting_patients.remove(p)
                return 1.0
        return -1.0

    # -----------------------------
    # 🛏 ASSIGN BED
    # -----------------------------
    def _assign(self, patient_id, bed_type):
        if bed_type not in self.available_beds:
            return -1.0

        if self.available_beds[bed_type] <= 0:
            return -2.0

        for p in self.admitted_patients:
            if p.id == patient_id:
                self.available_beds[bed_type] -= 1
                return 2.0

        return -1.0

    # -----------------------------
    # 🚪 DISCHARGE
    # -----------------------------
    def _discharge(self, patient_id):
        for p in self.admitted_patients:
            if p.id == patient_id:
                p.status = "DISCHARGED"
                self.admitted_patients.remove(p)
                return 2.0
        return -1.0

    # -----------------------------
    # 👁 OBSERVATION
    # -----------------------------
    def _get_observation(self, reward=0.0, done=False):
        return HospitalObservation(
            available_beds=BedAvailability(**self.available_beds),
            waiting_patients=self.waiting_patients,
            admitted_patients=self.admitted_patients,
            resources=self.resources,
            time_step=self.time_step,
            reward=reward,
            done=done,
        )

    # -----------------------------
    # 📦 STATE
    # -----------------------------
    @property
    def state(self) -> State:
        return self._state


