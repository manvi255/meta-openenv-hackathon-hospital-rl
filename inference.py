import os
import asyncio
from typing import List, Optional
from openai import OpenAI
from client import HospitalEnv

try:
    from models import HospitalAction
except (ImportError, ModuleNotFoundError):
    from .models import HospitalAction

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN")

ENV_URL = "https://manvigupta1620-hospital-env.hf.space"

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "hospital_env"
TASKS = ["easy", "medium", "hard"]

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_TOTAL_REWARD = MAX_STEPS * 1.0

# -----------------------------
# 🧠 LLM CLIENT
# -----------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# -----------------------------
# 🪵 LOGGING
# -----------------------------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    # Clamp score strictly between 0.001 and 0.999 as per validation rules
    clamped_score = min(max(score, 0.001), 0.999)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={clamped_score:.3f} rewards={rewards_str}",
        flush=True,
    )

# -----------------------------
# 🤖 LLM ACTION
# -----------------------------
def get_action_from_llm(observation) -> HospitalAction:
    prompt = f"""
You are managing a hospital.

Choose ONE action:
- ADMIT patient_id
- ASSIGN patient_id ICU
- ASSIGN patient_id GENERAL
- ASSIGN patient_id EMERGENCY
- DISCHARGE patient_id
- REJECT patient_id

State:
{observation}

Respond exactly:
ACTION_TYPE patient_id target(optional)
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a hospital optimization agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        text = (response.choices[0].message.content or "").strip()
        parts = text.split()

        action_type = parts[0].upper() if len(parts) > 0 else "REJECT"
        patient_id = parts[1] if len(parts) > 1 else None
        target = parts[2] if len(parts) > 2 else None

        if action_type not in ["ADMIT", "ASSIGN", "DISCHARGE", "REJECT"]:
            return HospitalAction(action_type="REJECT")

        return HospitalAction(
            action_type=action_type,
            patient_id=patient_id,
            target_bed=target,
        )

    except Exception:
        return HospitalAction(action_type="REJECT")

# -----------------------------
# 🚀 MAIN
# -----------------------------
async def main():
    for task_name in TASKS:
        env = HospitalEnv(base_url=ENV_URL)
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset(task_name=task_name)
            last_obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_obj = get_action_from_llm(last_obs)

                action_str = action_obj.action_type
                if action_obj.patient_id:
                    action_str += f" {action_obj.patient_id}"
                if action_obj.target_bed:
                    action_str += f" {action_obj.target_bed}"

                try:
                    result = await env.step(action_obj)
                    reward = result.reward or 0.0
                    done = result.done
                    error = None
                except Exception as e:
                    reward = 0.0
                    done = True
                    error = str(e)

                rewards.append(reward)
                steps_taken = step

                log_step(step, action_str, reward, done, error)

                if done:
                    break

                last_obs = result.observation

            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            try:
                await env.close()
            except Exception:
                pass

            log_end(success, steps_taken, score, rewards)


# -----------------------------
# ▶️ RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())