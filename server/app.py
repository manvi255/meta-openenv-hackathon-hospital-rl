
# Copyright (c) Meta Platforms, Inc.
# Licensed under BSD-style license.

"""
FastAPI application for the Hospital RL Environment.

Exposes endpoints:
- POST /reset
- POST /step
- GET /state
- GET /schema
- WS /ws
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies using:\n    uv sync\n"
    ) from e

# ✅ IMPORT YOUR NEW MODELS + ENV

try:
    from ..models import HospitalAction, HospitalObservation
except ImportError:
    from models import HospitalAction, HospitalObservation

try:
    from .kernel_env_environment import KernelEnvironment
except ImportError:
    from kernel_env_environment import KernelEnvironment

# ✅ CREATE APP (THIS CONNECTS EVERYTHING)

app = create_app(
    KernelEnvironment,
    HospitalAction,
    HospitalObservation,
    env_name="hospital_env",
    max_concurrent_envs=1,
)

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Hospital RL Environment is live 🚀"
    }


# -----------------------------
# 🚀 RUN SERVER (OPTIONAL)
# -----------------------------
def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
