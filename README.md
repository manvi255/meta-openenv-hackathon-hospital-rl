# 🏥 Hospital Resource Management RL Environment (OpenEnv)

## 📌 Overview

This project implements a **real-world reinforcement learning (RL) environment** using the OpenEnv framework to simulate hospital operations. The environment models dynamic patient arrivals, resource constraints, and intelligent decision-making for patient admission, bed allocation, and discharge.

The RL agent acts as a **hospital decision-maker**, optimizing patient flow and resource utilization under uncertainty.

---

## 🎯 Motivation

Efficient hospital management is a critical real-world challenge. Hospitals must:

* Handle unpredictable patient inflow
* Prioritize critical cases
* Manage limited ICU beds and staff
* Minimize waiting times

This project simulates these challenges and enables training RL agents to learn **optimal policies for healthcare operations**.

---

## 🧠 Environment Design

The environment is built using **OpenEnv** and follows its standard interface:

* `reset()` → initializes the environment
* `step(action)` → executes an action and returns next state
* `state()` → returns current environment state

The simulation runs in discrete **time steps**, with stochastic patient arrivals.

---

## 📊 Observation Space

Each observation includes:

* **Bed Availability**

  * ICU beds
  * General beds
  * Emergency beds

* **Patient Queues**

  * Waiting patients (severity, wait time, required care)
  * Admitted patients

* **Resources**

  * Available staff
  * Ventilators

* **Time Step**

  * Current simulation time

---

## 🎮 Action Space

The agent can perform the following actions:

* `ADMIT patient_id` → Admit a patient from waiting queue
* `ASSIGN patient_id bed_type` → Assign a patient to ICU / GENERAL / EMERGENCY
* `DISCHARGE patient_id` → Discharge a patient
* `REJECT patient_id` → Reject a patient

---

## ⚙️ System Dynamics

* Patients arrive probabilistically (stochastic simulation)
* Each patient has:

  * Severity (1–10)
  * Required care type
* Waiting time increases every step
* Limited resources (beds, staff, ventilators)

---

## 🏆 Tasks & Difficulty Levels

### 🟢 Easy — Correct Bed Assignment

* Assign patients to appropriate bed types
* Focus on correctness

---

### 🟡 Medium — Minimize Waiting Time

* Reduce patient waiting time
* Improve admission efficiency

---

### 🔴 Hard — Full Hospital Optimization

* Balance:

  * Critical patient handling
  * Resource utilization
  * Throughput

---

## 📈 Reward Function

The reward function provides **dense and meaningful signals**:

### ✅ Positive Rewards

* Admitting patients
* Correct bed assignment
* Treating critical patients
* Efficient bed utilization

### ❌ Penalties

* Long waiting times (severity-weighted)
* Wrong bed assignment
* Rejecting critical patients
* Premature discharge

👉 The reward is **continuous and step-based**, enabling learning throughout the episode.

---

## 🧪 Evaluation Metrics

* Average waiting time
* Bed utilization rate
* Critical patient handling success
* Throughput (patients treated)
* Overall efficiency score

---

## 🤖 Baseline Inference

The baseline agent uses the **OpenAI API** to generate actions.

### Features:

* Runs on all 3 tasks: easy, medium, hard
* Deterministic (temperature = 0.0)
* Produces reproducible scores

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd kernel_env
```

---

### 2. Install Dependencies

```bash
pip install -r server/requirements.txt
```

---

### 3. Run Locally

```bash
uvicorn server.app:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

### 4. Run Inference

```bash
python inference.py
```

---

## 🐳 Docker Setup

### Build

```bash
docker build -t hospital-env -f server/Dockerfile .
```

### Run

```bash
docker run -p 8000:8000 hospital-env
```

---

## ☁️ Deployment (Hugging Face Space)

* Create a new Space with **Docker SDK**
* Upload repository
* Add tag: `openenv`
* The app runs automatically via Docker

---

## 📊 Baseline Results

| Task   | Score Range |
| ------ | ----------- |
| Easy   | 0.4 – 0.7   |
| Medium | 0.3 – 0.6   |
| Hard   | 0.2 – 0.5   |

*(Scores depend on environment randomness and policy quality)*

---

## ✅ OpenEnv Compliance

✔ Typed Pydantic models (Action, Observation)
✔ step / reset / state implemented
✔ openenv.yaml configured
✔ Tasks + graders implemented
✔ Reward function with partial progress signals
✔ Baseline inference script

---

## 🏁 Conclusion

This project demonstrates how reinforcement learning can be applied to **real-world healthcare optimization problems**, balancing efficiency, fairness, and critical decision-making under constraints.

---

## 👩‍💻 Author

Manvi Gupta Rituraj Datta Riddhima Jaiswal


```
```
