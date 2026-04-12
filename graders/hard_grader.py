

def grade(env):
    admitted = len(env.admitted_patients)
    waiting = len(env.waiting_patients)

    total = admitted + waiting
    if total == 0:
        return 0.0

    # 1️⃣ efficiency
    efficiency = admitted / total

    # 2️⃣ critical patient handling
    critical_total = sum(
        1 for p in env.waiting_patients + env.admitted_patients if p.severity >= 8
    )
    critical_handled = sum(
        1 for p in env.admitted_patients if p.severity >= 8
    )

    critical_score = (
        critical_handled / critical_total if critical_total > 0 else 1.0
    )

    # 3️⃣ waiting penalty
    total_wait = sum(p.wait_time for p in env.waiting_patients)
    wait_penalty = 1 / (1 + total_wait)

    # final score
    score = (
        0.4 * efficiency +
        0.4 * critical_score +
        0.2 * wait_penalty
    )

    return max(0.0, min(1.0, score))


