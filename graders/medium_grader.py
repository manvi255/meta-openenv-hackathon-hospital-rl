

def grade(env):
    total_patients = len(env.waiting_patients) + len(env.admitted_patients)

    if total_patients == 0:
        return 1.0

    total_wait = sum(p.wait_time for p in env.waiting_patients)

    # normalize
    avg_wait = total_wait / total_patients

    score = 1 / (1 + avg_wait)

    return max(0.0, min(1.0, score))


