

def grade(env):
    total = len(env.admitted_patients)
    if total == 0:
        return 0.0

    correct = 0

    for p in env.admitted_patients:
        # correct if assigned care matches requirement
        if p.required_care in ["ICU", "GENERAL", "EMERGENCY"]:
            correct += 1

    # also penalize if beds overused
    overuse_penalty = sum(
        max(0, -v) for v in env.available_beds.values()
    )

    score = correct / total
    score -= 0.1 * overuse_penalty

    return max(0.0, min(1.0, score))



