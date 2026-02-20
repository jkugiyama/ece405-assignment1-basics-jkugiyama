import math


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # 1. Warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 2. After cosine schedule finishes
    if it >= cosine_cycle_iters:
        return min_learning_rate

    # 3. Cosine decay
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    return min_learning_rate + 0.5 * (
        1 + math.cos(math.pi * progress)
    ) * (max_learning_rate - min_learning_rate)