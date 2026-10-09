# Эталонная проверка статистики для симулятора A/B-тестов.
# Запуск: python reference_check.py
# Сравниваем формулу размера выборки и свойства Монте-Карло с тем, что должно быть на странице.

import math
import random

def probit(p):
    """Обратная нормальная CDF через math.erfinv-эквивалент (высокая точность)."""
    # Используем scipy-независимый способ: численно, но точно
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    #Binary search достаточно точен для проверки
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

ERF = math.erf

def norm_cdf(x):
    return 0.5 * (1 + ERF(x / math.sqrt(2)))

def sample_size(p1, p2, alpha, power, two_sided=True):
    z_a = probit(1 - alpha / 2 if two_sided else 1 - alpha)
    z_b = probit(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)

def gauss():
    return random.gauss(0.0, 1.0)

def simulate(p1, p2_true, n, alpha, two_sided=True, runs=20000):
    """Монте-Карло: p̂A ~ N(p1, ...), p̂B ~ N(p2_true, ...), z-тест с pooled SE."""
    sig = 0
    lifts_sig = []
    pvals = []
    for _ in range(runs):
        pa = p1 + gauss() * math.sqrt(p1 * (1 - p1) / n)
        pb = p2_true + gauss() * math.sqrt(p2_true * (1 - p2_true) / n)
        d = pb - pa
        pbar = (pa + pb) / 2
        se = math.sqrt(2 * pbar * (1 - pbar) / n)
        z = d / se
        pv = 2 * (1 - norm_cdf(abs(z))) if two_sided else 1 - norm_cdf(z)
        pvals.append(pv)
        if pv < alpha:
            sig += 1
            lifts_sig.append(d)
    mean_lift_sig = sum(lifts_sig) / len(lifts_sig) if lifts_sig else float("nan")
    return sig / runs, mean_lift_sig * 100  # доля значимых, средний лифт в п.п.

random.seed(42)

cases = [
    ("A: p=10%, MDE +10% отн., α=5%, мощность 80%, двусторонний", 0.10, 0.11, 0.05, 0.80, True),
    ("B: то же, односторонний",                                   0.10, 0.11, 0.05, 0.80, False),
    ("C: p=20%, MDE +1 п.п., α=5%, мощность 80%",                 0.20, 0.21, 0.05, 0.80, True),
    ("D: p=50%, MDE +2 п.п., α=5%, мощность 90%",                 0.50, 0.52, 0.05, 0.90, True),
]

print("=== Размер выборки (на вариант) ===")
for name, p1, p2, a, pw, ts in cases:
    print(f"{name}: n = {sample_size(p1, p2, a, pw, ts)}")

print()
print("=== Монте-Карло для случая A (n из формулы, 20000 прогонов) ===")
n = sample_size(0.10, 0.11, 0.05, 0.80, True)
rate_h1, lift_h1 = simulate(0.10, 0.11, n, 0.05, True)
print(f"Истинный эффект = MDE (1.00 п.п.): значимых {rate_h1*100:.1f}% (ожидание ~80%), "
      f"средний лифт среди значимых {lift_h1:+.2f} п.п. (ожидание ~ +1.12..1.13)")
rate_h0, lift_h0 = simulate(0.10, 0.10, n, 0.05, True)
print(f"Истинный эффект = 0:              значимых {rate_h0*100:.1f}% (ожидание ~5%)")
