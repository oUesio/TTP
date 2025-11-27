'''
TTP evaluation and objective handling
(evaluate_ttp, compute_hv, any normalization helpers).
'''

import math
from typing import List, Tuple
from loader import ceil2d

def evaluate_ttp(
    pi: List[int],
    z: List[bool],
    coords: List[List[float]],
    items_at_city: List[List[int]],
    weights: List[float],
    profits: List[float],
    max_weight: float,
    min_speed: float,
    max_speed: float,
) -> Tuple[float, float]:
    """
    Bi-objective TTP:
      f1 = total travel time (to minimize)
      f2 = - total profit       (to minimize)
    No renting term in bi-objective. Speed drops linearly with weight.
    """
    n = len(pi)
    if pi[0] != 0:
        raise RuntimeError("Tour must start at city 0")

    time_v = 0.0
    profit_sum = 0.0
    w = 0.0

    for i in range(n):
        city = pi[i]
        # pick items at arrival
        for j in items_at_city[city]:
            if z[j]:
                w += weights[j]
                profit_sum += profits[j]
        if w > max_weight:
            return float("inf"), float("+inf")  # infeasible dominated
        speed = max_speed - (w / max_weight) * (max_speed - min_speed)
        nxt = pi[(i + 1) % n]
        dist = ceil2d(coords, city, nxt)
        time_v += dist / max(speed, 1e-12)

    return time_v, -profit_sum

def compute_hv(objs, ideal, nadir):
    # objs in (time, -profit), both minimized
    t_ideal, m_ideal = ideal
    t_nadir, m_nadir = nadir
    if t_nadir <= t_ideal or m_nadir <= m_ideal:
        return 0.0
    pts=[]
    for (t,m) in objs:
        if not (math.isfinite(t) and math.isfinite(m)): continue
        t = min(max(t, t_ideal), t_nadir)
        m = min(max(m, m_ideal), m_nadir)
        if t < t_nadir and m < m_nadir:
            pts.append((t,m))
    if not pts: return 0.0
    pts.sort(key=lambda x: x[0])
    env=[]; best_m=float('inf')
    for (t,m) in pts:
        if m<best_m:
            best_m=m; env.append((t,best_m))
    area=0.0; prev_t=t_nadir
    for t,m in reversed(env):
        w = max(0.0, prev_t - t)
        h = max(0.0, m_nadir - m)
        area += w*h
        prev_t = t
    total = (t_nadir - t_ideal) * (m_nadir - m_ideal)
    if total <= 0: return 0.0
    hv = area/total
    return 0.0 if hv<0 else (1.0 if hv>1 else hv)

