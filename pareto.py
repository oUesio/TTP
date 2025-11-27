'''
Pareto archive logic
(pareto_insert).
'''

import math
from typing import List
from representation import Individual

def pareto_insert(archive: List[Individual], cand: Individual) -> None:
    if not (math.isfinite(cand.obj[0]) and math.isfinite(cand.obj[1])): return
    # dominated by existing?
    for s in archive:
        if (s.obj[0] <= cand.obj[0] and s.obj[1] <= cand.obj[1]) and (s.obj != cand.obj):
            return
    # keep only those not dominated by cand
    nd=[]
    for s in archive:
        if (cand.obj[0] <= s.obj[0] and cand.obj[1] <= s.obj[1]) and (cand.obj != s.obj):
            continue
        nd.append(s)
    nd.append(cand)
    archive[:] = nd