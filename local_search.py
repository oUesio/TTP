'''
2-opt, 1-flip, and the combined local_search driver
(two_opt_once, item_1flip_LS, local_search).
'''

import random
import math
from loader import ceil2d

### def lin_kernighan(...):

### def chained_lin_kernighan(...):

### def packiterative(...):

def two_opt_once(pi, eval_fn, z, coords, items_at, weights, profits,
                 maxW, vmin, vmax, cand_lists, max_checks):
    """
    Candidate-restricted 2-opt:
      - For each edge (a,b) in tour
      - For each candidate c in cand_lists[a]  (typically k=10–20)
      - Determine d = successor of c
      - Compute pure-distance delta
      - Only if distance improves, run full TTP eval
    """
    n = len(pi)
    base_t, _ = eval_fn(pi, z, coords, items_at, weights, profits, maxW, vmin, vmax)
    checks = 0

    # map city -> index in pi
    pos = [0] * n
    for idx, city in enumerate(pi):
        pos[city] = idx

    for idx_a in range(n):
        a = pi[idx_a]
        idx_b = (idx_a + 1) % n
        b = pi[idx_b]
        dist_ab = ceil2d(coords, a, b)

        for c in cand_lists[a]:
            idx_c = pos[c]
            idx_d = (idx_c + 1) % n
            d = pi[idx_d]

            if c == a or c == b or d == a:
                continue

            dist_cd = ceil2d(coords, c, d)
            dist_ac = ceil2d(coords, a, c)
            dist_bd = ceil2d(coords, b, d)

            # Pure distance delta
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta >= 0:
                continue  # reject non-improving candidates

            # Now test full TTP objective
            checks += 1
            if checks > max_checks:
                return False

            # Perform 2-opt excl 0 (ensure correct slice direction)
            i = min(idx_a + 1, idx_c)
            k = max(idx_a + 1, idx_c)

            # forbid reversing through index 0
            if i == 0 or k == 0:
                continue

            new_pi = pi[:i] + pi[i:k+1][::-1] + pi[k+1:]


            t_new, _ = eval_fn(new_pi, z, coords, items_at, weights, profits,
                               maxW, vmin, vmax)
            if math.isfinite(t_new) and t_new + 1e-12 < base_t:
                pi[:] = new_pi
                return True

    return False

def item_1flip_LS(z, pi, eval_fn, coords, items_at, weights, profits, maxW, vmin, vmax, alpha, norm_bounds):
    (tmin,tmax),(mmin,mmax)=norm_bounds
    def score(t,m):
        tn=0 if tmax==tmin else (t-tmin)/(tmax-tmin)
        mn=0 if mmax==mmin else (m-mmin)/(mmax-mmin)
        return alpha*tn + (1-alpha)*mn
    bt,bm = eval_fn(pi,z,coords,items_at,weights,profits,maxW,vmin,vmax)
    if bt==float("inf"):  # infeasible → drop worst density until feasible
        picked=[j for j,b in enumerate(z) if b]
        picked.sort(key=lambda jj: profits[jj]/(weights[jj]+1e-9))
        for jj in picked:
            z[jj]=False
            t,m=eval_fn(pi,z,coords,items_at,weights,profits,maxW,vmin,vmax)
            if t!=float("inf"): bt, bm = t, m; break
        else:
            return False
    bs=score(bt,bm)
    order=list(range(len(z))); random.shuffle(order)
    for j in order:
        z[j]=not z[j]
        t,m=eval_fn(pi,z,coords,items_at,weights,profits,maxW,vmin,vmax)
        if t==float("inf"):
            z[j]=not z[j]; continue
        s=score(t,m)
        if s+1e-12 < bs:
            return True
        z[j]=not z[j]
    return False

def local_search(pi, z, eval_fn, coords, items_at, weights, profits, maxW, vmin, vmax, alpha, norm_bounds, max_iters, max_2opt_checks, cand_lists):



    for _ in range(max_iters):
        imp=False
        imp |= two_opt_once(pi, eval_fn, z, coords, items_at, weights, profits, maxW, vmin, vmax, cand_lists, max_2opt_checks)


        imp |= item_1flip_LS(z,pi,eval_fn,coords,items_at,weights,profits,maxW,vmin,vmax,alpha,norm_bounds)
        if not imp: break