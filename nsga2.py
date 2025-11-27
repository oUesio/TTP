'''
NSGA-II utilities
(nondominated_sort, crowding_distance, pareto dominance helpers).
'''

from typing import List, Tuple, Dict

def nondominated_sort(objs: List[Tuple[float, float]]) -> List[List[int]]:
    N = len(objs)
    S = [[] for _ in range(N)]
    n = [0] * N
    ranks = [0] * N
    fronts: List[List[int]] = [[]]

    def dom(a,b):
        return (objs[a][0] <= objs[b][0] and objs[a][1] <= objs[b][1]) and (objs[a] != objs[b])

    for p in range(N):
        for q in range(N):
            if p==q: continue
            if dom(p,q): S[p].append(q)
            elif dom(q,p): n[p]+=1
        if n[p]==0:
            ranks[p]=0
            fronts[0].append(p)
    i=0
    while fronts[i]:
        nxt=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0:
                    ranks[q]=i+1
                    nxt.append(q)
        i+=1
        fronts.append(nxt)
    if not fronts[-1]: fronts.pop()
    return fronts

def crowding_distance(front: List[int], objs: List[Tuple[float, float]]) -> Dict[int, float]:
    if not front:
        return {}
    L = len(front)
    dist = {idx: 0.0 for idx in front}
    for m in range(2):
        fs = sorted(front, key=lambda i: objs[i][m])
        dist[fs[0]] = float("inf")
        dist[fs[-1]] = float("inf")
        min_m = objs[fs[0]][m]; max_m = objs[fs[-1]][m]
        denom = (max_m - min_m) if max_m>min_m else 1.0
        for k in range(1, L-1):
            prev = objs[fs[k-1]][m]; nxt = objs[fs[k+1]][m]
            dist[fs[k]] += (nxt - prev)/denom
    return dist