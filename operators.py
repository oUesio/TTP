'''
Genetic operators
(ox_crossover, mutate_tour_swap, inherit_items, greedy_density).
'''

import random
from typing import List

def random_tour(nc: int) -> List[int]:
    perm=list(range(nc))
    tail=perm[1:]; random.shuffle(tail)
    return [0]+tail

def ox_crossover(a: List[int], b: List[int]) -> List[int]:
    assert a[0]==0 and b[0]==0
    A=a[1:]; B=b[1:]; n1=len(A)
    i,j=sorted(random.sample(range(n1),2))
    mid=A[i:j+1]
    rem=[x for x in B if x not in mid]
    return [0]+rem[:i]+mid+rem[i:]

### def eax_crossover(...):      

def mutate_tour_swap(pi: List[int], p):
    if random.random()<p:
        n=len(pi); i,j=random.sample(range(1,n),2)
        pi[i],pi[j]=pi[j],pi[i]

def inherit_items(za: List[bool], zb: List[bool]) -> List[bool]:
    n=len(za); c=[False]*n
    for i in range(n):
        c[i] = za[i] if za[i]==zb[i] else (random.random()<0.5)
    return c

def greedy_density(weights, profits, maxW) -> List[bool]:
    idx=list(range(len(weights)))
    idx.sort(key=lambda j: profits[j]/(weights[j]+1e-9), reverse=True)
    z=[False]*len(weights); W=0.0
    for j in idx:
        w=weights[j]
        if W+w <= maxW:
            z[j]=True; W+=w
    return z

### def two_point_crossover(...):