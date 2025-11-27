'''
Main Hybrid Genetic Search algorithm orchestration
(run_hgs, seeding strategies).
'''

import math, random, time
from typing import List, Dict, Any
import time
from representation import Individual
from evaluation import evaluate_ttp, compute_hv
from pareto import pareto_insert
from local_search import two_opt_once, local_search
from operators import ox_crossover, mutate_tour_swap, greedy_density, random_tour, inherit_items
from loader import build_items_at_city, build_candidate_lists, ceil2d
from nsga2 import nondominated_sort, crowding_distance

def run_hgs(prob: Dict[str, Any]):
    random.seed(42)
    # === CENTRAL HYPERPARAMETER BLOCK ===

    PRINT_EVERY = 1     # reduce or increase depending on variables being tested

    # Population structure
    POP_SIZE = 50
    OFFSPRING = 50
    GENS = 100

    # Tournament size (new)
    TOUR_K = 3           # selection pressure; 2 = minimal, >3 = more greedy

    # Archive control
    MAX_ARCHIVE = 4000  # Maximum number of non-dominated solutions stored in the external archive

    # Mutation
    MUT_TOUR_P = 0.01    # Probability the tour receives a swap mutation (one swap of two cities).
    MUT_ITEM_P = 0.01    # Probability that one random item bit flips after crossover.

    # Local search
    LS_MAX_ITERS = 100    # Max number of LS cycles per child. One LS iteration = Try one improving 2-opt move (tour), then try one improving 1-flip (items).
    C_2OPT = 8            # MAX_2OPT_CHECKS = C_2OPT * n_cities
    LS_ALPHA_SCHEDULE = [1.0, 1.0, 1.0, 0.5, 0.0]   # cycling bias toward time→balanced→profit
    CANDIDATE_K = 15   # local city search candidates

    # Bounds behaviour
    EARLY_GLOBAL_BOUNDS_GENS = 200

    # Timing
    MAX_GEN_SEC = 5000

    # Seeding ratios
    INIT_EMPTY_RATIO = 0.2      # fraction of POP
    INIT_SPEED_AWARE_RATIO = 0.33 
    INIT_PROFIT_HEAVY_RATIO = 0.33
    # remainder = random feasible

    RANDOM_PACK_PROB = 0.10



    # HV box commonly used for a280-n279 (time, -profit)
    IDEAL=(2613.0, -42036.0)
    NADIR=(5444.0, -0.0)

    prob["items_at"] = build_items_at_city(prob["num_cities"], prob["city_of_item"])

    # init population: empty, density-greedy, random feasible
    pop: List[Individual]=[]
    nC = prob["num_cities"]; nI = prob["num_items"]
    cand_lists = build_candidate_lists(prob["coords"], CANDIDATE_K)

    def seed_empty_fast():
        pi = random_tour(nC)
        for _ in range(4):
            two_opt_once(pi, evaluate_ttp, [False]*nI,
                        prob["coords"], prob["items_at"], prob["weights"], prob["profits"],
                        prob["max_weight"], prob["min_speed"], prob["max_speed"],
                        cand_lists,
                        C_2OPT * nC)


        z = [False]*nI
        return make_individual(pi,z,evaluate_ttp,prob)
    
    def seed_speed_aware():
        pi = random_tour(nC)
        # compute remaining distance from each city along the tour
        rem = [0.0] * nC
        acc = 0.0
        for i in range(nC-1, -1, -1):
            j = (i+1) % nC
            acc += ceil2d(prob["coords"], pi[i], pi[j])
            rem[i] = acc
        pos = [0] * nC
        for idx, city in enumerate(pi):
            pos[city] = idx

        cand = []
        vmax = prob["max_speed"]


        for city in range(nC):
            for j in prob["items_at"][city]:
                w = prob["weights"][j]
                p = prob["profits"][j]
                rd = rem[pos[city]]
                penalty = rd * w / vmax
                score = p - penalty
                cand.append((score, j, w))

        cand.sort(key=lambda x: x[0], reverse=True)
        z = [False] * nI
        W = 0.0
        for score, j, w in cand:
            if W + w <= prob["max_weight"]:
                z[j] = True
                W += w
        return make_individual(pi, z, evaluate_ttp, prob)


    def seed_density_greedy():
        pi = random_tour(nC)
        z = greedy_density(prob["weights"], prob["profits"], prob["max_weight"])
        return make_individual(pi,z,evaluate_ttp,prob)
    

    seeds = []

    num_empty   = max(2, int(POP_SIZE * INIT_EMPTY_RATIO))
    num_speed   = max(2, int(POP_SIZE * INIT_SPEED_AWARE_RATIO))
    num_greedy  = max(2, int(POP_SIZE * INIT_PROFIT_HEAVY_RATIO))

    seeds += [seed_empty_fast() for _ in range(num_empty)]
    seeds += [seed_speed_aware() for _ in range(num_speed)]
    seeds += [seed_density_greedy() for _ in range(num_greedy)]


    while len(seeds) < POP_SIZE:
        pi = random_tour(nC)
        z=[False]*nI; W=0.0
        for j in random.sample(range(nI), nI):
            w=prob["weights"][j]
            if random.random() < RANDOM_PACK_PROB and W+w <= prob["max_weight"]:
                z[j]=True; W+=w
        seeds.append(make_individual(pi,z,evaluate_ttp,prob))
    pop = seeds[:POP_SIZE]

    archive: List[Individual]=[]
    for ind in pop: pareto_insert(archive, ind)

    last_time = time.time()
    best_prev = float("inf")

    for g in range(GENS):
        objs=[ind.obj for ind in pop]
        fronts = nondominated_sort(objs)
        ranks=[None]*len(pop)
        for r,F in enumerate(fronts):
            for idx in F: ranks[idx]=r
        distances={}
        for F in fronts:
            distances.update(crowding_distance(F, objs))

        children: List[Individual]=[]
        for _ in range(OFFSPRING):
            # tournament select (rank, then crowding)

            def tournament_select():
                cand = random.sample(range(len(pop)), TOUR_K)
                cand_sorted = sorted(
                    cand,
                    key=lambda idx: (ranks[idx], -distances.get(idx, 0.0))
                )
                return pop[cand_sorted[0]]

            p1 = tournament_select()
            p2 = tournament_select()


            child_pi = ox_crossover(p1.pi, p2.pi)
            child_z = inherit_items(p1.z, p2.z)
            mutate_tour_swap(child_pi, p=MUT_TOUR_P)
            if random.random() < MUT_ITEM_P:
                j = random.randrange(len(child_z))
                child_z[j] = not child_z[j]


            # LS normalization bounds
            if g < EARLY_GLOBAL_BOUNDS_GENS:
                # Stable heuristic early bounds
                # (teams used constants or wide heuristic ranges)
                tmin = 2000.0
                tmax = 80000.0
                mmin = -120000.0   # -profit
                mmax = 0.0
            else:
                # Population-based bounds (what teams actually used)
                ts = [o[0] for o in objs if math.isfinite(o[0])]
                ms = [o[1] for o in objs if math.isfinite(o[1])]
                if not ts: ts = [0.0, 1.0]
                if not ms: ms = [-1.0, 0.0]

                tmin, tmax = min(ts), max(ts)
                mmin, mmax = min(ms), max(ms)

                # avoid zero ranges
                if tmax == tmin: tmax = tmin + 1.0
                if mmax == mmin: mmax = mmin + 1.0

            alpha = LS_ALPHA_SCHEDULE[g % len(LS_ALPHA_SCHEDULE)]
            MAX_2OPT_CHECKS = C_2OPT * nC

            local_search(child_pi, child_z, evaluate_ttp,
                        prob["coords"], prob["items_at"], prob["weights"], prob["profits"],
                        prob["max_weight"], prob["min_speed"], prob["max_speed"],
                        alpha, ((tmin,tmax),(mmin,mmax)),
                        max_iters=LS_MAX_ITERS,
                        max_2opt_checks=MAX_2OPT_CHECKS,
                        cand_lists=cand_lists)




            child = make_individual(child_pi, child_z, evaluate_ttp, prob)
            children.append(child)
            pareto_insert(archive, child)

            if len(archive) > MAX_ARCHIVE:
                objs_arch=[s.obj for s in archive]
                fronts_arch=nondominated_sort(objs_arch)
                new_arch=[]
                for F in fronts_arch:
                    if len(new_arch)+len(F) <= MAX_ARCHIVE:
                        new_arch += [archive[i] for i in F]
                    else:
                        d=crowding_distance(F, objs_arch)
                        F_sorted=sorted(F, key=lambda i: d.get(i,0.0), reverse=True)
                        need = MAX_ARCHIVE - len(new_arch)
                        new_arch += [archive[i] for i in F_sorted[:need]]
                        break
                archive[:] = new_arch

        # Environmental selection
        combined = pop + children
        c_objs=[ind.obj for ind in combined]
        c_fronts = nondominated_sort(c_objs)
        new_pop=[]
        for F in c_fronts:
            if len(new_pop)+len(F) <= POP_SIZE:
                new_pop += [combined[i] for i in F]
            else:
                d=crowding_distance(F, c_objs)
                F_sorted = sorted(F, key=lambda i: d.get(i,0.0), reverse=True)
                need = POP_SIZE - len(new_pop)
                new_pop += [combined[i] for i in F_sorted[:need]]
                break
        pop = new_pop

        # occasional stagnation kick on best time
        if (g+1)%200==0:
            best_now = min(ind.obj[0] for ind in archive)
            if g>0 and abs(best_now - best_prev) < 1e-9:
                for _ in range(max(1, POP_SIZE//25)):
                    pi=random_tour(nC)
                    z=[False]*nI
                    pop[random.randrange(len(pop))]=make_individual(pi,z,evaluate_ttp,prob)
            best_prev = best_now

        # per-gen wall clock cap
        if time.time()-last_time > MAX_GEN_SEC:
            print(f"[GEN TIME LIMIT HIT] g={g+1}, aborting gen early", flush=True)
            last_time=time.time()
            continue


        # logging
        if (g+1)%PRINT_EVERY==0:
            elapsed=time.time()-last_time; last_time=time.time()
            best_t = min(s.obj[0] for s in archive)
            best_profit = max(-s.obj[1] for s in archive)
            # HV box is for (time, -profit)
            hv = compute_hv([s.obj for s in archive], IDEAL, NADIR)
            print(f"Gen {g+1}: arch={len(archive)} best_time={best_t:.3f} best_profit={best_profit:.0f} HV={hv:.4f} gen_sec={elapsed:.2f}", flush=True)

    final_hv = compute_hv([s.obj for s in archive], IDEAL, NADIR)
    print(f"\nFinal HV: {final_hv:.4f}")

    # sort ND archive by time and return
    archive.sort(key=lambda s: (s.obj[0], s.obj[1]))
    return archive

def make_individual(pi, z, eval_fn, prob):
    obj = eval_fn(pi,z,prob["coords"],prob["items_at"],prob["weights"],prob["profits"],
                  prob["max_weight"],prob["min_speed"],prob["max_speed"])
    return Individual(pi,z,obj)
