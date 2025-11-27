'''
Parsing the TTP instance and building auxiliary structures
(load_ttp, build_items_at_city, build_candidate_lists, ceil2d).
'''

from typing import List, Dict, Any
import math, re

def load_ttp(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    num_cities = None
    num_items = None
    coords: List[List[float]] = []
    profits: List[float] = []
    weights: List[float] = []
    city_of_item: List[int] = []
    min_speed = None
    max_speed = None
    max_weight = None
    renting_ratio = None  # loaded, but unused in bi-objective

    i = 0
    while i < len(lines):
        line = lines[i]

        if "DIMENSION" in line:
            num_cities = int(line.split(":")[1].strip())
            coords = [[0.0, 0.0] for _ in range(num_cities)]

        elif "NUMBER OF ITEMS" in line:
            num_items = int(line.split(":")[1].strip())
            profits = [0.0] * num_items
            weights = [0.0] * num_items
            city_of_item = [0] * num_items

        elif "RENTING RATIO" in line:
            renting_ratio = float(line.split(":")[1].strip())

        elif "CAPACITY OF KNAPSACK" in line:
            max_weight = float(line.split(":")[1].strip())

        elif "MIN SPEED" in line:
            min_speed = float(line.split(":")[1].strip())

        elif "MAX SPEED" in line:
            max_speed = float(line.split(":")[1].strip())

        elif "EDGE_WEIGHT_TYPE" in line:
            ew = line.split(":")[1].strip()
            if ew != "CEIL_2D":
                raise RuntimeError("Only CEIL_2D supported")

        elif "NODE_COORD_SECTION" in line:
            for j in range(num_cities):
                i += 1
                parts = re.split(r"\s+", lines[i].strip())
                coords[j][0] = float(parts[1])
                coords[j][1] = float(parts[2])

        elif "ITEMS SECTION" in line:
            for j in range(num_items):
                i += 1
                parts = re.split(r"\s+", lines[i].strip())
                profits[j] = float(parts[1])
                weights[j] = float(parts[2])
                city_of_item[j] = int(parts[3]) - 1

        i += 1

    return {
        "coords": coords,
        "num_cities": num_cities,
        "num_items": num_items,
        "profits": profits,
        "weights": weights,
        "city_of_item": city_of_item,
        "max_weight": max_weight,
        "min_speed": min_speed,
        "max_speed": max_speed,
        "R": renting_ratio,  # loaded for completeness; NOT used in bi-objective
    }

def build_items_at_city(num_cities: int, city_of_item: List[int]) -> List[List[int]]:
    items_at_city: List[List[int]] = [[] for _ in range(num_cities)]
    for j, c in enumerate(city_of_item):
        items_at_city[c].append(j)
    return items_at_city

def build_candidate_lists(coords, k):
    """
    For each city, return its k nearest neighbours.
    Used to restrict 2-opt neighbourhood.
    """
    n = len(coords)
    cand = [[] for _ in range(n)]
    # compute full distance matrix row by row
    for i in range(n):
        drow = []
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            d = dx*dx + dy*dy  # square dist is enough for sorting
            drow.append((d, j))
        drow.sort(key=lambda x: x[0])
        cand[i] = [j for _, j in drow[:k]]
    return cand

# CEIL_2D distance
def ceil2d(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.ceil(math.sqrt(dx*dx + dy*dy))