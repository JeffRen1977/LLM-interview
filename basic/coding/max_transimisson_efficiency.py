"""
HVDC grid: find source -> destination path that maximizes
product of edge efficiencies (directed edges).

Example: 0 --0.9--> 1 --0.8--> 2 --0.7--> 3  =>  0.9 * 0.8 * 0.7 = 0.504
"""
import heapq
from collections import defaultdict


class MaxTransimissionEfficiency:
    def __init__(self, n: int, routes: list):
        self.n = n
        self.graph = defaultdict(list)
        for u, v, efficiency in routes:
            self.graph[u].append((v, efficiency))

    def find_max_efficiency_path(self, source: int, destination: int) -> float:
        hq = []
        heapq.heappush(hq, (-1.0, source))
        max_efficiency = [0.0] * self.n
        max_efficiency[source] = 1.0

        while hq:
            neg_efficiency, node = heapq.heappop(hq)
            efficiency = -neg_efficiency
            if efficiency < max_efficiency[node]:
                continue
            if node == destination:
                return efficiency
            for neighbor, edge_efficiency in self.graph[node]:
                next_efficiency = efficiency * edge_efficiency
                if next_efficiency > max_efficiency[neighbor]:
                    max_efficiency[neighbor] = next_efficiency
                    heapq.heappush(hq, (-next_efficiency, neighbor))

        return max_efficiency[destination]


if __name__ == "__main__":
    n = 4
    routes = [[0, 1, 0.9], [1, 2, 0.8], [2, 3, 0.7]]
    solver = MaxTransimissionEfficiency(n, routes)
    print(solver.find_max_efficiency_path(0, 3))  # 0.504
