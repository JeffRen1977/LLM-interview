"""
Tree path prefix matching (Word Tree).

For each prefix of `string`, count simple paths whose node labels match the prefix.

Example tree (labels in parentheses):
        0(r)
       /  |    \\
    1(o)   2(t) 3(e)
     / \\         |
  4(o) 5(r)     6(e)
   /     \\        \\
  7(o)    8(t)     9(e)

string = "trees"  ->  [2, 2, 1, 1, 0]
"""

from collections import defaultdict


class WordTree:
    def __init__(self, n: int, paths: list, labels: str, string: str):
        self.n = n
        self.paths = paths
        self.labels = labels
        self.string = string
        self.graph = defaultdict(list)

    def count_matching_paths(self) -> list[int]:
        for v1, v2 in self.paths:
            self.graph[v1].append(v2)
            self.graph[v2].append(v1)

        if not self.string:
            return []

        # (current_node, previous_node); prev=-1 means single-node path start
        active_states = [
            (i, -1)
            for i in range(self.n)
            if self.labels[i] == self.string[0]
        ]
        result = [len(active_states)]

        for char in self.string[1:]:
            next_states = []
            for curr, prev in active_states:
                for neighbor in self.graph[curr]:
                    if neighbor != prev and self.labels[neighbor] == char:
                        next_states.append((neighbor, curr))
            active_states = next_states
            result.append(len(active_states))
            if not active_states:
                break

        return result


if __name__ == "__main__":
    n = 10
    paths = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [3, 6],
        [4, 7], [5, 8], [6, 9], [7, 8], [8, 9],
    ]
    labels = ["r", "o", "t", "e", "o", "r", "e", "o", "t", "e"]
    string = "trees"

    word = WordTree(n, paths, labels, string)
    print(word.count_matching_paths())  # [2, 2, 1, 1, 0]
