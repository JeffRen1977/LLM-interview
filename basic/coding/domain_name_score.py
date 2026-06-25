"""
Domain name scoring.

A domain is a leaf if no other input domain is its subdomain.
Total score of a leaf = sum of scores of itself + all ancestors present in input.

Example leaves:
  mail.test.mydomain.com  -> 15+10+5+20 = 50
  www.mydomain.com        -> 10+5+20 = 35
  mail.test.com           -> 10-10+20 = 20
  www.test.com            -> -5-10+20 = 5
"""
from __future__ import annotations


class DomainNameScore:
    def __init__(self, domains: list[str], scores: list[int]):
        self.domains = domains
        self.score_map = dict(zip(domains, scores))
        self.domain_set = set(domains)

    def _ancestors(self, domain: str):
        parts = domain.split(".")
        for i in range(len(parts)):
            yield ".".join(parts[i:])

    def total_score(self, domain: str) -> int:
        return sum(self.score_map[ancestor] for ancestor in self._ancestors(domain))

    def is_leaf(self, domain: str) -> bool:
        suffix = "." + domain
        return not any(
            other != domain and other.endswith(suffix) for other in self.domain_set
        )

    def leaf_scores(self) -> dict[str, int]:
        return {
            domain: self.total_score(domain)
            for domain in self.domains
            if self.is_leaf(domain)
        }


if __name__ == "__main__":
    domains = [
        "test.mydomain.com",
        "mail.test.mydomain.com",
        "test.com",
        "com",
        "mydomain.com",
        "www.mydomain.com",
        "mail.test.com",
        "www.test.com",
    ]
    scores = [10, 15, -10, 20, 5, 10, 10, -5]

    solver = DomainNameScore(domains, scores)
    for domain, total in sorted(solver.leaf_scores().items()):
        print(f"{domain:30s} {total}")
