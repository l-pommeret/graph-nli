from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "data" / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz"
BASE = ROOT / "data" / "conceptnet" / "conceptnet_en.tsv.gz"

_VOCAB = json.loads((ROOT / "data" / "relations.json").read_text())
KGRAPH_RELATIONS = set(_VOCAB["extraction"]) | set(_VOCAB["kgraph_extra"])

_LEXICAL = {"RelatedTo", "FormOf", "DerivedFrom", "Synonym", "SimilarTo"}


def _term(uri: str) -> str:
    return uri.split("/")[3]


def build_base(dump: Path = DUMP, out: Path = BASE) -> None:
    edges: dict[tuple[str, str, str], float] = {}
    n = 0
    with gzip.open(dump, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rel, start, end = f[1], f[2], f[3]
            if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
                continue
            if not rel.startswith("/r/") or rel.startswith("/r/dbpedia"):
                continue
            rel = rel[3:]
            if rel == "ExternalURL":
                continue
            w = json.loads(f[4]).get("weight", 1.0)
            key = (rel, _term(start), _term(end))
            if w > edges.get(key, 0.0):
                edges[key] = w
            n += 1
            if n % 1_000_000 == 0:
                print(f"[conceptnet] {n} lines read, {len(edges)} edges", flush=True)
    with gzip.open(out, "wt") as fh:
        for (rel, h, t), w in edges.items():
            fh.write(f"{rel}\t{h}\t{t}\t{w:.3f}\n")
    print(f"[conceptnet] base written: {len(edges)} edges -> {out}", flush=True)


class ConceptNet:
    def __init__(self, base: Path = BASE, relations: set[str] | None = None,
                 max_neighbors: int = 64):
        relations = relations or KGRAPH_RELATIONS
        fwd: dict[str, list] = defaultdict(list)
        bwd: dict[str, list] = defaultdict(list)
        self.edges: list[tuple[str, str, str, float]] = []
        with gzip.open(base, "rt") as fh:
            for line in fh:
                rel, h, t, w = line.rstrip("\n").split("\t")
                if rel not in relations:
                    continue
                i = len(self.edges)
                self.edges.append((h, rel, t, float(w)))
                fwd[h].append(i)
                bwd[t].append(i)

        for adj in (fwd, bwd):
            for k, idxs in adj.items():
                idxs.sort(key=lambda i: (self.edges[i][1] in _LEXICAL,
                                         -self.edges[i][3]))
                del idxs[max_neighbors:]
        self.fwd, self.bwd = dict(fwd), dict(bwd)


    def _match(self, entity: str) -> str | None:
        if entity in self.fwd or entity in self.bwd:
            return entity
        head = entity.rsplit("_", 1)[-1]
        if head != entity and (head in self.fwd or head in self.bwd):
            return head
        return None

    def _adj(self, node: str) -> list[int]:
        return self.fwd.get(node, []) + self.bwd.get(node, [])

    def kgraph(self, p_entities: list[str], h_entities: list[str],
               cap_bridges: int = 30, cap_per_entity: int = 5,
               cap_total: int = 80) -> list[list[str]]:
        p_nodes = {m for e in p_entities if (m := self._match(e))}
        h_nodes = {m for e in h_entities if (m := self._match(e))}
        p_only, h_only = p_nodes - h_nodes, h_nodes - p_nodes
        shared = p_nodes | h_nodes
        chosen: dict[int, float] = {}


        direct = [i for n in p_only for i in self._adj(n)
                  if self.edges[i][0] in h_only or self.edges[i][2] in h_only]

        two_hop: list[tuple[int, int]] = []
        h_mid = {self.edges[i][0] for n in h_only for i in self._adj(n)} | \
                {self.edges[i][2] for n in h_only for i in self._adj(n)}
        for n in p_only:
            for i in self._adj(n):
                e = self.edges[i]
                mid = e[2] if e[0] == n else e[0]
                if mid in h_mid and mid not in shared:
                    for j in self._adj(mid):
                        f = self.edges[j]
                        if f[0] in h_only or f[2] in h_only:
                            two_hop.append((i, j))
        for i in sorted(set(direct), key=lambda i: -self.edges[i][3])[:cap_bridges]:
            chosen[i] = self.edges[i][3]
        pairs = sorted(set(two_hop),
                       key=lambda ij: -min(self.edges[ij[0]][3], self.edges[ij[1]][3]))
        for i, j in pairs:
            if len(chosen) >= cap_bridges * 2:
                break
            chosen[i] = self.edges[i][3]
            chosen[j] = self.edges[j][3]


        for n in shared:
            picked = 0
            for i in self._adj(n):
                if self.edges[i][1] in _LEXICAL:
                    continue
                chosen.setdefault(i, self.edges[i][3])
                picked += 1
                if picked >= cap_per_entity:
                    break

        best = sorted(chosen, key=lambda i: -chosen[i])[:cap_total]
        return [[self.edges[i][0], self.edges[i][1], self.edges[i][2]] for i in best]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_base()
    else:
        print("usage: python -m nli.conceptnet build")
