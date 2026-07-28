from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["entailment", "neutral", "contradiction"]


def serialize(p_triples: list, h_triples: list, k_triples: list) -> str:
    def sec(name: str, triples: list) -> str:
        if not triples:
            return f"{name}: none"
        return f"{name}: " + " ; ".join(f"{h} {r} {t}" for h, r, t in triples)

    return "\n".join([sec("knowledge", k_triples),
                      sec("premise", p_triples),
                      sec("hypothesis", h_triples)])


def load_triple_map(paths: list[str]) -> dict[str, list]:
    tmap: dict[str, list] = {}
    bad = 0
    for path in paths:
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    tmap[row["text"].strip()] = row["triples"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
    if bad:
        print(f"[dataset] {bad} malformed triple lines skipped", flush=True)
    return tmap


def graph_of(props: list[str], tmap: dict[str, list]) -> list:
    seen, out = set(), []
    for p in props:
        for h, r, t in tmap.get(p.strip(), []):
            if (h, r, t) not in seen:
                seen.add((h, r, t))
                out.append([h, r, t])
    return out


def nodes_of(triples: list) -> list[str]:
    seen: dict[str, None] = {}
    for h, _, t in triples:
        seen.setdefault(h)
        seen.setdefault(t)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description='assemble P, H, K and serialise for the classifier')
    ap.add_argument("--atomic", required=True)
    ap.add_argument("--triples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-kgraph", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="emit raw triples (p/h/k_triples) instead of serialised text")
    ap.add_argument("--cap-total", type=int, default=120)
    args = ap.parse_args()

    tmap = load_triple_map(args.triples)
    cn = None
    if not args.no_kgraph:
        from nli.conceptnet import ConceptNet
        cn = ConceptNet()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, missing_p, missing_h = 0, 0, 0
    with open(args.atomic) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["label"] not in (0, 1, 2):
                continue
            gp = graph_of(row["premise_propositions"], tmap)
            gh = graph_of(row["hypothesis_propositions"], tmap)
            missing_p += not gp
            missing_h += not gh
            gk = cn.kgraph(nodes_of(gp), nodes_of(gh),
                           cap_total=args.cap_total) if cn else []
            if args.raw:
                out_row = {"p_triples": gp, "h_triples": gh, "k_triples": gk,
                           "label": row["label"], "premise": row["premise"],
                           "hypothesis": row["hypothesis"]}
            else:
                out_row = {"text": serialize(gp, gh, gk), "label": row["label"],
                           "premise": row["premise"],
                           "hypothesis": row["hypothesis"]}
            fout.write(json.dumps(out_row) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"[dataset] {n} paires", flush=True)
    print(f"[dataset] {n} paires -> {out_path} "
          f"(P without graph: {missing_p}, H without graph: {missing_h})", flush=True)


if __name__ == "__main__":
    main()
