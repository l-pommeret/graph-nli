from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VOCAB = json.loads((ROOT / "data" / "relations.json").read_text())
RELATIONS: dict[str, str] = _VOCAB["extraction"]
REL_LIST = list(RELATIONS)
KGRAPH_RELATIONS = REL_LIST + _VOCAB["kgraph_extra"]

MODEL = "Qwen/Qwen3.5-9B"
MODEL_AWQ = "QuantTrio/Qwen3.5-9B-AWQ"



SYSTEM = """You convert ONE English sentence (an atomic proposition) into knowledge-graph \
triples using the ConceptNet vocabulary. Output a JSON array of triples \
[{"h": head, "r": relation, "t": tail}, ...] and nothing else.

RELATIONS — use ONLY these ConceptNet relations:
""" + "\n".join(f"- {r}: {d}" for r, d in RELATIONS.items()) + """

NODE RULES:
- a node is a lowercase English lemma: the bare HEAD word, without its modifiers
- NEVER glue modifier+noun or verb+object into one node (wrong: old_man, go_package, \
wear_jacket); each modifier becomes its own HasProperty triple on the head noun
- multi-word nodes ONLY for proper names (new_york, forza_italia) and true lexical \
compounds (winter_hat, hot_dog); exception: non-intersective adjectives stay attached \
(former_senator, fake_gun)
- verbs: bare lemma (jump, not jumps/jumping); phrasal verbs keep their particle (jump_over)
- nouns keep their lexical form (linguistics stays linguistics); never use bare adverbs \
or prepositions as nodes (outside, just, very)
- numerals ONLY if the number is explicit: "two men" -> [man, HasProperty, two]; a bare \
plural ("the sisters") NEVER yields a numeral
- worn or printed identifiers are not quantities: "the number 2 jersey" -> \
[jersey, HasProperty, number_2]

STRUCTURE — every content word of the sentence must land in at least one triple:
- agent of a verb: [agent, CapableOf, verb] — the agent ALWAYS takes CapableOf, never \
HasProperty and never ReceivesAction
- direct object: [object, ReceivesAction, verb] — for every transitive verb emit BOTH \
the agent triple AND the patient triple
- passive voice: the subject is the patient ("the book was written" -> \
[book, ReceivesAction, write]); a "by X" agent gets [X, CapableOf, verb]
- recipient or beneficiary ("to/for X"): [verb, HasContext, X] — NEVER ReceivesAction
- destination ("walk to school"): [verb, MotivatedByGoal, school] — not AtLocation
- place where it happens: [x, AtLocation, place]; time, date or period: \
[x, HasContext, time] — dates are context, never HasProperty
- instrument: [instrument, UsedFor, verb] — purpose: [verb, MotivatedByGoal, goal]
- possession: [owner, HasA, thing] — part or aspect of: [part, PartOf, whole]
- role or profession: [person, IsA, role]; named entity class: [name, InstanceOf, class]
- created works: [work, CreatedBy, creator] — the WORK is always the head, in active \
voice too: "Panini wrote a grammar" -> [grammar, CreatedBy, panini], NEVER \
[panini, CreatedBy, grammar]
- "outside/near X": attach the real place directly ([fight, LocatedNear, deli]), never \
a bare "outside" node
- NEGATION: use NotCapableOf / NotHasProperty / NotDesires, with the full verb phrase as \
tail when needed ([politician, NotCapableOf, attend_meeting]); NEVER create not_* nodes.
Produce 2 to 8 triples."""

FEW_SHOTS: list[tuple[str, list[dict]]] = [
    ("A person is training his horse for a competition.", [
        {"h": "person", "r": "CapableOf", "t": "train"},
        {"h": "horse", "r": "ReceivesAction", "t": "train"},
        {"h": "person", "r": "HasA", "t": "horse"},
        {"h": "train", "r": "MotivatedByGoal", "t": "competition"},
    ]),
    ("Two young girls are playing soccer in the park.", [
        {"h": "girl", "r": "HasProperty", "t": "two"},
        {"h": "girl", "r": "HasProperty", "t": "young"},
        {"h": "girl", "r": "CapableOf", "t": "play"},
        {"h": "soccer", "r": "ReceivesAction", "t": "play"},
        {"h": "play", "r": "AtLocation", "t": "park"},
    ]),
    ("The chef is cutting fresh bread with a steel knife.", [
        {"h": "chef", "r": "CapableOf", "t": "cut"},
        {"h": "bread", "r": "ReceivesAction", "t": "cut"},
        {"h": "bread", "r": "HasProperty", "t": "fresh"},
        {"h": "knife", "r": "UsedFor", "t": "cut"},
        {"h": "knife", "r": "MadeOf", "t": "steel"},
    ]),
    ("The politicians did not attend the meeting.", [
        {"h": "politician", "r": "NotCapableOf", "t": "attend_meeting"},
    ]),
    ("The sisters are selling lemonade to a customer.", [
        {"h": "sister", "r": "CapableOf", "t": "sell"},
        {"h": "lemonade", "r": "ReceivesAction", "t": "sell"},
        {"h": "sell", "r": "HasContext", "t": "customer"},
    ]),
    ("A boy wearing a jersey with the number 2 walks to school.", [
        {"h": "boy", "r": "CapableOf", "t": "wear"},
        {"h": "jersey", "r": "ReceivesAction", "t": "wear"},
        {"h": "jersey", "r": "HasProperty", "t": "number_2"},
        {"h": "boy", "r": "CapableOf", "t": "walk"},
        {"h": "walk", "r": "MotivatedByGoal", "t": "school"},
    ]),
    ("The Indian grammarian Panini wrote the Ashtadhyayi, a description of "
     "Sanskrit, in the 4th century BCE.", [
        {"h": "panini", "r": "IsA", "t": "grammarian"},
        {"h": "panini", "r": "HasProperty", "t": "indian"},
        {"h": "panini", "r": "CapableOf", "t": "write"},
        {"h": "ashtadhyayi", "r": "ReceivesAction", "t": "write"},
        {"h": "ashtadhyayi", "r": "CreatedBy", "t": "panini"},
        {"h": "ashtadhyayi", "r": "DefinedAs", "t": "description"},
        {"h": "description", "r": "HasContext", "t": "sanskrit"},
        {"h": "write", "r": "HasContext", "t": "4th_century_bce"},
    ]),
    ("The Parma trolleybus system has been in operation since 1953.", [
        {"h": "parma_trolleybus_system", "r": "InstanceOf", "t": "trolleybus_system"},
        {"h": "parma_trolleybus_system", "r": "AtLocation", "t": "parma"},
        {"h": "parma_trolleybus_system", "r": "ReceivesAction", "t": "operate"},
        {"h": "operate", "r": "HasContext", "t": "since_1953"},
    ]),
    ("A man sleeps on the couch because he is tired.", [
        {"h": "man", "r": "CapableOf", "t": "sleep"},
        {"h": "sleep", "r": "AtLocation", "t": "couch"},
        {"h": "man", "r": "HasProperty", "t": "tired"},
        {"h": "tired", "r": "Causes", "t": "sleep"},
    ]),
]

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "h": {"type": "string", "maxLength": 60},
            "r": {"enum": REL_LIST},
            "t": {"type": "string", "maxLength": 60},
        },
        "required": ["h", "r", "t"],
        "additionalProperties": False,
    },
    "minItems": 1,
    "maxItems": 8,
}


def messages_for(prop: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for text, triples in FEW_SHOTS:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": json.dumps(triples)})
    msgs.append({"role": "user", "content": prop.strip()})
    return msgs


_DROP_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PUNCT", "PART", "AUX", "PRON"}
_KEEP = {"not", "no", "never"}
_nlp = None


def _spacy():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def normalize_node(node: str) -> str:
    text = node.replace("_", " ").strip().lower()
    if not text:
        return ""
    parts = [t.lower_ if t.lower_.endswith("ics") else t.lemma_.lower()
             for t in _spacy()(text)
             if t.pos_ not in _DROP_POS or t.lower_ in _KEEP]
    return "_".join(p for p in parts if p) or text.replace(" ", "_")


def clean_triples(raw: list[dict]) -> list[list[str]]:
    seen, out = set(), []
    for tr in raw:
        h, r, t = normalize_node(tr["h"]), tr["r"], normalize_node(tr["t"])
        if h and t and r in RELATIONS and h != t and (h, r, t) not in seen:
            seen.add((h, r, t))
            out.append([h, r, t])
    return out


def load_llm(model: str, util: float = 0.90, eager: bool = False,
             max_model_len: int = 4096):
    from vllm import LLM
    return LLM(model=model, dtype="auto", max_model_len=max_model_len,
               gpu_memory_utilization=util, enable_prefix_caching=True,
               enforce_eager=eager, max_num_seqs=128)


def sampling_params():
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0.0, max_tokens=512,
                              structured_outputs=StructuredOutputsParams(
                                  json=SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0.0, max_tokens=512,
                              guided_decoding=GuidedDecodingParams(json=SCHEMA))


def collect_props(paths: list[str]) -> list[str]:
    props, seen = [], set()

    def add(p: str):
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            props.append(p)

    for path in paths:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                for key in ("premise_propositions", "hypothesis_propositions", "props"):
                    for p in row.get(key) or []:
                        add(p)
    return props


def main() -> None:
    ap = argparse.ArgumentParser(description='atomic propositions -> ConceptNet triples')
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--awq", action="store_true", help="Qwen3.5-9B-AWQ (tight VRAM)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first n propositions (smoke test)")
    ap.add_argument("--shard", default=None, help="i/n: process shard i only (multi-GPU)")
    ap.add_argument("--done-from", nargs="*", default=[],
                    help="extra jsonl counted as already extracted (sharding)")
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    model = args.model or (MODEL_AWQ if args.awq else MODEL)

    props = collect_props(args.inputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in [out_path] + [Path(p) for p in args.done_from]:
        if path.exists():
            with open(path) as fh:
                done |= {json.loads(l)["text"] for l in fh}
    pool = props[: args.limit]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        pool = pool[i::n]
    todo = [p for p in pool if p not in done]
    print(f"[extract] {len(props)} unique propositions, {len(done)} already done, "
          f"{len(todo)} to extract" + (f" (shard {args.shard})" if args.shard else ""),
          flush=True)
    if not todo:
        return


    print(f"[extract] model {model}", flush=True)
    llm = load_llm(model)
    params = sampling_params()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        outs = llm.chat([messages_for(p) for p in chunk], params,
                        chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as fh:
            for prop, o in zip(chunk, outs):
                try:
                    triples = clean_triples(json.loads(o.outputs[0].text))
                except (json.JSONDecodeError, KeyError, TypeError):
                    triples = []
                fh.write(json.dumps({"text": prop, "triples": triples}) + "\n")
        print(f"[extract] {min(i + args.chunk, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
