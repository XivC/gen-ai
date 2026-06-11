import json

from pipeline import ingest, retrieve

GOLD_PATH = "data/gold.json"
RESULTS_PATH = "eval_results.json"
K = 5


def load_gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def hit_rate(retrieved_ids, gold_sources):
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run():
    gold = load_gold()
    total = 0.0
    results = []

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]
        hits = retrieve(q, k=K)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]
        score = hit_rate(retrieved_ids, gold_sources)
        total += score
        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": q,
                "score": score,
                "gold": gold_sources,
                "retrieved_ids": retrieved_ids,
                "retrieved_sources": retrieved_sources,
            }
        )
        mark = "OK" if score == 1.0 else ("PART" if score > 0 else "MISS")
        print(
            f"  [{item['id']:2d}] {item['type']}  "
            f"hit@{K} = {score}  {mark} {q}"
        )

    mean = total / len(gold)
    print(f"\n  ИТОГО: hit-rate@{K} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return {"mean": mean, "results": results}


def main():
    summary = {}
    for strategy in ("fixed", "recursive"):
        print(f"Strategy: {strategy}")
        ingest(strategy)
        out = run()
        summary[strategy] = {"mean": out["mean"], "results": out["results"]}
    for strategy, data in summary.items():
        print(f"  {strategy:10s}  hit-rate@{K} = {data['mean']:.2f}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
