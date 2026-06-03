from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    MULTI_DOC_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    ASPECTS,
    ChunkSummary,
    DiscussionSummary,
    JudgeReport,
    MultiDocSummary,
    Review,
    ReviewSentiment,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = make_client()
MODEL = get_model()

PRICE_INPUT_PER_1M = 0.14
PRICE_OUTPUT_PER_1M = 0.28
JUDGE_THRESHOLD = 0.7
MR_WORKERS = 6
DOC_WORKERS = 4

REVIEW_SPLIT_RE = re.compile(r"^--- отзыв \d+ ---\s*$", re.MULTILINE)


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, completion) -> None:
        if completion.usage:
            self.input_tokens += completion.usage.prompt_tokens or 0
            self.output_tokens += completion.usage.completion_tokens or 0

    def cost_usd(self) -> float:
        return (self.input_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (
            self.output_tokens / 1_000_000
        ) * PRICE_OUTPUT_PER_1M


@dataclass
class PipelineStats:
    usage: UsageStats = field(default_factory=UsageStats)
    ghost_quotes: list[tuple[str, str]] = field(default_factory=list)
    validation_errors: int = 0
    reviews_extracted: int = 0
    elapsed_sec: float = 0.0


def split_by_review(corpus: str) -> list[str]:
    sep = corpus.find("═══")
    body = corpus[sep:] if sep != -1 else corpus
    parts = REVIEW_SPLIT_RE.split(body)
    chunks = [p.strip() for p in parts if p.strip() and len(p.strip()) > 40]
    if not chunks:
        chunks = [body.strip()]
    return chunks


def extract_reviews(corpus: str, stats: PipelineStats) -> list[Review]:
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=list[Review],
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": corpus},
        ],
    )
    stats.usage.add(completion)
    stats.reviews_extracted += len(result)
    return result


def extract_aspects(corpus: str, stats: PipelineStats) -> list[ReviewSentiment]:
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=list[ReviewSentiment],
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": corpus},
        ],
    )
    stats.usage.add(completion)
    return result


def check_quotes(
    aspects: list[ReviewSentiment],
    corpus: str,
) -> list[tuple[str, str]]:
    t = corpus.lower()
    ghosts: list[tuple[str, str]] = []
    for p in aspects:
        for a in p.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((p.name, a.quote))
    return ghosts


def build_heatmap(
    aspects: list[ReviewSentiment],
    out_path: str,
) -> None:
    names = [p.name for p in aspects]
    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(names), len(ASPECTS)), np.nan)
    for i, p in enumerate(aspects):
        for a in p.aspects:
            if a.aspect in ASPECTS:
                j = ASPECTS.index(a.aspect)
                matrix[i, j] = sent_to_num[a.sentiment]
    plt.figure(figsize=(8, max(4, len(names) * 0.3)))
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=ASPECTS,
        yticklabels=names,
        center=0,
    )
    plt.title("Аспектная тональность по отзывам")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_chunk(chunk: str, stats: PipelineStats) -> ChunkSummary:
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": chunk},
        ],
    )
    stats.usage.add(completion)
    return result


def reduce_summaries(
    summaries: list[ChunkSummary],
    stats: PipelineStats,
    reduce_prompt: str = REDUCE_SYSTEM,
) -> DiscussionSummary:
    joined = "\n\n".join(
        f"## {s.speaker} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": joined},
        ],
    )
    stats.usage.add(completion)
    return result


def summarize_discussion(
    corpus: str,
    stats: PipelineStats,
    reduce_prompt: str = REDUCE_SYSTEM,
    workers: int = MR_WORKERS,
) -> DiscussionSummary:
    chunks = split_by_review(corpus)
    n = len(chunks)
    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk, c, stats): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            summaries[futures[fut]] = fut.result()
    return reduce_summaries([s for s in summaries if s is not None], stats, reduce_prompt)


def build_evidence_packet(reviews: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации (которые оцениваем)"]
    for i, action in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {action}")
    parts.append("\n## Проблемы из отзывов (исходные данные)")
    for r in reviews:
        for issue in r.get("issues", []):
            parts.append(
                f"  - [{r.get('author', '?')}/{issue['category']}, "
                f"sev={issue['severity']}] «{issue['quote']}»"
            )
    return "\n".join(parts)


def judge(
    reviews: list[dict],
    summary: dict,
    stats: PipelineStats,
) -> JudgeReport:
    evidence = build_evidence_packet(reviews, summary)
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": evidence},
        ],
    )
    stats.usage.add(completion)
    return result


def fidelity(reviews: list[dict], corpus: str) -> float:
    t = corpus.lower()
    total, ok = 0, 0
    for r in reviews:
        for issue in r.get("issues", []):
            total += 1
            probe = issue["quote"].strip().lower()[:30]
            if probe and probe in t:
                ok += 1
    return ok / total if total else 0.0


def aggregate_aspects_rows(docs: list[dict]) -> pd.DataFrame:
    rows = []
    for d in docs:
        for p in d["aspects"]:
            for a in p.aspects:
                rows.append(
                    {
                        "app": d["app"],
                        "name": p.name,
                        "aspect": a.aspect,
                        "sentiment": a.sentiment,
                        "confidence": a.confidence,
                        "quote": a.quote,
                    }
                )
    return pd.DataFrame(rows)


def cross_app_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["app"], df["aspect"])


def consolidate(
    summaries: list[DiscussionSummary],
    apps: list[str],
    stats: PipelineStats,
) -> MultiDocSummary:
    joined = "\n\n".join(
        f"## {app}\nЗаголовок: {s.headline}\n"
        + "\n".join(f"- {kf}" for kf in s.key_findings)
        for app, s in zip(apps, summaries)
    )
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=MultiDocSummary,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": MULTI_DOC_SYSTEM},
            {"role": "user", "content": joined},
        ],
    )
    stats.usage.add(completion)
    return result


def process_document(path: Path, stats: PipelineStats) -> dict:
    corpus = path.read_text(encoding="utf-8")
    app = path.stem
    reviews = extract_reviews(corpus, stats)
    aspects = extract_aspects(corpus, stats)
    ghosts = check_quotes(aspects, corpus)
    stats.ghost_quotes.extend([(f"{app}/{n}", q) for n, q in ghosts])
    summary = summarize_discussion(corpus, stats)
    return {
        "app": app,
        "corpus": corpus,
        "reviews": reviews,
        "aspects": aspects,
        "summary": summary,
    }


def list_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    paths = sorted(input_path.glob("*.txt"))
    return paths


def analyze(input_path: str, out_dir: str) -> None:
    t0 = time.time()
    stats = PipelineStats()
    inp = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = list_inputs(inp)
    print(f"Total: {len(paths)}")

    docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=DOC_WORKERS) as pool:
        futures = {pool.submit(process_document, p, stats): p for p in paths}
        for fut in as_completed(futures):
            docs.append(fut.result())
            print(f"Ready: {fut.result()['app']}")

    all_reviews: list[dict] = []
    all_aspects: list[ReviewSentiment] = []
    for d in docs:
        for r in d["reviews"]:
            row = r.model_dump()
            row["app"] = d["app"]
            all_reviews.append(row)
        all_aspects.extend(d["aspects"])

    (out / "reviews.json").write_text(
        json.dumps(all_reviews, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aspects_data = [p.model_dump() for p in all_aspects]
    (out / "aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_heatmap(all_aspects, str(out / "heatmap.png"))

    df = aggregate_aspects_rows(docs)
    df.to_csv(out / "multi_doc.csv", index=False, encoding="utf-8")
    cross_app_table(df).to_csv(out / "cross_app.csv", encoding="utf-8")

    multi = consolidate([d["summary"] for d in docs], [d["app"] for d in docs], stats)
    (out / "multi_doc_summary.json").write_text(
        multi.model_dump_json(indent=2),
        encoding="utf-8",
    )

    full_corpus = "\n\n".join(d["corpus"] for d in docs)
    reduce_prompt = REDUCE_SYSTEM
    global_summary = summarize_discussion(full_corpus, stats, reduce_prompt)
    summary_payload = global_summary.model_dump()
    (out / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = judge(all_reviews, summary_payload, stats)
    if report.overall_score < JUDGE_THRESHOLD:
        print(f"  judge {report.overall_score} < {JUDGE_THRESHOLD}")
        global_summary = summarize_discussion(
            full_corpus, stats, REDUCE_SYSTEM_STRICT
        )
        summary_payload = global_summary.model_dump()
        (out / "summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report = judge(all_reviews, summary_payload, stats)

    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    ghost_pct = len(stats.ghost_quotes) / max(
        sum(len(p.aspects) for p in all_aspects), 1
    )
    metrics = {
        "fidelity": fidelity(all_reviews, full_corpus),
        "ghost_aspect_quotes": len(stats.ghost_quotes),
        "ghost_aspect_pct": ghost_pct,
        "reviews_count": len(all_reviews),
        "overall_score": report.overall_score,
        "input_tokens": stats.usage.input_tokens,
        "output_tokens": stats.usage.output_tokens,
        "cost_usd": stats.usage.cost_usd(),
        "elapsed_sec": time.time() - t0,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats.elapsed_sec = metrics["elapsed_sec"]

    print(f"Отзывов: {len(all_reviews)}")
    print(f"Ghost цитат: {len(stats.ghost_quotes)} ({ghost_pct})")
    print(f"Judge overall_score: {report.overall_score}")


def main() -> None:
    analyze("input", "output")


if __name__ == "__main__":
    main()
