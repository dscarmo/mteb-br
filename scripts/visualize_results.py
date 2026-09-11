#!/usr/bin/env python
"""Visualize local MTEB(por, v2) results from ``$MTEB_CACHE/results``.

Discovers every model/revision under the cache (no hardcoded models), so new
runs from ``scripts/run_mteb_por_v2.py`` appear automatically.

Usage::

    streamlit run scripts/visualize_results.py

``MTEB_CACHE`` is read from the environment (default ``~/.cache/mteb``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import mteb_pt

_SKIP_FILES = {"model_meta.json"}
_TINYBERT_HINT = "TinyBERT"


def default_cache() -> Path:
    return Path(os.environ.get("MTEB_CACHE", "~/.cache/mteb")).expanduser()


def results_root(cache: Path) -> Path:
    """Return ``cache/results`` if it exists, else ``cache`` (upload-script fallback)."""
    nested = cache / "results"
    return nested if nested.is_dir() else cache


def task_category_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for category, tasks in mteb_pt.TASKS_BY_CATEGORY.items():
        for task in tasks:
            mapping[task] = category
    return mapping


def discover_models(root: Path) -> dict[str, list[str]]:
    """Map model slug -> sorted list of revision dirs that contain task JSONs."""
    found: dict[str, list[str]] = {}
    if not root.is_dir():
        return found
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        revisions: list[str] = []
        for rev_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if any(
                f.is_file() and f.suffix == ".json" and f.name not in _SKIP_FILES
                for f in rev_dir.iterdir()
            ):
                revisions.append(rev_dir.name)
        if revisions:
            found[model_dir.name] = revisions
    return found


def slug_to_display(slug: str) -> str:
    return slug.replace("__", "/")


def prefer_default_model(slugs: list[str]) -> str:
    for s in slugs:
        if _TINYBERT_HINT in s:
            return s
    return slugs[0]


def _main_score(payload: dict) -> float | None:
    scores = payload.get("scores") or {}
    for split in ("test", "validation", "dev", "train"):
        entries = scores.get(split)
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            val = entries[0].get("main_score")
            if val is not None:
                return float(val)
    for entries in scores.values():
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            val = entries[0].get("main_score")
            if val is not None:
                return float(val)
    return None


def load_run(root: Path, model_slug: str, revision: str) -> tuple[pd.DataFrame, dict]:
    """Load one model revision into a task dataframe + model_meta dict."""
    run_dir = root / model_slug / revision
    categories = task_category_map()
    rows: list[dict] = []
    meta: dict = {}
    meta_path = run_dir / "model_meta.json"
    if meta_path.is_file():
        with meta_path.open() as f:
            meta = json.load(f)

    for path in sorted(run_dir.glob("*.json")):
        if path.name in _SKIP_FILES:
            continue
        with path.open() as f:
            payload = json.load(f)
        task = payload.get("task_name") or path.stem
        rows.append(
            {
                "task": task,
                "category": categories.get(task, "other"),
                "main_score": _main_score(payload),
                "evaluation_time": payload.get("evaluation_time"),
                "is_headline": task in mteb_pt.HEADLINE_TASKS,
                "model_slug": model_slug,
                "model": meta.get("name") or slug_to_display(model_slug),
                "revision": revision,
            }
        )

    # Ensure every headline task appears (missing → NaN score)
    present = {r["task"] for r in rows}
    for task in mteb_pt.HEADLINE_TASKS:
        if task not in present:
            rows.append(
                {
                    "task": task,
                    "category": categories.get(task, "other"),
                    "main_score": None,
                    "evaluation_time": None,
                    "is_headline": True,
                    "model_slug": model_slug,
                    "model": meta.get("name") or slug_to_display(model_slug),
                    "revision": revision,
                }
            )

    order = {t: i for i, t in enumerate(mteb_pt.HEADLINE_TASKS)}
    df = pd.DataFrame(rows)
    df["_ord"] = df["task"].map(lambda t: order.get(t, 10_000 + hash(t) % 1000))
    df = df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return df, meta


def category_means(df: pd.DataFrame) -> pd.DataFrame:
    headline = df[df["is_headline"] & df["main_score"].notna()]
    if headline.empty:
        return pd.DataFrame(columns=["category", "mean_main_score", "n_tasks"])
    grouped = (
        headline.groupby("category", sort=False)["main_score"]
        .agg(mean_main_score="mean", n_tasks="count")
        .reset_index()
    )
    cat_order = list(mteb_pt.TASKS_BY_CATEGORY.keys())
    grouped["_ord"] = grouped["category"].map(
        lambda c: cat_order.index(c) if c in cat_order else len(cat_order)
    )
    return grouped.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def run_summary(df: pd.DataFrame) -> dict:
    headline = df[df["is_headline"]]
    done = headline["main_score"].notna().sum()
    total = len(mteb_pt.HEADLINE_TASKS)
    scored = headline[headline["main_score"].notna()]["main_score"]
    times = df["evaluation_time"].dropna()
    return {
        "done": int(done),
        "total": total,
        "mean_main_score": float(scored.mean()) if len(scored) else None,
        "total_eval_time": float(times.sum()) if len(times) else None,
    }


def _bar_horizontal(labels: list[str], values: list[float | None], title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    y = range(len(labels))
    numeric = [0.0 if v is None or pd.isna(v) else float(v) for v in values]
    colors = ["#4C78A8" if v is not None and not pd.isna(v) else "#CCCCCC" for v in values]
    ax.barh(list(y), numeric, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    return fig


def _grouped_compare(dfs: dict[str, pd.DataFrame]):
    """Grouped bar: tasks × models."""
    tasks = list(mteb_pt.HEADLINE_TASKS)
    models = list(dfs.keys())
    n_tasks, n_models = len(tasks), len(models)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * n_tasks)))
    height = 0.8 / max(n_models, 1)
    y = list(range(n_tasks))
    for i, model in enumerate(models):
        df = dfs[model]
        score_map = dict(zip(df["task"], df["main_score"], strict=False))
        vals = [score_map.get(t) for t in tasks]
        offsets = [yi - 0.4 + height / 2 + i * height for yi in y]
        numeric = [0.0 if v is None or pd.isna(v) else float(v) for v in vals]
        ax.barh(offsets, numeric, height=height * 0.9, label=model)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()
    ax.set_xlabel("main_score")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-task main_score comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="MTEB-BR results", layout="wide")
    st.title("MTEB-BR results visualizer")
    st.caption(
        "Reads eval JSONs from `$MTEB_CACHE/results`. New model runs appear automatically."
    )

    with st.sidebar:
        st.header("Cache")
        cache_str = st.text_input("MTEB_CACHE", value=str(default_cache()))
        cache = Path(cache_str).expanduser()
        root = results_root(cache)
        st.code(str(root), language=None)

        models = discover_models(root)
        if not models:
            st.error(f"No results found under {root}")
            st.stop()

        slugs = list(models.keys())
        default_slug = prefer_default_model(slugs)
        model_slug = st.selectbox(
            "Model",
            slugs,
            index=slugs.index(default_slug),
            format_func=slug_to_display,
        )
        revisions = models[model_slug]
        revision = st.selectbox("Revision", revisions, index=len(revisions) - 1)

        other_slugs = [s for s in slugs if s != model_slug]
        compare_slugs = st.multiselect(
            "Compare with",
            other_slugs,
            format_func=slug_to_display,
            help="Pick other models in this cache for a side-by-side chart.",
        )

    df, meta = load_run(root, model_slug, revision)
    summary = run_summary(df)
    display_name = meta.get("name") or slug_to_display(model_slug)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", display_name.split("/")[-1])
    c2.metric("Headline tasks", f"{summary['done']} / {summary['total']}")
    c3.metric(
        "Mean main_score",
        f"{summary['mean_main_score']:.4f}" if summary["mean_main_score"] is not None else "—",
    )
    if summary["total_eval_time"] is not None:
        mins = summary["total_eval_time"] / 60
        c4.metric("Total eval time", f"{mins:.1f} min")
    else:
        c4.metric("Total eval time", "—")

    if summary["done"] < summary["total"]:
        missing = df[df["is_headline"] & df["main_score"].isna()]["task"].tolist()
        st.warning(f"Incomplete run — missing: {', '.join(missing)}")

    with st.expander("model_meta", expanded=False):
        st.json(meta or {"name": display_name, "revision": revision})

    headline_df = df[df["is_headline"]].copy()
    st.subheader("Per-task main_score")
    fig = _bar_horizontal(
        headline_df["task"].tolist(),
        headline_df["main_score"].tolist(),
        title=display_name,
        xlabel="main_score",
    )
    st.pyplot(fig)
    plt.close(fig)

    cats = category_means(df)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Category averages")
        if cats.empty:
            st.info("No scored headline tasks yet.")
        else:
            fig = _bar_horizontal(
                cats["category"].tolist(),
                cats["mean_main_score"].tolist(),
                title="Mean main_score by category",
                xlabel="mean main_score",
            )
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(cats, use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Evaluation time")
        timed = headline_df[headline_df["evaluation_time"].notna()]
        if timed.empty:
            st.info("No timing data.")
        else:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(timed))))
            ax.barh(timed["task"], timed["evaluation_time"], color="#F58518")
            ax.invert_yaxis()
            ax.set_xlabel("seconds")
            ax.set_title("evaluation_time by task")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Scores table")
    show = df.copy()
    show["main_score"] = show["main_score"].map(
        lambda x: None if x is None or pd.isna(x) else round(float(x), 6)
    )
    st.dataframe(
        show[["task", "category", "main_score", "evaluation_time", "is_headline"]],
        use_container_width=True,
        hide_index=True,
    )

    if compare_slugs:
        st.subheader("Model comparison")
        compare_dfs: dict[str, pd.DataFrame] = {display_name: df}
        cat_rows: list[pd.DataFrame] = []
        primary_cats = category_means(df).assign(model=display_name)
        cat_rows.append(primary_cats)
        for slug in compare_slugs:
            rev = models[slug][-1]
            cdf, cmeta = load_run(root, slug, rev)
            label = cmeta.get("name") or slug_to_display(slug)
            compare_dfs[label] = cdf
            cat_rows.append(category_means(cdf).assign(model=label))
        fig = _grouped_compare(compare_dfs)
        st.pyplot(fig)
        plt.close(fig)
        cat_cmp = pd.concat(cat_rows, ignore_index=True)
        pivot = cat_cmp.pivot_table(
            index="category", columns="model", values="mean_main_score"
        )
        cat_order = list(mteb_pt.TASKS_BY_CATEGORY.keys())
        pivot = pivot.reindex([c for c in cat_order if c in pivot.index])
        st.dataframe(pivot, use_container_width=True)


if __name__ == "__main__":
    main()
