"""Datasets as slot features: one pool per dataset, one stratified split per seed.

A pool is a float32 matrix whose columns are grouped into named slots, a 0/1
label, and a flag per column saying whether it is standardised on the training
split. Text slots are embedded with ``sentence-transformers/all-MiniLM-L6-v2``
(384 dimensions, unit norm), the setting TRL's changelog benchmarks
("6 slots of 384 features"). Tabular slots carry a constant 1 so that a revealed
slot is never the zero vector TRL uses for "not revealed".

Build every pool once with ``python -m grpo_trl.data``; pools are cached in
``data/cache`` and keyed by the SHA-256 of their raw files (``data/raw/SHA256SUMS``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent.parent
RAW, CACHE = ROOT / "data" / "raw", ROOT / "data" / "cache"
POOL_SIZE = 12_000
POOL_SEED = 20261122
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DATASETS = ("synth-route", "founders", "kickstarter", "fakejobs", "credit", "adult")
SPLITS = {"default": (2000, 500, 2000), "founders": (1800, 400, 800)}

Floats = npt.NDArray[np.float32]


@dataclass
class Pool:
    """A dataset as slot features."""

    name: str
    x: Floats
    y: npt.NDArray[np.int64]
    slots: dict[str, int]
    standardise: npt.NDArray[np.bool_]  # per column

    def save(self) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE / f"{self.name}.npz",
            x=self.x.astype(np.float32),
            y=self.y,
            standardise=self.standardise,
            slots=json.dumps(self.slots),
        )


def load_pool(name: str) -> Pool:
    """A cached pool; build it first with ``python -m grpo_trl.data``."""
    stored = np.load(CACHE / f"{name}.npz", allow_pickle=False)
    return Pool(
        name,
        stored["x"],
        stored["y"],
        json.loads(str(stored["slots"])),
        stored["standardise"],
    )


def split_pool(
    pool: Pool, seed: int
) -> tuple[tuple[Floats, Floats], tuple[Floats, Floats], tuple[Floats, Floats]]:
    """A stratified train/validation/test split for ``seed``, standardised on train."""
    n_train, n_val, n_test = SPLITS.get(pool.name, SPLITS["default"])
    rng = np.random.default_rng([seed, 517])
    parts: list[list[int]] = [[], [], []]
    total = n_train + n_val + n_test
    for label in (0, 1):
        members = rng.permutation(np.flatnonzero(pool.y == label))
        share = len(members) / len(pool.y)
        counts = [round(n * share) for n in (n_train, n_val, n_test)]
        if sum(counts) > len(members):
            raise ValueError(f"{pool.name}: pool too small for a split of {total}")
        start = 0
        for part, count in zip(parts, counts, strict=True):
            part.extend(members[start : start + count].tolist())
            start += count
    train, val, test = (np.array(rng.permutation(p)) for p in parts)
    x = pool.x.copy()
    columns = pool.standardise
    mean = x[train][:, columns].mean(axis=0)
    std = x[train][:, columns].std(axis=0)
    x[:, columns] = (x[:, columns] - mean) / np.where(std > 1e-8, std, 1.0)
    y = pool.y.astype(np.float32)
    return (x[train], y[train]), (x[val], y[val]), (x[test], y[test])


# --------------------------------------------------------------------------- builders


def _stratified_pool(labels: npt.NDArray[np.int64], size: int) -> npt.NDArray[np.intp]:
    """A stratified sample of row indices of at most ``size`` rows."""
    if len(labels) <= size:
        return np.arange(len(labels))
    rng = np.random.default_rng(POOL_SEED)
    keep: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        count = round(size * len(members) / len(labels))
        keep.extend(rng.choice(members, size=count, replace=False).tolist())
    return np.sort(np.array(keep))


def _embed(texts: list[str]) -> Floats:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER, device="cpu")
    model.max_seq_length = 128
    vectors = model.encode(
        texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)


def _one_hot(values: list[str], keep: int) -> Floats:
    """One column per frequent value (at most ``keep``), plus one for the rest."""
    names, counts = np.unique(values, return_counts=True)
    frequent = [str(n) for n in names[np.argsort(-counts)][:keep]]
    index = {name: i for i, name in enumerate(frequent)}
    out = np.zeros((len(values), len(frequent) + 1), dtype=np.float32)
    for row, value in enumerate(values):
        out[row, index.get(value, len(frequent))] = 1.0
    return out


def _tabular(*blocks: tuple[Floats, bool]) -> tuple[Floats, npt.NDArray[np.bool_]]:
    """A tabular slot: its blocks side by side, then the constant 1."""
    parts = [np.asarray(b, dtype=np.float32).reshape(len(b), -1) for b, _ in blocks]
    flags = [np.full(p.shape[1], scale) for p, (_, scale) in zip(parts, blocks, strict=True)]
    n = len(parts[0])
    return (
        np.hstack([*parts, np.ones((n, 1), dtype=np.float32)]),
        np.concatenate([*flags, [False]]),
    )


def _assemble(
    name: str,
    slots: dict[str, tuple[Floats, npt.NDArray[np.bool_]]],
    y: npt.NDArray[np.int64],
) -> Pool:
    return Pool(
        name,
        np.hstack([x for x, _ in slots.values()]).astype(np.float32),
        y.astype(np.int64),
        {slot: int(x.shape[1]) for slot, (x, _) in slots.items()},
        np.concatenate([flags for _, flags in slots.values()]),
    )


def _text(vectors: Floats) -> tuple[Floats, npt.NDArray[np.bool_]]:
    return vectors, np.zeros(vectors.shape[1], dtype=bool)


def build_synth_route(n: int = POOL_SIZE) -> Pool:
    """A problem whose best strategy is adaptive, and known.

    The label is Bernoulli(0.3). A router slot names one of four candidate
    slots; only that candidate carries the label (a shift of +-1.5 along a fixed
    direction, in unit Gaussian noise), the other three are pure noise. A sixth
    slot always carries a weak signal (shift +-0.5). The best strategy reads the
    router, then the candidate it names, then stops; any fixed set of slots
    must buy all four candidates to do as well.
    """
    rng = np.random.default_rng(POOL_SEED)
    width, n_candidates, shift, weak = 8, 4, 1.5, 0.5
    y = (rng.random(n) < 0.3).astype(np.int64)
    sign = (2 * y - 1).astype(np.float32)
    route = rng.integers(n_candidates, size=n)
    directions = rng.normal(size=(n_candidates + 1, width))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    router = rng.normal(scale=0.3, size=(n, width))
    router[np.arange(n), route] += 2.0
    slots: dict[str, tuple[Floats, npt.NDArray[np.bool_]]] = {
        "router": _text(router.astype(np.float32))
    }
    for k in range(n_candidates):
        noise = rng.normal(size=(n, width))
        signal = np.where(route == k, sign * shift, 0.0)[:, None] * directions[k]
        slots[f"candidate_{k + 1}"] = _text((noise + signal).astype(np.float32))
    noise = rng.normal(size=(n, width))
    slots["weak"] = _text((noise + sign[:, None] * weak * directions[-1]).astype(np.float32))
    return _assemble("synth-route", slots, y)


FOUNDER_SLOTS = ("company", "background", "education", "jobs", "prior_startups", "prior_exits")


def founder_slot_texts(profile: str) -> dict[str, str]:
    """A founder profile (the trl-for-angels template) cut into its six parts."""
    lines = [line.strip() for line in profile.strip().split("\n") if line.strip()]
    parts: dict[str, list[str]] = {slot: [] for slot in FOUNDER_SLOTS}
    current = "company"
    for i, line in enumerate(lines):
        lower = line.lower()
        if i == 0:
            current = "company"
        elif lower.startswith("founder background"):
            current = "background"
        elif lower.startswith("education"):
            current = "education"
        elif lower.startswith("jobs"):
            current = "jobs"
        elif lower.startswith("prior startups"):
            current = "prior_startups"
        elif lower.startswith("prior exits"):
            current = "prior_exits"
        parts[current].append(line)
    return {slot: " ".join(text) or f"{slot}: unknown" for slot, text in parts.items()}


def build_founders() -> Pool:
    """The 3,000 generated founder profiles of ``Sidi3355/trl-for-angels``."""
    rows = [json.loads(line) for line in (RAW / "founders.jsonl").open()]
    outcome = [re.sub(r"\s+", " ", r["output"].split("\n")[0]).lower() for r in rows]
    y = np.array([int(("$500m-$1b" in o) or ("over $1b" in o)) for o in outcome], dtype=np.int64)
    texts = [founder_slot_texts(r["input"]) for r in rows]
    slots = {slot: _text(_embed([t[slot] for t in texts])) for slot in FOUNDER_SLOTS}
    return _assemble("founders", slots, y)


def build_kickstarter() -> Pool:
    """Kickstarter projects (HF ``james-burton/kick_starter_funding``): funded or not."""
    import pandas as pd

    frame = pd.read_parquet(RAW / "kickstarter_train.parquet")
    frame = frame.iloc[_stratified_pool(frame["final_status"].to_numpy(), POOL_SIZE)]
    created = pd.to_datetime(frame["created_at"], unit="s")
    days = (frame["deadline"] - frame["created_at"]).to_numpy() / 86400.0
    month = created.dt.month.to_numpy() * (2 * np.pi / 12)
    goal = frame["goal"].to_numpy(dtype=float)
    slots = {
        "name": _text(_embed(frame["name"].fillna("").astype(str).tolist())),
        "description": _text(_embed(frame["desc"].fillna("").astype(str).tolist())),
        "keywords": _text(
            _embed(frame["keywords"].fillna("").astype(str).str.replace("-", " ").tolist())
        ),
        "goal": _tabular((np.log10(goal + 1.0), True), ((goal % 1000 == 0).astype(float), False)),
        "geography": _tabular(
            (_one_hot(frame["country"].astype(str).tolist(), 8), False),
            (frame["disable_communication"].to_numpy(dtype=float), False),
        ),
        "timing": _tabular(
            (np.log1p(np.clip(days, 0, None)), True),
            ((created.dt.year.to_numpy() - 2009).astype(float), True),
            (np.sin(month), False),
            (np.cos(month), False),
        ),
    }
    return _assemble("kickstarter", slots, frame["final_status"].to_numpy())


def _salary(ranges: list[object]) -> Floats:
    out = np.zeros((len(ranges), 3), dtype=np.float32)
    for row, value in enumerate(ranges):
        numbers = re.findall(r"\d+", value) if isinstance(value, str) else []
        if len(numbers) >= 2:
            out[row] = [1.0, np.log1p(float(numbers[0])), np.log1p(float(numbers[1]))]
    return out


def build_fakejobs() -> Pool:
    """Job postings (HF ``james-burton/fake_job_postings2``): fraudulent or not."""
    import pandas as pd

    frame = pd.concat(
        [pd.read_parquet(RAW / f"fakejobs_{part}.parquet") for part in ("train", "val", "test")],
        ignore_index=True,
    )
    frame = frame.iloc[_stratified_pool(frame["fraudulent"].to_numpy(), POOL_SIZE)]
    description = frame["description"].fillna("").astype(str).str.slice(0, 1000).tolist()
    slots = {
        "title": _text(_embed(frame["title"].fillna("").astype(str).tolist())),
        "description": _text(_embed(description)),
        "salary": _tabular((_salary(frame["salary_range"].tolist()), False)),
        "experience": _tabular(
            (_one_hot(frame["required_experience"].fillna("missing").tolist(), 7), False)
        ),
        "education": _tabular(
            (_one_hot(frame["required_education"].fillna("missing").tolist(), 8), False)
        ),
    }
    return _assemble("fakejobs", slots, frame["fraudulent"].to_numpy())


def build_credit() -> Pool:
    """UCI Taiwan credit (HF ``Lancer73/uci-credit-card-default``): default next month."""
    import pandas as pd

    frame = pd.read_csv(RAW / "credit.csv")
    label = "default.payment.next.month"
    frame = frame.iloc[_stratified_pool(frame[label].to_numpy(), POOL_SIZE)]
    slots = {
        "profile": _tabular(
            (np.log10(frame["LIMIT_BAL"].to_numpy(dtype=float)), True),
            ((frame["SEX"].to_numpy() == 2).astype(float), False),
            (_one_hot(frame["EDUCATION"].astype(str).tolist(), 4), False),
            (_one_hot(frame["MARRIAGE"].astype(str).tolist(), 3), False),
            (frame["AGE"].to_numpy(dtype=float), True),
        )
    }
    for month, status in enumerate(("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"), 1):
        bill = frame[f"BILL_AMT{month}"].to_numpy(dtype=float)
        paid = frame[f"PAY_AMT{month}"].to_numpy(dtype=float)
        slots[f"month_{month}"] = _tabular(
            (frame[status].to_numpy(dtype=float), True),
            (np.sign(bill) * np.log1p(np.abs(bill)), True),
            (np.log1p(paid), True),
        )
    return _assemble("credit", slots, frame[label].to_numpy())


def build_adult() -> Pool:
    """Census income (HF ``scikit-learn/adult-census-income``): over $50K or not."""
    import pandas as pd

    frame = pd.read_csv(RAW / "adult.csv")
    y = (frame["income"].str.strip() == ">50K").to_numpy().astype(np.int64)
    keep = _stratified_pool(y, POOL_SIZE)
    frame, y = frame.iloc[keep], y[keep]

    def hot(column: str, top: int) -> tuple[Floats, bool]:
        return _one_hot(frame[column].astype(str).str.strip().tolist(), top), False

    slots = {
        "demographics": _tabular(
            (frame["age"].to_numpy(dtype=float), True),
            ((frame["sex"].str.strip() == "Male").to_numpy(dtype=float), False),
            hot("race", 4),
            ((frame["native.country"].str.strip() == "United-States").to_numpy(dtype=float), False),
        ),
        "education": _tabular((frame["education.num"].to_numpy(dtype=float), True)),
        "work": _tabular(
            hot("workclass", 8),
            hot("occupation", 14),
            (frame["hours.per.week"].to_numpy(dtype=float), True),
        ),
        "family": _tabular(hot("marital.status", 6), hot("relationship", 5)),
        "capital": _tabular(
            (np.log1p(frame["capital.gain"].to_numpy(dtype=float)), True),
            (np.log1p(frame["capital.loss"].to_numpy(dtype=float)), True),
        ),
    }
    return _assemble("adult", slots, y)


BUILDERS = {
    "synth-route": build_synth_route,
    "founders": build_founders,
    "kickstarter": build_kickstarter,
    "fakejobs": build_fakejobs,
    "credit": build_credit,
    "adult": build_adult,
}


def main() -> None:
    """Build and cache every pool that is not cached yet."""
    import sys
    import time

    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        if (CACHE / f"{name}.npz").exists():
            print(f"{name}: cached", flush=True)
            continue
        start = time.time()
        pool = BUILDERS[name]()
        pool.save()
        print(
            f"{name}: {pool.x.shape[0]} rows, {pool.x.shape[1]} columns, slots {pool.slots}, "
            f"positive rate {pool.y.mean():.3f}, {time.time() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
