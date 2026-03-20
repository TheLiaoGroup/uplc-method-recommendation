"""RDKit fingerprint helpers for similarity analysis."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

Fingerprint = Optional[DataStructs.cDataStructs.ExplicitBitVect]


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Fingerprint:
    """Convert SMILES into a Morgan fingerprint bit vector."""
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_similarity(fp1: Fingerprint, fp2: Fingerprint) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def _process_row(args: Tuple[str, str, str, Dict[str, Fingerprint], Dict[str, Fingerprint], Dict[str, Fingerprint]]):
    p_smiles, s1_smiles, s2_smiles, p_cache, s1_cache, s2_cache = args
    p_fp = p_cache.get(p_smiles)
    s1_fp = s1_cache.get(s1_smiles)
    s2_fp = s2_cache.get(s2_smiles)

    sim_s1_p = compute_similarity(p_fp, s1_fp)
    sim_s2_p = compute_similarity(p_fp, s2_fp)

    if max(sim_s1_p, sim_s2_p) == 0:
        return sim_s1_p, sim_s2_p, 0.0, "None"

    if sim_s1_p >= sim_s2_p:
        return sim_s1_p, sim_s2_p, sim_s1_p, "S1"
    return sim_s1_p, sim_s2_p, sim_s2_p, "S2"


def build_fingerprint_cache(df: pd.DataFrame, columns: Sequence[str], max_cores: int) -> Dict[str, Fingerprint]:
    unique_smiles: List[str] = []
    seen = set()
    for column in columns:
        for smiles in df[column].dropna().unique():
            if smiles not in seen:
                seen.add(smiles)
                unique_smiles.append(smiles)

    if not unique_smiles:
        return {}

    worker_count = min(max_cores, cpu_count())
    with Pool(worker_count) as pool:
        fps = pool.map(smiles_to_fp, unique_smiles)
    return dict(zip(unique_smiles, fps))


def split_column_caches(df: pd.DataFrame, cache: Dict[str, Fingerprint]) -> Tuple[Dict[str, Fingerprint], Dict[str, Fingerprint], Dict[str, Fingerprint]]:
    p_cache = {smiles: cache.get(smiles) for smiles in df["P"].dropna().unique() if smiles in cache}
    s1_cache = {smiles: cache.get(smiles) for smiles in df["S1"].dropna().unique() if smiles in cache}
    s2_cache = {smiles: cache.get(smiles) for smiles in df["S2"].dropna().unique() if smiles in cache}
    return p_cache, s1_cache, s2_cache


def compute_similarities(
    df: pd.DataFrame,
    p_cache: Dict[str, Fingerprint],
    s1_cache: Dict[str, Fingerprint],
    s2_cache: Dict[str, Fingerprint],
    max_cores: int,
) -> Tuple[List[float], List[float], List[float], List[str]]:
    args: List[Tuple[str, str, str, Dict[str, Fingerprint], Dict[str, Fingerprint], Dict[str, Fingerprint]]] = []
    for _, row in df.iterrows():
        args.append((row["P"], row["S1"], row["S2"], p_cache, s1_cache, s2_cache))

    if not args:
        return [], [], [], []

    worker_count = min(max_cores, cpu_count())
    with Pool(worker_count) as pool:
        results = pool.map(_process_row, args)

    sim_s1 = [res[0] for res in results]
    sim_s2 = [res[1] for res in results]
    max_sim = [res[2] for res in results]
    predicted_by = [res[3] for res in results]
    return sim_s1, sim_s2, max_sim, predicted_by


__all__ = [
    "build_fingerprint_cache",
    "compute_similarities",
    "compute_similarity",
    "split_column_caches",
    "smiles_to_fp",
]
