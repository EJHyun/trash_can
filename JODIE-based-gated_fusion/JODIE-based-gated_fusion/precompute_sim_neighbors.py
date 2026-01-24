# -*- coding: utf-8 -*-
"""Precompute similarity-based neighbors (U-U and I-I).

What it does
- Loads the dataset with `load_network(args)` from library_data.py to ensure that
  user/item indexing and interaction ordering are identical to training.
- Takes the training prefix (0 .. train_end_idx-1).
- Uses a fixed-size sliding window over interactions (row-based window).
- Builds U-U (user-user) neighbors based on weighted Jaccard over user->item
  counters within the window.
- Builds I-I (item-item) neighbors based on weighted Jaccard over item->user
  counters within the window.
- Selects neighbors using: top-k + (optional) threshold expansion.
- Saves per-interaction neighbor id lists for the *current* (u,i) at that
  interaction index.

Outputs
- A compressed .npz with two object arrays:
    uu_neighbors[idx] = np.array([neighbor_user_ids...], dtype=np.int32)
    ii_neighbors[idx] = np.array([neighbor_item_ids...], dtype=np.int32)
  for idx in [window_size-1, train_end_idx-1]. Indices before window is full are
  stored as empty arrays.

Notes
- This script *includes* the current interaction in the window, matching your
  stated design.
- Neighbor embeddings should be gathered from the *pre-update* snapshot during
  training (you already plan to do this) to avoid same-event circularity.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict, deque
from typing import DefaultDict, Dict, List, Set, Tuple, Optional

import numpy as np
from tqdm import tqdm
import copy

from library_data import load_network


def _weighted_jaccard(a: Counter, b: Counter) -> float:
    """Weighted Jaccard for multisets represented as Counters.

    sim(a,b) = sum_f min(a_f, b_f) / sum_f max(a_f, b_f)

    Here, union = sum(a)+sum(b)-intersection.
    """
    if not a or not b:
        return 0.0

    # Iterate over the smaller counter for speed
    if len(a) <= len(b):
        small, other = a, b
    else:
        small, other = b, a

    inter = 0
    for k, c in small.items():
        oc = other.get(k, 0)
        if oc:
            inter += min(c, oc)

    union = sum(a.values()) + sum(b.values()) - inter
    return (inter / union) if union > 0 else 0.0


def _select_pos_and_neg(
    node_id: int,
    node_to_featcnt: Dict[int, Counter], # user → Counter(item->count)
    feat_to_nodecnt: Dict[int, Counter], # item → Counter(user->count)
    node_deg: Dict[int, int],
    all_active_nodes: Set[int],
    top_k: int,
    bottom_k: int,
    threshold: float,
    min_deg: int,
    include_threshold: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pos_ids, pos_sims, neg_ids, neg_sims
    """

    if rng is None:
        rng = random

    a_cnt = node_to_featcnt.get(node_id)

    # if not a_cnt or node_deg.get(node_id, 0) < min_deg:
    if not a_cnt:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_f, empty_i, empty_f

    # candidate generation (inverted index)
    pos_candidates: Set[int] = set()
    for feat in a_cnt.keys(): # features of node_id
        for other in feat_to_nodecnt.get(feat, {}).keys():
            if other != node_id:
                pos_candidates.add(other)

    if not pos_candidates: # no candidates
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_f, empty_i, empty_f

    sims: List[Tuple[int, float]] = []
    for other in pos_candidates:
        b_cnt = node_to_featcnt.get(other)
        if not b_cnt:
            continue
        s = _weighted_jaccard(a_cnt, b_cnt)
        sims.append((other, s))

    if not sims:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_f, empty_i, empty_f

    rng.shuffle(sims) # for stability when sim scores are tied
    sims.sort(key=lambda x: -x[1])

    pos = sims[:top_k]
    if include_threshold and threshold is not None:
        for n, s in sims[top_k:]:
            if s >= threshold:
                pos.append((n, s))
            else:
                break

    pos_ids_set = {n for n, _ in pos}

    # ---------- negatives (bottom-k with sim=0 priority) ----------
    zero_candidates = [
        n for n in all_active_nodes
        if n != node_id
        and n not in pos_candidates
        and n not in pos_ids_set
    ]

    neg: List[Tuple[int, float]] = []

    if len(zero_candidates) >= bottom_k:
        sampled = rng.sample(zero_candidates, bottom_k)
        neg = [(n, 0.0) for n in sampled]
    else:
        neg = [(n, 0.0) for n in zero_candidates]
        need = bottom_k - len(neg)
        if need > 0:
            # bottom of positive-sim list
            pos_sim_sorted = sorted(
                [(n, s) for n, s in sims if n not in pos_ids_set],
                key=lambda x: (x[1])
            )
            neg.extend(pos_sim_sorted[:need])

    def _pack(pairs: List[Tuple[int, float]]):
        ids, sims_ = [], []
        seen = set()
        for n, s in pairs:
            if n not in seen:
                seen.add(n)
                ids.append(n)
                sims_.append(s)
        return (
            np.asarray(ids, dtype=np.int32),
            np.asarray(sims_, dtype=np.float32),
        )

    return *_pack(pos), *_pack(neg)
    # return _pack(pos)[0], _pack(pos)[1], _pack(neg)[0], _pack(neg)[1]

def _resolve_window_size(window_size_arg: float, train_end_idx: int) -> int:
    """Interpret window_size argument.

    - If 0 < window_size_arg <= 1: treat as proportion of *train interactions*.
    - If window_size_arg > 1: treat as absolute window length.
    """
    if window_size_arg <= 0:
        raise ValueError("window_size must be positive")
    if window_size_arg <= 1.0:
        w = int(round(train_end_idx * float(window_size_arg)))
        return max(1, w)
    return int(window_size_arg)


def precompute(args):
    rng = random.Random(args.seed)

    (
        user2id,
        user_sequence_id,
        user_timediff_seq,
        user_prev_itemid_seq,
        item2id,
        item_sequence_id,
        item_timediff_seq,
        timestamp_sequence,
        feature_sequence,
        timediff_seq_for_adj,
    ) = load_network(args, time_scaling=False)

    user_seq = np.asarray(user_sequence_id, dtype=np.int64)
    item_seq = np.asarray(item_sequence_id, dtype=np.int64)
    
    n_train = int(args.train_proportion * len(user_seq))

    window_size = _resolve_window_size(args.window_size, n_train)

    # storage
    uu_pos_n = np.empty((n_train,), dtype=object)
    uu_pos_s = np.empty((n_train,), dtype=object)
    uu_neg_n = np.empty((n_train,), dtype=object)
    uu_neg_s = np.empty((n_train,), dtype=object)

    ii_pos_n = np.empty((n_train,), dtype=object)
    ii_pos_s = np.empty((n_train,), dtype=object)
    ii_neg_n = np.empty((n_train,), dtype=object)
    ii_neg_s = np.empty((n_train,), dtype=object)

    EMPTY_I = np.empty((0,), dtype=np.int32)
    EMPTY_F = np.empty((0,), dtype=np.float32)
    for arr in [uu_pos_n, uu_neg_n, ii_pos_n, ii_neg_n]:
        for i in range(n_train):
            arr[i] = EMPTY_I

    for arr in [uu_pos_s, uu_neg_s, ii_pos_s, ii_neg_s]:
        for i in range(n_train):
            arr[i] = EMPTY_F

    # window state
    window = deque()
    user_item_cnt = defaultdict(Counter)
    item_user_cnt = defaultdict(Counter)
    user_deg = defaultdict(int)
    item_deg = defaultdict(int)

    def _add(u, i):
        user_item_cnt[u][i] += 1
        item_user_cnt[i][u] += 1
        user_deg[u] += 1
        item_deg[i] += 1

    def _remove(u, i):
        user_item_cnt[u][i] -= 1
        if user_item_cnt[u][i] <= 0:
            del user_item_cnt[u][i]
        if not user_item_cnt[u]:
            del user_item_cnt[u]
        item_user_cnt[i][u] -= 1
        if item_user_cnt[i][u] <= 0:
            del item_user_cnt[i][u]
        if not item_user_cnt[i]:
            del item_user_cnt[i]

        # Decrease degrees
        # user_deg[u] -= 1
        # item_deg[i] -= 1

    # init window
    for u, i in zip(user_seq[:window_size], item_seq[:window_size]):
        window.append((int(u), int(i)))
        _add(int(u), int(i))

    # process
    for idx in tqdm(range(window_size - 1, n_train), desc="precompute neighbors"):
        u = int(user_seq[idx])
        i = int(item_seq[idx])
        
        # FIX: Remove current interaction from counters before computing neighbors
        # to prevent data leakage (using current interaction's own info to find neighbors)
        # Check if current interaction exists in counters
        current_was_in_counters = False
        if u in user_item_cnt and i in user_item_cnt[u] and user_item_cnt[u][i] > 0:
            current_was_in_counters = True
            _remove(u, i)
        
        active_users = {u for u, d in user_deg.items() if d >= args.min_deg}
        active_items = {i for i, d in item_deg.items() if d >= args.min_deg}

        uu = _select_pos_and_neg(
            u, user_item_cnt, item_user_cnt, user_deg, active_users,
            args.top_k, args.bottom_k, args.threshold,
            args.min_deg, not args.no_threshold, rng
        )
        ii = _select_pos_and_neg(
            i, item_user_cnt, user_item_cnt, item_deg, active_items,
            args.top_k, args.bottom_k, args.threshold,
            args.min_deg, not args.no_threshold, rng
        )

        uu_pos_n[idx], uu_pos_s[idx], uu_neg_n[idx], uu_neg_s[idx] = uu
        ii_pos_n[idx], ii_pos_s[idx], ii_neg_n[idx], ii_neg_s[idx] = ii
        
        # FIX: Add current interaction back to counters after computing neighbors
        # (needed for future iterations, especially for global window)
        if current_was_in_counters:
            _add(u, i)

        if idx == n_train - 1:
            break

        # Window update based on window_type
        if args.window_type == 'local':
            # Sliding window: remove oldest interaction
            u_old, i_old = window.popleft()
            _remove(u_old, i_old)

        # Add new interaction (both local and global)
        u_new = int(user_seq[idx + 1])
        i_new = int(item_seq[idx + 1])
        window.append((u_new, i_new))
        _add(u_new, i_new)

    np.savez_compressed(
        args.out,
        uu_pos_neighbors=uu_pos_n,
        uu_pos_sims=uu_pos_s,
        uu_neg_neighbors=uu_neg_n,
        uu_neg_sims=uu_neg_s,
        ii_pos_neighbors=ii_pos_n,
        ii_pos_sims=ii_pos_s,
        ii_neg_neighbors=ii_neg_n,
        ii_neg_sims=ii_neg_s,
    )

def _build_outpath(args: argparse.Namespace, window_size: int) -> str:
    out_dir = os.path.join(args.out, args.network)
    os.makedirs(out_dir, exist_ok=True)
    # Make filename stable and descriptive
    fname = (
        f"wt{args.window_type}__ws{window_size}"
        f"__tk{args.top_k}__th{args.threshold:g}__bk{args.bottom_k}__md{args.min_deg}.npz"
    )
    return os.path.join(out_dir, fname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute dynamic sim-based neighbors for DR")
    parser.add_argument('--network', default='wikipedia', help='Dataset name (same as main.py)')
    parser.add_argument('--train_proportion', default=0.8, type=float, help='Train prefix proportion (<=0.8)')
    parser.add_argument('--top_k', default=5, type=int, help='Top-k neighbors to keep')
    parser.add_argument("--bottom_k", type=int, default=None, help='Number of negative neighbors to keep')
    parser.add_argument('--threshold', default=0.8, type=float, help='Similarity threshold for extra neighbors')
    parser.add_argument('--no_threshold', action='store_true', help='Disable threshold expansion; keep only top-k')
    parser.add_argument("--min_deg", type=int, default=2)
    parser.add_argument('--window_size', default=0.05, type=float,
                        help='Window size: if <=1 treated as proportion of train interactions, else absolute int')
    parser.add_argument('--window_type', default='local', choices=['local', 'global', 'fusion'],
                        help='Window type: local (sliding window), global (cumulative window), or fusion (local+global)')
    # timediff scaling options (to match load_network)
    parser.add_argument('--timediff_scale_method', type=str, default='standard',
                        choices=['standard', 'log', 'log_minmax'],
                        help='Timediff scaling: standard (scale), log (log1p), log_minmax (log1p then minmax)')
    parser.add_argument('--timediff_separate_encoder', action='store_true',
                        help='If set, normalize timediff separately for user/item (log_minmax)')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--out', default='precomputed_neighbors', help='Output directory')
    args = parser.parse_args()

    if args.train_proportion > 0.8:
        raise SystemExit('train_proportion cannot be greater than 0.8 (same constraint as main.py).')

    args.datapath = f"data/{args.network}.csv"

    if args.bottom_k is None:
        args.bottom_k = args.top_k


    print("\n===== Precompute config =====")
    print(f"network={args.network}")
    print(f"datapath={args.datapath}")
    print(f"window_size={args.window_size}")
    print(f"window_type={args.window_type}")
    include_threshold = (not args.no_threshold)
    print(f"top_k={args.top_k}, threshold={args.threshold}, include_threshold={include_threshold}, bottom_k={args.bottom_k}, min_deg={args.min_deg}")

    window_plan = ['local', 'global'] if args.window_type == 'fusion' else [args.window_type]
    for wt in window_plan:
        args_single = copy.deepcopy(args)
        args_single.window_type = wt
        args_single.out = _build_outpath(args_single, args_single.window_size)

        if os.path.exists(args_single.out):
            print(f"Output file {args_single.out} already exists, skipping precomputation for {wt} window.")
            continue

        print(f"\n>>> Precomputing for window_type={wt}")
        precompute(args_single)

if __name__ == '__main__':
    main()
