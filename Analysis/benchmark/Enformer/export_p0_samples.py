#!/usr/bin/env python3
"""
Export P0-filtered 207,352 samples as a CSV preserving the original CREs format.
"""
import pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths & Constants ──
ORIGINAL_PATH = "/cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/TestData_20260509.randomCRE.4.pkl"
P0_CELL_LINES = ['A375', 'HT29', 'HepG2', 'K562', 'hPSC']
P0_RELABEL = {'WTC11': 'hPSC'}
LOCUS_CHR = "chr11"
LOCUS_START = 5_200_000
LOCUS_END = 5_300_000
RRA_BOTTOM = 0.5
MIN_CELL_LINES = 2
SPLIT_SEED = 42

OUTPUT_PATH = "./enformer_embeddings/p0_filtered_samples.csv"

class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)

# ── Load original data ──
print('[Data] Loading original 8-tuple...')
with open(ORIGINAL_PATH, 'rb') as f:
    bundle = _NumpyCompatUnpickler(f).load()

trainset_o, validset_o, testset_o, train_idx, valid_idx, test_idx, cre_sub, final_df = bundle[:8]
print(f'[Data] Original: train={trainset_o["length"]}, valid={validset_o["length"]}, test={testset_o["length"]}')

pool_index = list(train_idx) + list(valid_idx) + list(test_idx)

# ── Build lookup ──
def _build_lookup(t_idx, v_idx, te_idx):
    lookup = {}
    for j, idx in enumerate(t_idx):
        lookup[int(idx)] = ('train', j)
    for j, idx in enumerate(v_idx):
        lookup[int(idx)] = ('valid', j)
    for j, idx in enumerate(te_idx):
        lookup[int(idx)] = ('test', j)
    return lookup

lookup = _build_lookup(train_idx, valid_idx, test_idx)

# ── Filter: keep positives + bottom negatives shared >=2 cell lines ──
meta = cre_sub.loc[pool_index].copy()
meta['_pool_idx'] = np.arange(len(pool_index))

is_bottom_neg = (meta['label'] == 0) & (meta['RRA'] < RRA_BOTTOM)
shared_ids = (
    meta.loc[is_bottom_neg]
    .groupby('id')['cell_line']
    .nunique()
    .pipe(lambda s: s[s >= MIN_CELL_LINES].index)
)
keep_mask = (meta['label'] == 1) | (is_bottom_neg & meta['id'].isin(shared_ids))

# ── Locus holdout ──
locus_mask = (
    (meta['chr'] == LOCUS_CHR)
    & (meta['start'] < LOCUS_END)
    & (meta['end'] > LOCUS_START)
)
keep_split = keep_mask & ~locus_mask

# ── id-level 7:2:1 split ──
split_meta = meta.loc[keep_split]
ids_arr = split_meta['id'].unique()
rng = np.random.default_rng(SPLIT_SEED)
rng.shuffle(ids_arr)
n_id = len(ids_arr)
n_train_id = int(0.7 * n_id)
n_valid_id = int(0.2 * n_id)
train_ids_set = set(ids_arr[:n_train_id])
valid_ids_set = set(ids_arr[n_train_id:n_train_id + n_valid_id])
test_ids_set = set(ids_arr[n_train_id + n_valid_id:])

train_rows = split_meta['id'].isin(train_ids_set)
valid_rows = split_meta['id'].isin(valid_ids_set)
test_rows = split_meta['id'].isin(test_ids_set)

# ── P0 filter ──
def _is_p0(idx):
    cl = str(cre_sub.loc[idx, 'cell_line'])
    if cl.upper() == 'WTC11':
        return 'hPSC' in P0_RELABEL.values()
    return cl in set(P0_CELL_LINES)

# Collect indices per split
split_data = {}  # split_name → list of original indices
for name, row_set in [('train', train_rows), ('valid', valid_rows), ('test', test_rows)]:
    _split_idx = split_meta.loc[row_set, '_pool_idx'].astype(int).tolist()
    filtered = [pool_index[i] for i in _split_idx if _is_p0(pool_index[i])]
    split_data[name] = filtered

# Locus
_locus_raw = meta.loc[keep_mask & locus_mask, '_pool_idx'].astype(int).tolist()
locus_filtered = [pool_index[i] for i in _locus_raw if _is_p0(pool_index[i])]
split_data['locus'] = locus_filtered

total = sum(len(v) for v in split_data.values())
print(f'[Data] P0-filtered: train={len(split_data["train"])}, valid={len(split_data["valid"])}, test={len(split_data["test"])}, locus={len(split_data["locus"])} (total={total})')

# ── Build output DataFrame ──
rows_list = []
for split_name, indices in split_data.items():
    subset = cre_sub.loc[indices].copy()
    subset['split'] = split_name
    subset['original_index'] = indices
    rows_list.append(subset)

df = pd.concat(rows_list)
# Reorder columns: put split and original_index first, then original columns
orig_cols = [c for c in cre_sub.columns if c not in ('split', 'original_index')]
df = df[['split', 'original_index'] + orig_cols]
df = df.reset_index(drop=True)

print(f'[Data] Output shape: {df.shape}')
print(f'[Data] Split counts:')
print(df['split'].value_counts())

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f'[Data] Saved to {OUTPUT_PATH}')

# Also save as pickle for easy loading
pkl_path = OUTPUT_PATH.replace('.csv', '.pkl')
df.to_pickle(pkl_path)
print(f'[Data] Also saved to {pkl_path}')
