import json
import math
import os
import pathlib
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np

root = pathlib.Path(__file__).resolve().parents[2]
vectors_path = pathlib.Path(sys.argv[1])
audit_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
vectors = json.loads(vectors_path.read_text(encoding='utf8'))
audit = json.loads(audit_path.read_text(encoding='utf8'))
names = ['sft_08b'] + [f'{stage}_seed{seed}' for seed in (10, 20, 30) for stage in ('direct_fkl', 'direct_rkl', 'via_fkl', 'via_rkl')]
assert set(vectors) == set(names) == set(audit['models'])
labels = 'ABCDE'
gold = np.array([labels.index(row['gold']) for row in vectors[names[0]]])
n = len(gold)
assert n == 2194
features = [np.eye(5, dtype=np.float64)[gold]]
for name in names:
    rows = vectors[name]
    assert len(rows) == n and all(x['id'] == y['id'] and x['gold'] == y['gold'] for x, y in zip(rows, vectors[names[0]]))
    pred = np.array([labels.index(x['predicted']) if x['predicted'] in labels else -1 for x in rows])
    probs = np.array([x['choice_probabilities'] for x in rows])
    logp = np.array([x['choice_log_probabilities'] for x in rows])
    cap = np.array([x['hit_length_cap'] for x in rows], dtype=float)
    indicators = np.stack([(pred == k) for k in range(5)], axis=1).astype(float)
    tp = indicators * np.eye(5)[gold]
    features.append(np.column_stack((pred == gold, -logp[np.arange(n), gold], ((probs - np.eye(5)[gold]) ** 2).sum(axis=1), cap, indicators, tp)))
feature = np.column_stack(features)

def compute(counts):
    z = counts @ feature
    support = z[:, :5]
    metrics = np.empty((len(counts), len(names), 5))
    for j in range(len(names)):
        offset = 5 + 14 * j
        metrics[:, j, 0] = z[:, offset] * 100 / n
        metrics[:, j, 1] = np.mean(np.divide(2 * z[:, offset + 9:offset + 14], support + z[:, offset + 4:offset + 9], out=np.zeros_like(support), where=(support + z[:, offset + 4:offset + 9]) != 0), axis=1)
        metrics[:, j, 2] = z[:, offset + 1] / n
        metrics[:, j, 3] = z[:, offset + 2] / n
        metrics[:, j, 4] = z[:, offset + 3]
    return metrics

metric_names = ['accuracy_percent', 'macro_f1', 'choice_nll', 'brier_score', 'length_cap_count']
observed = compute(np.ones((1, n)))[0]
for j, name in enumerate(names):
    m = audit['models'][name]['overall']
    target = np.array([m['accuracy'] * 100, m['macro_f1'], m['choice_nll'], m['brier_score'], m['length_cap_count']])
    assert np.allclose(observed[j], target, atol=1e-10), name

count = 20000
rng = np.random.default_rng(20261027)
samples = np.empty((count, len(names), 5))
batch = 200
for begin in range(0, count, batch):
    size = min(batch, count - begin)
    ids = rng.integers(0, n, size=(size, n))
    counts = np.zeros((size, n), dtype=np.float64)
    np.add.at(counts, (np.arange(size)[:, None], ids), 1)
    samples[begin:begin + size] = compute(counts)
    if begin % 2000 == 0:
        print(f'bootstrap {begin + size}/{count}', flush=True)

def ci(a):
    q = np.percentile(a, [2.5, 97.5], axis=0)
    return {k: [float(q[0, i]), float(q[1, i])] for i, k in enumerate(metric_names)}

result = {'bootstrap': {'iterations': count, 'rng': 'numpy.default_rng PCG64', 'seed': 20261027, 'resampling': 'same question ID indices across every model and seed', 'percentiles': [2.5, 97.5], 'interpretation': 'question uncertainty conditional on these fixed trained models; not uncertainty over training seeds'}, 'models': {}, 'seeds': {}, 'seed_summary': {}, 'comparisons': {}, 'limitations': ['one fixed SFT initialization', 'via paths add separate intermediate teacher training', 'explanation factuality not reviewed', 'three training seeds are not 6582 independent questions', 'individual metric intervals are not simultaneous intervals']}
for j, name in enumerate(names):
    result['models'][name] = {'overall': audit['models'][name]['overall'], 'question_ci95': ci(samples[:, j])}

stages = ('direct_fkl', 'direct_rkl', 'via_fkl', 'via_rkl')
for stage in stages:
    ix = [names.index(f'{stage}_seed{s}') for s in (10, 20, 30)]
    result['seed_summary'][stage] = {'mean': dict(zip(metric_names, observed[ix].mean(axis=0).tolist())), 'sample_sd': dict(zip(metric_names, observed[ix].std(axis=0, ddof=1).tolist())), 'mean_question_ci95': ci(samples[:, ix].mean(axis=1))}

pairs = [('direct_fkl', 'sft_08b'), ('direct_rkl', 'sft_08b'), ('via_fkl', 'sft_08b'), ('via_rkl', 'sft_08b'), ('via_fkl', 'direct_fkl'), ('via_rkl', 'direct_rkl'), ('direct_fkl', 'direct_rkl'), ('via_fkl', 'via_rkl')]
for seed in (10, 20, 30):
    seeddata = {}; pvals = []
    for a, b in pairs:
        an = f'{a}_seed{seed}'; bn = b if b == 'sft_08b' else f'{b}_seed{seed}'
        ia, ib = names.index(an), names.index(bn)
        ar, br = vectors[an], vectors[bn]
        ac = np.array([x['predicted'] == x['gold'] for x in ar]); bc = np.array([x['predicted'] == x['gold'] for x in br])
        ao = int(np.sum(ac & ~bc)); bo = int(np.sum(~ac & bc)); total = ao + bo
        p = min(1.0, 2 * sum(math.comb(total, k) for k in range(min(ao, bo) + 1)) / (2 ** total)) if total else 1.0
        key = a + '_minus_' + b
        seeddata[key] = {'difference': dict(zip(metric_names, (observed[ia] - observed[ib]).tolist())), 'question_ci95': ci(samples[:, ia] - samples[:, ib]), 'paired_correct': {'both_correct': int(np.sum(ac & bc)), 'a_only': ao, 'b_only': bo, 'both_wrong': int(np.sum(~ac & ~bc))}, 'mcnemar_two_sided_exact_p': p}
        pvals.append((p, key))
    running = 0.0
    for rank, (p, key) in enumerate(sorted(pvals)):
        running = max(running, min(1., p * (len(pvals) - rank)))
        seeddata[key]['holm_adjusted_p_within_seed_8'] = running
    fi, fd = names.index(f'via_fkl_seed{seed}'), names.index(f'direct_fkl_seed{seed}')
    ri, rd = names.index(f'via_rkl_seed{seed}'), names.index(f'direct_rkl_seed{seed}')
    seeddata['interaction_delta_r_minus_delta_f'] = {'difference': dict(zip(metric_names, ((observed[ri] - observed[rd]) - (observed[fi] - observed[fd])).tolist())), 'question_ci95': ci((samples[:, ri] - samples[:, rd]) - (samples[:, fi] - samples[:, fd]))}
    result['seeds'][str(seed)] = seeddata

for a, b in pairs:
    key = a + '_minus_' + b
    ais = [names.index(f'{a}_seed{s}') for s in (10, 20, 30)]
    bis = [names.index(b if b == 'sft_08b' else f'{b}_seed{s}') for s in (10, 20, 30)]
    values = observed[ais] - observed[bis]
    result['comparisons'][key] = {'seed_mean': dict(zip(metric_names, values.mean(axis=0).tolist())), 'seed_sample_sd': dict(zip(metric_names, values.std(axis=0, ddof=1).tolist())), 'mean_question_ci95': ci((samples[:, ais] - samples[:, bis]).mean(axis=1))}
f = [names.index(f'via_fkl_seed{s}') for s in (10, 20, 30)]; fd = [names.index(f'direct_fkl_seed{s}') for s in (10, 20, 30)]
rr = [names.index(f'via_rkl_seed{s}') for s in (10, 20, 30)]; rd = [names.index(f'direct_rkl_seed{s}') for s in (10, 20, 30)]
interaction = (observed[rr] - observed[rd]) - (observed[f] - observed[fd])
result['comparisons']['interaction_delta_r_minus_delta_f'] = {'seed_mean': dict(zip(metric_names, interaction.mean(axis=0).tolist())), 'seed_sample_sd': dict(zip(metric_names, interaction.std(axis=0, ddof=1).tolist())), 'mean_question_ci95': ci(((samples[:, rr] - samples[:, rd]) - (samples[:, f] - samples[:, fd])).mean(axis=1))}
out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
json.loads(out_path.read_text(encoding='utf8'))
print('wrote', out_path, flush=True)
