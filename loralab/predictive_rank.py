"""Exact linear-softmax quotient rank, not numerical rank or a training method.

Input scalars are integers or rational strings. Archived binary64 factors are
converted to their exact integer ratios before multiplication, not rounded BA.
"""
from __future__ import annotations
import argparse
import json
import re
from fractions import Fraction as Q

SCHEMA = 'loralab.predictive-rank.v1'


def scalar(x, limit=256):
    if isinstance(x, bool) or not isinstance(x, (str, int)):
        raise ValueError('use integers or rational strings, not JSON floats')
    if isinstance(x, int) and abs(x) > 9007199254740991:
        raise ValueError('large integers must use rational strings')
    s = str(x)
    if len(s) > limit or not re.fullmatch(r'-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?', s):
        raise ValueError('invalid or oversized rational')
    return Q(s)


def matrix(x, rows=128, cols=128, limit=256):
    if not isinstance(x, list) or not 1 <= len(x) <= rows:
        raise ValueError('matrix row bound exceeded')
    if not isinstance(x[0], list) or not 1 <= len(x[0]) <= cols:
        raise ValueError('matrix column bound exceeded')
    n = len(x[0])
    if any(not isinstance(r, list) or len(r) != n for r in x):
        raise ValueError('matrix must be rectangular')
    return [[scalar(v, limit) for v in r] for r in x]


def trans(a):
    return [list(r) for r in zip(*a)]


def mul(a, b):
    return [[sum((x*y for x, y in zip(r, c)), Q(0)) for c in trans(b)] for r in a]


def strings(a):
    return [[str(v) for v in r] for r in a]


def rref(a):
    a = [r[:] for r in a]
    pivots = []
    for j in range(len(a[0])):
        k = len(pivots)
        p = next((i for i in range(k, len(a)) if a[i][j]), None)
        if p is None:
            continue
        a[k], a[p] = a[p], a[k]
        v = a[k][j]
        a[k] = [x/v for x in a[k]]
        for i in range(len(a)):
            if i != k:
                v = a[i][j]
                a[i] = [x-v*y for x, y in zip(a[i], a[k])]
        pivots.append(j)
        if len(pivots) == len(a):
            break
    return a, pivots


def inverse(a):
    n = len(a)
    rr, piv = rref([r+[Q(i == j) for j in range(n)] for i, r in enumerate(a)])
    if piv[:n] != list(range(n)):
        raise ValueError('singular core')
    return [r[n:] for r in rr]


def determinant(a):
    """Gaussian determinant, used only by the producer."""
    a = [r[:] for r in a]
    out = Q(1)
    for j in range(len(a)):
        p = next((i for i in range(j, len(a)) if a[i][j]), None)
        if p is None:
            return Q(0)
        if p != j:
            a[j], a[p] = a[p], a[j]
            out = -out
        v = a[j][j]
        out *= v
        for i in range(j+1, len(a)):
            t = a[i][j]/v
            for k in range(j+1, len(a)):
                a[i][k] -= t*a[j][k]
    return out


def factor(a):
    """Row skeleton upper witness plus nonzero-minor lower witness."""
    _, rows = rref(trans(a))
    _, cols = rref(a)
    r = len(rows)
    if not r:
        return dict(rank=0, B=[[] for _ in a], A=[], rows=[], columns=[], minor='1')
    core = [[a[i][j] for j in cols] for i in rows]
    b = mul([[row[j] for j in cols] for row in a], inverse(core))
    aa = [a[i][:] for i in rows]
    return dict(rank=r, B=strings(b), A=strings(aa), rows=rows,
                columns=cols, minor=str(determinant(core)))


def expanded(cert, m, d):
    if cert['rank'] == 0:
        return [[Q(0)]*d for _ in range(m)]
    return mul([[Q(x) for x in row] for row in cert['B']],
               [[Q(x) for x in row] for row in cert['A']])


def validate(p):
    if not isinstance(p, dict) or set(p) not in ({'matrix', 'reference', 'probes'}, {'A', 'B', 'reference', 'probes'}):
        raise ValueError('requires matrix or A+B, reference, probes only')
    if 'matrix' in p:
        a = matrix(p['matrix'], rows=8)
        width = None
    else:
        aa, b = matrix(p['A'], rows=8), matrix(p['B'], rows=8, cols=8)
        if len(aa) != len(b[0]):
            raise ValueError('factor shape mismatch')
        a, width = mul(b, aa), len(aa)
    m, d = len(a), len(a[0])
    t = p['reference']
    if m < 2 or isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < m:
        raise ValueError('requires 2-8 classes and an in-range integer reference')
    x = matrix(p['probes'], rows=16)
    if len(x[0]) != d:
        raise ValueError('probe shape mismatch')
    return a, x, t, width


def null_witness(x):
    rr, piv = rref(x)
    free = next((j for j in range(len(x[0])) if j not in piv), None)
    if free is None:
        return None
    u = [Q(0)]*len(x[0])
    u[free] = Q(1)
    for i, j in enumerate(piv):
        u[j] = -rr[i][free]
    return u


def audit(p):
    a, x, t, width = validate(p)
    m, d = len(a), len(a[0])
    delta = [[v-w for v, w in zip(row, a[t])] for row in a]
    raw, global_cert = factor(a), factor(delta)
    q = mul(delta, trans(x))
    probe = factor(q)
    s = probe['rank']
    if s:
        rows, cols = probe['rows'], probe['columns']
        core = [[q[i][j] for j in cols] for i in rows]
        b = [[q[i][j] for j in cols] for i in range(m)]
        aa = mul(inverse(core), [delta[i] for i in rows])
    else:
        b, aa = [[] for _ in range(m)], []
    extension = dict(rank=s, B=strings(b), A=strings(aa))
    n = expanded(extension, m, d)
    mismatch = next(((i, j) for i in range(m) for j in range(d) if delta[i][j] != n[i][j]), None)
    unseen = None
    if mismatch:
        i, j = mismatch
        unseen = dict(feature=j, **{'class': i}, contrast_change=str(n[i][j]-delta[i][j]))
    u = null_witness(x)
    blind = None if u is None else dict(direction=[str(v) for v in u],
        changed_class=(t+1) % m, unseen_contrast_change=str(sum(v*v for v in u)))
    report = dict(schema=SCHEMA, problem=p, arithmetic='exact rational; no rounded-probability guarantee',
                classes=m, features=d, supplied_width=width, reference=t,
                raw=raw, global_certificate=global_cert, probe_certificate=probe,
                probe_extension=extension, probe_span=factor(x),
                extension_unseen_coordinate=unseen, blind_direction=blind,
                parameter_counts=dict(supplied=None if width is None else width*(m+d),
                    global_factors=global_cert['rank']*(m+d), probe_factors=s*(m+d)))
    def check(v):
        if isinstance(v, str) and len(v) > 8192:
            raise ValueError('generated rational exceeds 8192-character operational bound')
        if isinstance(v, list):
            for x in v: check(x)
        if isinstance(v, dict):
            for x in v.values(): check(x)
    check(report)
    if len(json.dumps(report)) > 4_000_000:
        raise ValueError('generated report exceeds 4 MB operational bound')
    return report


def archive_problem(archive, width=4, epoch=30, probe_count=3):
    run = next(r for r in archive['runs'] if r['factor_width'] == width)
    s = next(s for s in run['states'] if s['epoch'] == epoch)
    d = len(s['A'][0])
    def exact(a):
        return [[str(Q(v)) for v in row] for row in a]
    return dict(A=exact(s['A']), B=exact(s['B']), reference=0,
                probes=[[int(i == j) for j in range(d)] for i in range(probe_count)])


def unique(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError('duplicate JSON key: '+k)
        d[k] = v
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        with open(args.input, encoding='utf8') as f:
            text = f.read(4_000_001)
        if len(text) > 4_000_000:
            raise ValueError('input exceeds 4 MB text bound')
        p = json.loads(text, object_pairs_hook=unique)
        if args.verify:
            from .rank_verify import verify_report
            verify_report(p)
            print('Exact certificate verified; runtime probabilities and training not certified')
        else:
            print(json.dumps(audit(p), indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, StopIteration, OSError, RecursionError) as e:
        parser.exit(2, 'Refused: '+str(e)+'\n')


if __name__ == '__main__':
    main()
