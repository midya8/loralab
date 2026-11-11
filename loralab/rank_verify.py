"""Independent upper/lower witness checks. Never calls rank, rref or audit."""
from fractions import Fraction as Q
from .predictive_rank import scalar, matrix, validate, mul, trans, SCHEMA


def require(ok, message):
    if not ok:
        raise ValueError('certificate: '+message)


def bareiss(a):
    """Fraction-free cross-product determinant, independent of producer."""
    a = [r[:] for r in a]
    n, sign, previous = len(a), 1, Q(1)
    if not n:
        return Q(1)
    for k in range(n-1):
        p = next((i for i in range(k, n) if a[i][k]), None)
        if p is None:
            return Q(0)
        if p != k:
            a[k], a[p] = a[p], a[k]
            sign *= -1
        pivot = a[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                a[i][j] = (a[i][j]*pivot-a[i][k]*a[k][j])/previous
        previous = pivot
    return sign*a[-1][-1]


def factors(c, m, d):
    r = c['rank']
    require(type(r) is int and 0 <= r <= min(m, d), 'invalid rank')
    if not r:
        require(c['A'] == [] and c['B'] == [[] for _ in range(m)], 'rank-zero shapes')
        return [[Q(0)]*d for _ in range(m)]
    a, b = matrix(c['A'], rows=16, limit=8192), matrix(c['B'], rows=128, cols=16, limit=8192)
    require(len(a) == r and len(a[0]) == d and len(b) == m and len(b[0]) == r, 'factor shapes')
    return mul(b, a)


def verify_rank(a, c):
    require(isinstance(c, dict) and set(c) == {'rank','B','A','rows','columns','minor'}, 'rank fields')
    m, d = len(a), len(a[0])
    require(factors(c, m, d) == a, 'upper product witness')
    r, rows, cols = c['rank'], c['rows'], c['columns']
    for v, n in [(rows, m), (cols, d)]:
        require(isinstance(v, list) and len(v) == r and all(type(i) is int and 0 <= i < n for i in v) and len(set(v)) == r, 'minor indices')
    actual = bareiss([[a[i][j] for j in cols] for i in rows])
    require(actual != 0 and scalar(c['minor'], 8192) == actual, 'nonzero minor lower witness')
    return r


def verify_report(r):
    require(isinstance(r, dict) and set(r) == {'schema','problem','arithmetic','classes','features','supplied_width','reference','raw','global_certificate','probe_certificate','probe_extension','probe_span','extension_unseen_coordinate','blind_direction','parameter_counts'}, 'report fields')
    require(r['schema'] == SCHEMA and r['arithmetic'] == 'exact rational; no rounded-probability guarantee', 'schema/arithmetic')
    a, x, t, width = validate(r['problem'])
    m, d = len(a), len(a[0])
    require(all(type(r[k]) is int for k in ['classes','features','reference']) and (r['supplied_width'] is None or type(r['supplied_width']) is int), 'metadata types')
    require([r['classes'],r['features'],r['supplied_width'],r['reference']] == [m,d,width,t], 'metadata')
    delta = [[v-w for v,w in zip(row,a[t])] for row in a]
    verify_rank(a, r['raw'])
    g = verify_rank(delta, r['global_certificate'])
    q = mul(delta, trans(x))
    s = verify_rank(q, r['probe_certificate'])
    span = verify_rank(x, r['probe_span'])
    e = r['probe_extension']
    require(set(e) == {'rank','B','A'} and e['rank'] == s, 'extension fields')
    n = factors(e, m, d)
    require(mul(n, trans(x)) == q and all(v == 0 for v in n[t]), 'extension must match all probe contrasts')
    unseen = r['extension_unseen_coordinate']
    if unseen is None:
        require(n == delta, 'false all-input extension match')
    else:
        require(set(unseen) == {'feature','class','contrast_change'}, 'unseen fields')
        i,j = unseen['class'],unseen['feature']
        require(type(i) is int and type(j) is int and 0 <= i < m and 0 <= j < d, 'unseen indices')
        v = n[i][j]-delta[i][j]
        require(v != 0 and scalar(unseen['contrast_change'],8192) == v, 'unseen basis-vector contrast')
    blind = r['blind_direction']
    if blind is None:
        require(span == d, 'missing probe blind direction')
    else:
        require(span < d and set(blind) == {'direction','changed_class','unseen_contrast_change'}, 'blind fields')
        u = [scalar(v,8192) for v in blind['direction']]
        require(len(u) == d and any(u), 'blind nonzero shape')
        require(all(sum(a*b for a,b in zip(row,u)) == 0 for row in x), 'blind must annihilate probes')
        require(type(blind['changed_class']) is int and 0 <= blind['changed_class'] < m and blind['changed_class'] != t, 'blind class')
        require(scalar(blind['unseen_contrast_change'],8192) == sum(v*v for v in u), 'blind unseen response')
    require(isinstance(r['parameter_counts'], dict) and all(v is None or type(v) is int for v in r['parameter_counts'].values()), 'count types')
    require(r['parameter_counts'] == dict(supplied=None if width is None else width*(m+d), global_factors=g*(m+d), probe_factors=s*(m+d)), 'algebraic scalar counts')
    return True
