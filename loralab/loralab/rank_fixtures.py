"""Portable exact witness corpus, derived from preserved original epochs."""
import gzip
import hashlib
import io
import json
from importlib.resources import files
from .predictive_rank import audit, archive_problem
from .rank_verify import verify_report


def controls():
    return [
        ('pure-common-mode',dict(matrix=[[1,2],[1,2],[1,2]], reference=0,probes=[[1,0]])),
        ('full-rank-quotient',dict(matrix=[[1,0,0],[0,1,0],[0,0,1]],reference=0,probes=[[1,0,0]])),
        ('probe-blind-zero',dict(matrix=[[0,0],[0,1],[0,0]],reference=0,probes=[[1,0]])),
        ('spanning-probes',dict(matrix=[[1,0],[0,1],[0,0]],reference=2,probes=[[1,0],[0,1]])),
        ('duplicate-zero-probes',dict(matrix=[[1,2],[3,4]],reference=1,probes=[[0,0],[1,0],[1,0]])),
        ('tiny-nonzero-not-numerical-rank',dict(matrix=[[0,0],['1/1000000000000000000000000000000',0],[0,1]],reference=0,probes=[[1,0],[0,1]])),
        ('dependent-rational-rows',dict(matrix=[['1/3','2/7','-1/5'],['2/3','4/7','-2/5'],[0,0,0]],reference=2,probes=[[1,0,1],[0,1,0]])),
        ('factor-null-cancellation',dict(A=[[1,2],[1,2]],B=[[1,-1],[2,-2],[3,-3]],reference=1,probes=[[0,0]])),
        ('zero-factor',dict(A=[[1,2]],B=[[0],[0]],reference=0,probes=[[1,0]])),
        ('negative-pivot',dict(matrix=[[0,-2,0],[-1,0,0],[1,2,0]],reference=1,probes=[[1,1,0],[2,2,0]])),
        ('sixteen-probe-basis',dict(matrix=[[int(j==i) for j in range(16)]for i in range(8)],reference=0,probes=[[int(j==i) for j in range(16)]for i in range(16)])),
    ]


def digest(report):
    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def corpus():
    arc=json.loads(gzip.decompress(files('loralab').joinpath('coordinate_data/6-original-epochs.json.gz').read_bytes()))
    out={'schema':'loralab.predictive-rank-corpus.v1','controls':[],'archive':[]}
    for name,p in controls():
        r=audit(p);verify_report(r)
        out['controls'].append(dict(name=name,problem=p,report=r))
    for run in arc['runs']:
        for state in run['states']:
            width,epoch=run['factor_width'],state['epoch']
            r=audit(archive_problem(arc,width,epoch));verify_report(r)
            out['archive'].append(dict(width=width,epoch=epoch,raw=r['raw']['rank'],global_rank=r['global_certificate']['rank'],sha256=digest(r)))
    return out


def portable_gzip(data):
    b=io.BytesIO()
    with gzip.GzipFile(fileobj=b,mode='wb',filename='',mtime=0) as f:
        f.write(data)
    return b.getvalue()


if __name__=='__main__':
    from pathlib import Path
    c=corpus()
    payload=json.dumps(c,sort_keys=True,separators=(',',':')).encode()
    Path('loralab/rank_data').mkdir(exist_ok=True)
    Path('loralab/rank_data/9-rank-witnesses.json').write_bytes(payload)
    Path('docs/rank-data.js').write_text('globalThis.LORA_RANK_CORPUS='+payload.decode()+';\n')
    print(len(c['controls']),'controls;',len(c['archive']),'complete-report archive hashes')
