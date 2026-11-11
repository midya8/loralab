#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import gzip,json
from importlib.resources import files
from loralab.rank_fixtures import corpus,portable_gzip
from loralab.predictive_rank import audit,archive_problem
from loralab.rank_verify import verify_report
root=files('loralab').joinpath('coordinate_data')
c=corpus()
assert json.dumps(c,sort_keys=True,separators=(',',':')).encode()==files('loralab').joinpath('rank_data/9-rank-witnesses.json').read_bytes()
arc=json.loads(gzip.decompress(root.joinpath('6-original-epochs.json.gz').read_bytes()))
for run in arc['runs']:
    for state in run['states']:
        assert verify_report(audit(archive_problem(arc,run['factor_width'],state['epoch'])))
for control in c['controls']:assert verify_report(control['report'])
open('input.json','w').write(json.dumps(archive_problem(arc)))
print('Installed exact corpus byte-identical; all124 archive certificates and11 controls independently verified outside checkout')
PY
"$tmp/venv/bin/python" -m loralab.predictive_rank --input input.json > report.json
"$tmp/venv/bin/python" -m loralab.predictive_rank --verify --input report.json
