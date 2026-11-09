"""Strict bounded report input and JSON serialization for coordinate audits."""
import json
import math

from .coordinates import audit


def read_json(text):
    if not isinstance(text,str) or len(text.encode('utf-8'))>1_000_000:
        raise ValueError('JSON input must be at most 1 MB')
    depth=0;quoted=False;escaped=False
    for ch in text:
        if quoted:
            if escaped:escaped=False
            elif ch=='\\':escaped=True
            elif ch=='"':quoted=False
        elif ch=='"':quoted=True
        elif ch in '[{':
            depth+=1
            if depth>16:raise ValueError('JSON nesting exceeds 16')
        elif ch in ']}':depth-=1
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key: '+k)
            out[k]=v
        return out
    def number(s):
        x=float(s)
        if not math.isfinite(x):raise ValueError('JSON numeric overflow')
        return x
    def invalid(s):raise ValueError('nonfinite JSON token')
    try:return json.loads(text,object_pairs_hook=pairs,parse_float=number,parse_constant=invalid)
    except (json.JSONDecodeError,RecursionError) as exc:raise ValueError('invalid JSON') from exc


def report_json(text, scale=4, learning_rate=.1):
    return json.dumps(audit(read_json(text),scalar_gauge=scale,learning_rate=learning_rate),
                      allow_nan=False,indent=2,sort_keys=True)
