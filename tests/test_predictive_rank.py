import copy
import gzip
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path
from loralab.predictive_rank import audit, archive_problem, scalar, unique
from loralab.rank_verify import verify_report, verify_rank, bareiss
from loralab.rank_fixtures import controls, corpus, portable_gzip


def brute_rank(a):
    for k in range(min(len(a),len(a[0])),0,-1):
        for rows in itertools.combinations(range(len(a)),k):
            for cols in itertools.combinations(range(len(a[0])),k):
                # Independent Leibniz expansion, no elimination/rref.
                total=Q(0)
                for p in itertools.permutations(range(k)):
                    v=Q((-1)**sum(p[i]>p[j] for i in range(k) for j in range(i+1,k)))
                    for i in range(k):v*=a[rows[i]][cols[p[i]]]
                    total+=v
                if total:return k
    return 0


class PredictiveRankTests(unittest.TestCase):
    def test_exhaustive_small_matrices_all_references_against_minors(self):
        for vals in itertools.product([-1,0,1],repeat=6):
            a=[list(map(Q,vals[i:i+2]))for i in range(0,6,2)]
            raw=brute_rank(a)
            for t in range(3):
                p=dict(matrix=[list(vals[i:i+2])for i in range(0,6,2)],reference=t,probes=[[1,0]])
                r=audit(p);self.assertTrue(verify_report(r))
                d=[[v-w for v,w in zip(row,a[t])]for row in a]
                self.assertEqual(r['raw']['rank'],raw)
                self.assertEqual(r['global_certificate']['rank'],brute_rank(d))
                self.assertEqual(r['probe_certificate']['rank'],brute_rank([[row[0]]for row in d]))

    def test_reference_common_mode_and_gauge_invariance(self):
        p=dict(matrix=[[1,2,3],[2,4,6],[0,1,1]],reference=0,probes=[[1,0,0]])
        for t in range(3):
            p['reference']=t;r=audit(p);self.assertTrue(verify_report(r));self.assertEqual(r['global_certificate']['rank'],2)
            shifted=copy.deepcopy(p);shifted['matrix']=[[v+[7,-3,11][j]for j,v in enumerate(row)]for row in p['matrix']]
            self.assertEqual(audit(shifted)['global_certificate'],r['global_certificate'])
        a=dict(A=[[1,2],[3,4]],B=[[1,0],[0,1],[1,1]],reference=0,probes=[[1,0]])
        b=dict(A=[[1,2],[6,8]],B=[[1,0],[0,'1/2'],[1,'1/2']],reference=0,probes=[[1,0]])
        for key in ['raw','global_certificate','probe_certificate']:
            self.assertEqual(audit(a)[key],audit(b)[key])

    def test_complete_archived_corpus_regeneration_exact(self):
        c=corpus();stored=Path('loralab/rank_data/9-rank-witnesses.json').read_bytes()
        payload=json.dumps(c,sort_keys=True,separators=(',',':')).encode()
        self.assertEqual(payload+b'\n',stored)
        self.assertEqual(len(c['archive']),124)
        for s in c['archive']:
            self.assertEqual(s['raw'],0 if not s['epoch'] else min(3,s['width']))
            self.assertEqual(s['global_rank'],0 if not s['epoch'] else min(2,s['width']))

    def test_each_factor_leaf_and_minor_tamper_rejected(self):
        attempts=0
        for _,p in controls():
            original=audit(p)
            for key in ['raw','global_certificate','probe_certificate','probe_span']:
                c=original[key]
                for field in ['A','B']:
                    for i,row in enumerate(c[field]):
                        for j,_ in enumerate(row):
                            bad=copy.deepcopy(original);bad[key][field][i][j]=str(Q(row[j])+1)
                            with self.assertRaises(ValueError):verify_report(bad)
                            attempts+=1
                bad=copy.deepcopy(original);bad[key]['minor']=str(Q(c['minor'])+1)
                with self.assertRaises(ValueError):verify_report(bad)
                attempts+=1
        self.assertGreater(attempts,500)

    def test_finite_probe_rank_not_global_and_blind_counterexample(self):
        p=dict(matrix=[[0,0],[0,1],[0,0]],reference=0,probes=[[1,0]])
        r=audit(p);self.assertEqual((r['global_certificate']['rank'],r['probe_certificate']['rank']),(1,0));self.assertTrue(verify_report(r))
        self.assertEqual(r['blind_direction']['direction'],['0','1'])
        self.assertEqual(r['extension_unseen_coordinate']['feature'],1)
        p['probes']=[[1,0],[0,1]];r=audit(p);self.assertIsNone(r['blind_direction']);self.assertIsNone(r['extension_unseen_coordinate'])
        self.assertEqual(r['global_certificate']['rank'],r['probe_certificate']['rank'])

    def test_verifier_is_independent_and_metadata_strict(self):
        from unittest.mock import patch
        r=audit(controls()[1][1])
        with patch('loralab.predictive_rank.audit',side_effect=AssertionError),patch('loralab.predictive_rank.rref',side_effect=AssertionError):
            self.assertTrue(verify_report(r))
        for key in ['classes','features','reference','supplied_width']:
            bad=copy.deepcopy(r);bad[key]=True
            with self.assertRaises(ValueError):verify_report(bad)
        bad=copy.deepcopy(r);bad['parameter_counts']['global_factors']=-1
        with self.assertRaises(ValueError):verify_report(bad)

    def test_strict_schema_bounds_and_duplicate_refusal(self):
        good=controls()[0][1]
        bads=[None,{},dict(good,reference=True),dict(good,reference=3),dict(good,matrix=[[0]]),dict(good,probes=[]),dict(good,probes=[[1]]),dict(good,extra=1),dict(good,matrix=[[True,0],[0,1]]),dict(good,matrix=[[.1,0],[0,1]]),dict(good,matrix=[['1/0',0],[0,1]]),dict(good,matrix=[['01',0],[0,1]]),dict(good,matrix=[['1'*257,0],[0,1]]),dict(good,matrix=[[9007199254740992,0],[0,1]])]
        for bad in bads:
            with self.assertRaises(ValueError):audit(bad)
        with self.assertRaises(ValueError):json.loads('{"matrix":1,"matrix":2}',object_pairs_hook=unique)

    def test_cli_certificate_roundtrip_and_no_partial_report(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.json';p.write_text(json.dumps(controls()[1][1]))
            a=subprocess.run([sys.executable,'-m','loralab.predictive_rank','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(a.returncode,0);self.assertTrue(verify_report(json.loads(a.stdout)))
            p.write_text(a.stdout);v=subprocess.run([sys.executable,'-m','loralab.predictive_rank','--verify','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(v.returncode,0);self.assertIn('verified',v.stdout)
            p.write_text('{"matrix":1,"matrix":2}');v=subprocess.run([sys.executable,'-m','loralab.predictive_rank','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(v.returncode,2);self.assertEqual(v.stdout,'');self.assertIn('duplicate',v.stderr)

if __name__=='__main__':unittest.main()
