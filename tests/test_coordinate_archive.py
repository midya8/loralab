import copy,gzip,json,math,random,unittest
from importlib.resources import files
from loralab.coordinate_fixtures import build_archive,problem_at,trajectory_report
from loralab.coordinate_report import read_json,report_json
from loralab.eval import build_fixture,run_evaluation
from loralab.lora import LoRAClassifier
from loralab.coordinates import audit

def close(test,a,b,path='root'):
    if isinstance(b,float):
        test.assertTrue(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),path+': '+str(a)+' vs '+str(b))
    elif isinstance(b,dict):
        test.assertEqual(set(a),set(b),path)
        for k in b:close(test,a[k],b[k],path+'.'+k)
    elif isinstance(b,list):
        test.assertEqual(len(a),len(b),path)
        for i,v in enumerate(b):close(test,a[i],v,path+'['+str(i)+']')
    else:test.assertEqual(a,b,path)

class TestArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=json.loads(gzip.decompress(files('loralab').joinpath('coordinate_data/original-epochs.json.gz').read_bytes()))

    def test_full_regeneration(self):
        close(self,build_archive(),self.archive)

    def test_original_every_recorded_factor_and_metric(self):
        fx=build_fixture()
        close(self,self.archive['W'],fx['base'].W)
        close(self,self.archive['bias'],fx['base'].b)
        close(self,self.archive['original_report'],run_evaluation(fx))
        for run in self.archive['runs']:
            adapter=LoRAClassifier(fx['base'],run['factor_width'],random.Random(0))
            self.assertEqual(len(run['states']),31)
            for epoch,state in enumerate(run['states']):
                close(self,state['A'],adapter.A)
                close(self,state['B'],adapter.B)
                self.assertEqual(state['completed_online_updates'],epoch*120)
                for split in ['testA','testB','trainB']:
                    self.assertEqual(state[split]['accuracy'],adapter.accuracy(fx[split]))
                    losses=[]
                    for x,y in fx[split]:
                        z=adapter.logits(x);top=max(z)
                        losses.append(top-z[y]+math.log(sum(math.exp(v-top) for v in z)))
                    self.assertAlmostEqual(state[split]['mean_cross_entropy'],sum(losses)/len(losses),delta=1e-11)
                if epoch<30:adapter.train_epoch(fx['trainB'],fx['lr'])

    def test_all_complete_witnesses(self):
        witnesses=json.loads(gzip.decompress(files('loralab').joinpath('coordinate_data/witnesses.json.gz').read_bytes()))
        self.assertEqual(len(witnesses),48)
        for item in witnesses:
            s=item['selection']
            close(self,trajectory_report(self.archive,s['rank'],s['epoch'],s['batch'],s['scale'],s['learning_rate']),item['report'])

    def test_full_batch_is_not_original_epoch(self):
        r=trajectory_report(self.archive,2,1,'full_train',4,.1)
        self.assertEqual(len(r['original']['before']['logits']),120)
        self.assertLess(r['compensated_next_delta_max_abs_residual'],1e-12)
        self.assertFalse(r['fixture_selection']['diagnostic_is_retraining'])
        for rank,epoch in [(3,1),(True,1),(2,31),(2,True),(2,-1)]:
            with self.assertRaises(ValueError):problem_at(self.archive,rank,epoch)
        with self.assertRaises(ValueError):problem_at(self.archive,batch='testB')

class TestStrictJSON(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=json.loads(gzip.decompress(files('loralab').joinpath('coordinate_data/original-epochs.json.gz').read_bytes()))

    def test_bad_syntax_limits_duplicates_nonfinite(self):
        for s in ['{"x":1,"x":2}','[NaN]','[Infinity]','[1e999]','['*17+'0'+']'*17,'x','{'+' '*1_000_000+'}']:
            with self.assertRaises(ValueError):read_json(s)
        self.assertEqual(read_json('{"label":"[\\\"{}]"}'),{'label':'["{}]'})

    def test_strict_schema_and_full_report(self):
        p=problem_at(self.archive,rank=1,epoch=1)
        r=json.loads(report_json(json.dumps(p)))
        self.assertEqual(r,audit(p,scalar_gauge=4))
        p['extra']='not accepted'
        with self.assertRaises(ValueError):report_json(json.dumps(p))

if __name__=='__main__':unittest.main()
