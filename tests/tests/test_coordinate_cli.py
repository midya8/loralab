import gzip,json,subprocess,sys,unittest
from importlib.resources import files
from loralab.coordinate_fixtures import problem_at
class TestCoordinateCLI(unittest.TestCase):
    def test_installed_archive_trajectory_and_invalid_epoch(self):
        p=subprocess.run([sys.executable,'-m','loralab','trajectory','--rank','4','--epoch','0'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        r=json.loads(p.stdout);self.assertEqual(r['factor_width'],4);self.assertEqual(r['matrix_rank_upper_bound'],3)
        p=subprocess.run([sys.executable,'-m','loralab','trajectory','--epoch','31'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('epoch',p.stderr)
    def test_stdin_json_and_no_traceback(self):
        archive=json.loads(gzip.decompress(files('loralab').joinpath('coordinate_data/original-epochs.json.gz').read_bytes()))
        p=subprocess.run([sys.executable,'-m','loralab','coordinates','--input','-'],input=json.dumps(problem_at(archive)),capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['schema'],'loralab.factor-coordinates.v1')
        for text in ['{"x":0,"x":1}','[1e999]','x']:
            p=subprocess.run([sys.executable,'-m','loralab','coordinates','--input','-'],input=text,capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertNotIn('Traceback',p.stderr)
