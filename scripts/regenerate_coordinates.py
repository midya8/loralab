"""Rebuild all original epoch boundaries and complete coordinate witnesses."""
import gzip,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from loralab.coordinate_fixtures import build_archive,trajectory_report
from loralab.coordinates import audit

def write(path,obj):
    raw=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw,mtime=0))
    return raw
archive=build_archive()
raw=write(ROOT/'loralab/coordinate_data/6-original-epochs.json.gz',archive)
witnesses=[]
for rank in [1,2,4,8]:
    for epoch in [0,1,10,30]:
        for scale in [.25,1,4]:
            witnesses.append({'selection':{'rank':rank,'epoch':epoch,'scale':scale,'batch':'first_train','learning_rate':.1},
                              'report':trajectory_report(archive,rank,epoch,'first_train',scale,.1)})
toy={"W":[[.2,-.1],[.5,.1]],"bias":[0.,.1],"A":[[.3,.4],[-.2,.6]],"B":[[.7,-.2],[.1,.5]],"examples":[{"x":[1.,-.5],"y":0},{"x":[.2,.8],"y":1}]}
controls=[{"name":"identity","options":{"scalar_gauge":1}},
          {"name":"negative","options":{"scalar_gauge":-2}},
          {"name":"shear","options":{"gauge":[[1,2],[0,1]]}},
          {"name":"orthogonal","options":{"gauge":[[.6,-.8],[.8,.6]]}},
          {"name":"permutation","options":{"gauge":[[0,1],[1,0]]}},
          {"name":"zero-rate","options":{"scalar_gauge":4,"learning_rate":0}}]
for item in controls:
    item['problem']=toy
    item['report']=audit(toy,**item['options'])
write(ROOT/'loralab/coordinate_data/8-controls.json.gz',controls)
write(ROOT/'loralab/coordinate_data/7-witnesses.json.gz',witnesses)
(ROOT/'docs/5-data.js').write_text('globalThis.LORA_ARCHIVE='+raw.decode()+';\n')
print('124 full epoch states,',len(witnesses),'complete report witnesses,',len(raw),'archive JSON bytes')
