'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const C=require('../docs/coordinates.js');
const archive=JSON.parse(zlib.gunzipSync(fs.readFileSync('loralab/coordinate_data/original-epochs.json.gz')));
const witnesses=JSON.parse(zlib.gunzipSync(fs.readFileSync('loralab/coordinate_data/witnesses.json.gz')));
let count=0,maxDelta=0;
function compare(a,b,path='root'){
 assert.equal(typeof a,typeof b,path);
 if(typeof b==='number'){assert.ok(Number.isFinite(a));const error=Math.abs(a-b)/Math.max(1,Math.abs(a),Math.abs(b));maxDelta=Math.max(maxDelta,error);assert.ok(error<=1e-12,path+': '+a+' vs '+b);count++;}
 else if(Array.isArray(b)){assert.ok(Array.isArray(a),path);assert.equal(a.length,b.length,path);b.forEach((v,i)=>compare(a[i],v,path+'['+i+']'));}
 else if(b&&typeof b==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path);for(const k of Object.keys(b))compare(a[k],b[k],path+'.'+k);}
 else assert.equal(a,b,path);
}
test('48 complete native report witnesses plus six non-scalar/orthogonal controls',()=>{
 for(const w of witnesses){const s=w.selection;let r=C.audit(C.problemAt(archive,s.rank,s.epoch,s.batch),{scalar_gauge:s.scale,learning_rate:s.learning_rate});r.fixture_selection={factor_width:s.rank,epoch:s.epoch,batch:s.batch,original_online_learning_rate:.5,diagnostic_is_retraining:false};compare(r,w.report);}
 const controls=JSON.parse(zlib.gunzipSync(fs.readFileSync('loralab/coordinate_data/controls.json.gz')));
 for(const w of controls)compare(C.audit(w.problem,w.options),w.report);
 console.log('complete numeric leaves',count,'max normalized delta',maxDelta);
});
test('strict JSON preserves strings and rejects duplicates, overflow, depth and nonstandard whitespace',()=>{
 assert.deepEqual(C.strictJSON('{"label":"[\\\"{}]"}'),{label:'["{}]'});
 for(const x of ['{"x":1,"x":2}','[NaN]','[Infinity]','[1e999]','['.repeat(17)+'0'+']'.repeat(17),'[1,]','{}false','\v{}','1'.repeat(1000001)])assert.throws(()=>C.strictJSON(x));
 assert.deepEqual(C.strictJSON('{"__proto__":1}'),JSON.parse('{"__proto__":1}'));
});
test('every retained epoch and full-batch diagnostics retain identity/compensation',()=>{
 for(const rank of [1,2,4,8])for(let epoch=0;epoch<=30;epoch++){
 const p=C.problemAt(archive,rank,epoch),r=C.audit(p,{scalar_gauge:4,learning_rate:.1});
 assert.ok(r.same_model_logits_max_abs_residual<1e-12);
 assert.ok(r.compensated_next_delta_max_abs_residual<1e-12);
 assert.ok(r.transformed.update_identity_max_abs_residual<1e-12);
 }
 const full=C.audit(C.problemAt(archive,2,1,'full_train'),{scalar_gauge:4});assert.equal(full.original.before.logits.length,120);assert.ok(full.compensated_next_delta_max_abs_residual<1e-12);
 for(const [r,e]of[[3,1],[true,1],[2,31],[2,-1],[2,true]])assert.throws(()=>C.problemAt(archive,r,e));
});
