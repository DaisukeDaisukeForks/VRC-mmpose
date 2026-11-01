import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from 'node:fs';
import {POSE,makeROI,decodeSimCC,calibrationFromValues,estimateDistance,normalizeRGBA,OneEuro}from '../dist/core.js';
test('ROI pads and preserves virtual off-screen space without scaling original coordinates',()=>{
  const r=makeROI([-20,-30,80,170],288,384,1.25);
  assert.equal(r.x+r.width/2,30);assert.equal(r.y+r.height/2,70);assert.equal(r.width/r.height,.75);assert.ok(r.x<0);
});
test('SimCC3D uses depth=288 (not image height=384), preserves raw confidence and image coordinates',()=>{
  const build=(bins,index)=>{const data=new Float32Array(133*bins);for(let k=0;k<133;k++)data[k*bins+index]=.7;return{data,dims:[1,133,bins]};};
  const p=decodeSimCC(build(576,288),build(768,384),build(576,288),{x:100,y:50,width:600,height:800},1000,1000);
  assert.equal(p.length,133);assert.equal(p[0].x,400);assert.equal(p[0].y,450);assert.equal(p[0].relativeZ,0);assert.ok(Math.abs(p[0].rawScore-.7)<1e-6);
  const shifted=decodeSimCC(build(576,288),build(768,384),build(576,432),{x:0,y:0,width:288,height:384},288,384);
  assert.ok(Math.abs(shifted[0].relativeZ-POSE.zRange/2)<1e-8);
});
test('nonpositive SimCC confidence produces missing observations',()=>{
  const t=bins=>({dims:[1,133,bins],data:new Float32Array(133*bins)}),p=decodeSimCC(t(576),t(768),t(576),{x:0,y:0,width:288,height:384},288,384);
  assert.equal(p[0].score,0);assert.equal(p[0].relativeZ,null);assert.equal(p[0].inFrame,false);
});
test('RGB normalization uses MMPose means and NCHW channel order',()=>{
  const t=normalizeRGBA(new Uint8ClampedArray([123,116,103,255]),1,1);assert.ok(Math.abs(t[0]-(123-123.675)/58.395)<1e-6);assert.ok(Math.abs(t[2]-(103-103.53)/57.375)<1e-6);
});
test('distance fitting follows known bone lengths instead of bbox height; uncalibrated emits null',()=>{
  const c=calibrationFromValues({height:170,fov:65,fx:1000,fy:1000,cameraHeight:1.2},1920,1080);
  const positions={5:[-.1955,-.51,0],6:[.1955,-.51,0],7:[-.1955,-.1938,0],8:[.1955,-.1938,0],9:[-.1955,.0544,0],10:[.1955,.0544,0],11:[-.1445,0,0],12:[.1445,0,0],13:[-.1445,.4165,0],14:[.1445,.4165,0],15:[-.1445,.8347,0],16:[.1445,.8347,0]};
  const projected=depth=>Array.from({length:133},(_,id)=>{const p=positions[id];return p?{id,x:960+p[0]/depth*1000,y:540+p[1]/depth*1000,relativeZ:0,score:1,inFrame:true}:{id,x:0,y:0,relativeZ:0,score:0,inFrame:false};});
  assert.equal(estimateDistance(projected(3),null,.3),null);
  assert.ok(Math.abs(estimateDistance(projected(3),c,.3).depth-3)<.03);
  assert.ok(Math.abs(estimateDistance(projected(5),c,.3).depth-5)<.05);
  const missing=projected(3).map(p=>({...p,score:0}));assert.equal(estimateDistance(missing,c,.3),null);
});
test('One Euro filter resets on backward seeks and does not reuse future observations',()=>{
  const f=new OneEuro();assert.equal(f.filter('x',0,0),0);assert.ok(f.filter('x',10,.016)<10);assert.equal(f.filter('x',5,0),5);
});
test('browser decoder matches actual ONNX maxima and independently computed Python reference',()=>{
  const f=JSON.parse(readFileSync(new URL('../verification/decoder-fixture.json',import.meta.url),'utf8'));
  const tensors=[576,768,576].map((bins,axis)=>{const data=new Float32Array(133*bins);data.fill(-100);for(let k=0;k<133;k++)data[k*bins+f.indices[k][axis]]=f.scores[k];return{data,dims:[1,133,bins]};});
  const points=decodeSimCC(...tensors,f.roi,...f.source);
  for(let i=0;i<133;i++){assert.ok(Math.abs(points[i].x-f.decoded[i][0])<1e-5);assert.ok(Math.abs(points[i].y-f.decoded[i][1])<1e-5);assert.ok(Math.abs(points[i].relativeZ-f.decoded[i][2])<1e-8);}
});
test('every literal DOM reference in application code exists in the shipped HTML',()=>{
  const app=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8'),html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');
  const ids=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
  for(const m of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.has(m[1]),`Missing element: ${m[1]}`);
  assert.ok(!app.includes('getUserMedia'));assert.ok(!app.includes('RTCPeerConnection'));
});
