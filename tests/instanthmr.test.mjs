import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from 'node:fs';
import {POSE,JOINT_NAMES,FACE_IDS,HAND_IDS,BODY_IDS,BODY_EDGES,HAND_EDGES,personROI,cliffCondition,decodeInstant,halfToFloat,floatToHalf,checkProbe,COCO_TO_MHR,estimateDistance}from '../dist/instanthmr.js';
import {calibrationFromValues}from '../dist/core.js';
import {MODELS,modelURL}from '../dist/model-cache.js';
const fixture=JSON.parse(readFileSync(new URL('../verification/instant/decoder-fixture.json',import.meta.url),'utf8'));
test('MHR70 preserves ankles, both toe tips, heels and wrist/finger chains',()=>{
  assert.equal(POSE.count,70);assert.equal(JOINT_NAMES.length,70);assert.equal(new Set(JOINT_NAMES).size,70);
  assert.equal(FACE_IDS.size,5);assert.equal(HAND_IDS.size,42);assert.equal(BODY_IDS.size,23);
  assert.deepEqual(JOINT_NAMES.slice(13,21),['left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']);
  for(const pair of [[11,13],[12,14],[13,15],[13,17],[14,18],[14,20]])assert.ok(BODY_EDGES.some(e=>e.join()===pair.join()));
  assert.ok(HAND_EDGES.some(e=>e.join()==='41,24'));assert.ok(HAND_EDGES.some(e=>e.join()==='62,45'));
});
test('1.2 square crop retains off-screen padding; CLIFF uses original, unpadded box',()=>{
  const box=[-20,30,180,430],r=personROI(box);
  assert.equal(r.width,480);assert.equal(r.height,480);assert.equal(r.x,-160);assert.equal(r.y,-10);
  assert.deepEqual(r.bbox,box);
  const cond=cliffCondition(r.bbox,1000,500);
  assert.ok(Math.abs(cond[0]+.84)<1e-6);assert.ok(Math.abs(cond[1]+.08)<1e-6);assert.ok(Math.abs(cond[2]-.4)<1e-6);
});
test('browser MHR decoder matches independent actual ONNX sample reference for all 70 points',()=>{
  const prediction=decodeInstant(fixture.outputs,fixture.roi,...fixture.source);
  assert.equal(prediction.points.length,70);assert.equal(prediction.mhrParams.length,204);assert.equal(prediction.shapeParams.length,45);
  for(let i=0;i<70;i++){
    const p=prediction.points[i],expected=fixture.expectedRelative3d[i];
    assert.ok(Math.abs(p.x-fixture.expectedXY[i][0])<1e-4);assert.ok(Math.abs(p.y-fixture.expectedXY[i][1])<1e-4);
    assert.ok(Math.abs(p.relativeDisplay[0]-expected[0])<1e-7);assert.ok(Math.abs(p.relativeDisplay[1]+expected[1])<1e-7);assert.ok(Math.abs(p.relativeZ-expected[2])<1e-7);
    assert.equal(p.rawScore,null);assert.equal(p.confidenceAvailable,false);
    assert.ok(Math.abs(p.predictedCameraPosition[2]-(p.relativeModel[2]+prediction.camTrans[2]))<1e-7);
  }
  assert.ok(Math.abs(prediction.points[9].relativeDisplay[0]+prediction.points[10].relativeDisplay[0])<1e-7);
});
test('missing 3D output and non-finite half outputs fail before drawing or exports',()=>{
  const missing={...fixture.outputs};delete missing.joints_3d;assert.throws(()=>decodeInstant(missing,fixture.roi,...fixture.source),/joints_3d/);
  const bad={...fixture.outputs,joints_3d:{dims:[1,70,3],type:'float16',data:new Uint16Array(210).fill(0x7c00)}};
  assert.throws(()=>decodeInstant(bad,fixture.roi,...fixture.source),/非有限値/);
});
test('float16 output is decoded as IEEE values rather than integer bits',()=>{
  assert.equal(halfToFloat(0x3c00),1);assert.equal(halfToFloat(0xc000),-2);assert.equal(halfToFloat(1),2**-24);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
  for(const value of [0,-0,1,-2,.5,65504,2**-24])assert.equal(halfToFloat(floatToHalf(value)),value);
});
test('startup probe rejects plausible but incorrect finite GPU results',()=>{
  const reference=JSON.parse(readFileSync(new URL('../dist/instant-reference.json',import.meta.url),'utf8'));
  const shapes={mhr_params:[1,204],shape_params:[1,45],cam_trans:[1,3],joints_2d:[1,70,2],joints_3d:[1,70,3]};
  const outputs=Object.fromEntries(Object.entries(reference.values).map(([name,data])=>[name,{type:'float32',data:Float32Array.from(data),dims:shapes[name]}]));
  assert.equal(checkProbe(outputs,reference).joints_3d,0);
  outputs.joints_3d.data[13*3+2]+=.1;assert.throws(()=>checkProbe(outputs,reference),/数値異常/);
});
test('distance fit maps MHR hips/knees/ankles to body bones and returns all 70 positions',()=>{
  const c=calibrationFromValues({height:170,fov:65,fx:1000,fy:1000,cameraHeight:1.2},1920,1080);
  const positions={5:[-.1955,-.51],6:[.1955,-.51],7:[-.1955,-.1938],8:[.1955,-.1938],9:[-.1955,.0544],10:[.1955,.0544],11:[-.1445,0],12:[.1445,0],13:[-.1445,.4165],14:[.1445,.4165],15:[-.1445,.8347],16:[.1445,.8347]};
  const points=Array.from({length:70},(_,id)=>({id,x:0,y:0,score:0,valid:false,inFrame:false,relativeZ:0}));
  for(const[coco,pos]of Object.entries(positions)){const id=COCO_TO_MHR[coco];points[id]={id,x:960+pos[0]/3*1000,y:540+pos[1]/3*1000,score:1,valid:true,inFrame:true,relativeZ:0};}
  assert.equal(estimateDistance(points,null,.3),null);
  const result=estimateDistance(points,c,.3);assert.ok(Math.abs(result.depth-3)<.03);assert.equal(result.camera.length,70);assert.ok(result.world[13]);assert.equal(result.world[15],null);
});
test('main page fetches the exact original InstantHMR, with optional detector and no comparison redirect',()=>{
  assert.equal(MODELS.pose.repo,'momolesang/InstantHMR');assert.equal(MODELS.pose.size,80961622);
  assert.match(modelURL(MODELS.pose),/momolesang\/InstantHMR\/resolve\/3504446f/);
  const html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');assert.match(html,/<strong>InstantHMR<\/strong>/);assert.match(html,/value="full" selected/);assert.ok(!html.includes('http-equiv="refresh"'));
  const app=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8');assert.ok(!app.includes('decodeSimCC'));assert.ok(!app.includes('frameCanvas'));assert.ok(app.includes('else await waitForVideoFrame()'));
});
