import {makeROI,clamp,worldPoint,estimateDistance,BODY_EDGES,HAND_EDGES,FACE_EDGES,bboxFromPoints}from './core.js';
export {BODY_EDGES,HAND_EDGES,FACE_EDGES,bboxFromPoints};
export const POSE={width:192,height:256,count:133};
export function configurePose(model){POSE.is3D=model.id.startsWith('rtmw3d-');POSE.width=POSE.is3D?288:192;POSE.height=POSE.is3D?384:256;}
export const FACE_IDS=new Set(Array.from({length:68},(_,i)=>23+i)),HAND_IDS=new Set(Array.from({length:42},(_,i)=>91+i)),BODY_IDS=new Set(Array.from({length:23},(_,i)=>i));
export const groupOf=id=>id<23?'body':id<91?'face':'hand';
export const personROI=(bbox,padding=1.25)=>({...makeROI(bbox,POSE.width,POSE.height,padding),bbox:bbox.slice(0,4)});
const names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel'];
export function decode2D(outputs,roi,width,height){
  if(POSE.is3D)return decode3D(outputs,roi,width,height);
  const tx=outputs.simcc_x,ty=outputs.simcc_y;
  for(const[t,bins]of [[tx,384],[ty,512]])if(!t||t.dims.join()!==[1,133,bins].join()||t.data.some(v=>!Number.isFinite(v)))throw Error('RTMWのSimCC出力が不正です');
  const peak=(t,id)=>{let value=-Infinity,index=0,n=t.dims[2];for(let i=0;i<n;i++){const v=t.data[id*n+i];if(v>value){value=v;index=i;}}return[index/2,value];};
  return Array.from({length:133},(_,id)=>{const[px,sx]=peak(tx,id),[py,sy]=peak(ty,id),rawScore=Math.min(sx,sy),x=roi.x+px/192*roi.width,y=roi.y+py/256*roi.height;
    return{id,name:names[id]||`${groupOf(id)}_${id}`,x,y,rawScore,score:clamp(rawScore,0,1),valid:rawScore>0,inFrame:rawScore>0&&x>=0&&x<width&&y>=0&&y<height,relativeZ:null,relativeModel:[null,null,null],relativeDisplay:[null,null,null],predictedCameraPosition:null,depthSource:'unavailable'};});
}
function decode3D(outputs,roi,width,height){
  const xyz=outputs.keypoints,scores=outputs.scores;
  if(xyz?.dims.join()!=='1,133,3'||scores?.dims.join()!=='1,133'||xyz.data.some(v=>!Number.isFinite(v))||scores.data.some(v=>!Number.isFinite(v)))throw Error('RTMW3Dの関節・スコア出力が不正です');
  const points=Array.from({length:133},(_,id)=>{
    const px=xyz.data[id*3],py=xyz.data[id*3+1],z=xyz.data[id*3+2],rawScore=scores.data[id];
    const x=roi.x+px/288*roi.width,y=roi.y+py/384*roi.height;
    return{id,name:names[id]||`${groupOf(id)}_${id}`,x,y,rawScore,score:clamp(rawScore,0,1),valid:rawScore>0,inFrame:rawScore>0&&x>=0&&x<width&&y>=0&&y<height,relativeZ:z,relativeModel:[(px-144)*2.1744869/144,(py-192)*2.1744869/144,z],predictedCameraPosition:null,depthSource:'model-root-relative-z'};
  });
  const root=[0,1,2].map(c=>(points[11].relativeModel[c]+points[12].relativeModel[c])/2);
  for(const p of points){const q=p.relativeModel.map((v,c)=>v-root[c]);p.relativeDisplay=[q[0],-q[1],q[2]];}
  return points;
}
export function locate3D(points,c,threshold,previous,dt){
  const solution=estimateDistance(points,c,threshold,previous,dt);
  if(!solution)return null;
  for(const p of points){const q=solution.camera[p.id];if(q){const relative=q.map((v,c)=>v-solution.rootCameraPosition[c]);p.relativeModel=relative;p.relativeDisplay=[relative[0],-relative[1],relative[2]];}}
  return{...solution,method:'model-relative-z-and-calibrated-bone-length',depthAmbiguous:true};
}
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
const huber=x=>Math.abs(x)<.15?.5*x*x:.15*(Math.abs(x)-.075);
// Optimize ray depths, preserving each observed 2D position exactly. Depth is an estimate.
export function lift2D(points,c,threshold,previous=null,dt=1/60,direction=-1){
  const visible=i=>points[i]?.valid&&points[i].inFrame&&points[i].score>=threshold;
  if(!visible(11)||!visible(12))return null;
  const bones=[...c.bones,...[[15,17,.085],[15,18,.085],[15,19,.035],[16,20,.085],[16,21,.085],[16,22,.035]].map(([a,b,f])=>[a,b,c.heightM*f])].filter(([a,b])=>visible(a)&&visible(b));
  if(bones.length<4)return null;
  const rays=points.map(p=>[(p.x-c.cx)/c.fx,(p.y-c.cy)/c.fy,1]);
  const estimates=c.bones.filter(([a,b])=>visible(a)&&visible(b)).map(([a,b,l])=>l/Math.max(.001,Math.hypot(rays[a][0]-rays[b][0],rays[a][1]-rays[b][1]))).sort((a,b)=>a-b);
  const initial=clamp(estimates[Math.floor(estimates.length*.25)],.5,15),continuous=previous&&dt>0&&dt<.5;
  const depths=points.map((p,i)=>continuous&&previous.camera[i]?previous.camera[i][2]:initial+(i===13||i===14||i===9||i===10?direction*.04:0));
  const cameraAt=(i,z=depths[i])=>rays[i].map(v=>v*z);
  const active=[...new Set(bones.flatMap(b=>b.slice(0,2)))];
  const contacts=new Map();
  if(continuous)for(const id of [17,18,19,20,21,22]){
    const old=previous.world[id],older=previous.contactSamples?.[id];
    if(old&&older&&visible(id)&&Math.abs(old[1])<.10&&Math.hypot(points[id].x-older.x,points[id].y-older.y)/Math.max(dt,.001)<35&&(previous.contacts?.[id]||0)>=2)contacts.set(id,true);
  }
  const cost=()=>{
    let loss=0;
    for(const[a,b,l]of bones)loss+=Math.min(points[a].score,points[b].score)*huber((distance(cameraAt(a),cameraAt(b))-l)/l);
    const root=(depths[11]+depths[12])/2;
    loss+=.008*((root-initial)/Math.max(.5,initial))**2;
    if(continuous){
      loss+=.0002*((root-previous.depth)/.5)**2;
      for(const id of active)if(previous.camera[id])loss+=.012*(((depths[id]-root)-(previous.camera[id][2]-previous.depth))/.25)**2;
    }
    for(const id of contacts.keys())loss+=.10*(worldPoint(cameraAt(id),c)[1]/.10)**2;
    return loss;
  };
  for(let pass=0;pass<24;pass++){
    const step=.16*Math.pow(.87,pass);
    if(pass%3===0){const old=depths.slice(),base=cost();let best=base,shift=0;for(const delta of[-step,step]){for(const i of active)depths[i]=clamp(old[i]+delta,.25,20);const score=cost();if(score<best){best=score;shift=delta;}}for(const i of active)depths[i]=clamp(old[i]+shift,.25,20);}
    for(const id of active){const old=depths[id];let best=cost(),z=old;for(const delta of[-step,step]){depths[id]=clamp(old+delta,.25,20);const score=cost();if(score<best){best=score;z=depths[id];}}depths[id]=z;}
  }
  // Small landmarks inherit their anatomical anchor depth; do not claim independent depth observation.
  for(let i=0;i<133;i++)if(!active.includes(i)){const anchor=i>=112?10:i>=91?9:i>=23?0:i<5?5:11;depths[i]=depths[anchor];}
  const camera=points.map((p,i)=>visible(i)?cameraAt(i):null),world=camera.map(p=>p?worldPoint(p,c):null);
  const rootCameraPosition=camera[11].map((v,i)=>(v+camera[12][i])/2),worldRootPosition=worldPoint(rootCameraPosition,c);
  const residual=Math.sqrt(bones.reduce((sum,[a,b,l])=>sum+((distance(camera[a],camera[b])-l)/l)**2,0)/bones.length);
  for(const p of points){const q=camera[p.id];p.relativeZ=q?q[2]-rootCameraPosition[2]:null;p.relativeModel=q?q.map((v,i)=>v-rootCameraPosition[i]):[null,null,null];p.relativeDisplay=q?[p.relativeModel[0],-p.relativeModel[1],p.relativeModel[2]]:[null,null,null];p.depthSource=active.includes(p.id)?'bone-constrained-monocular-estimate':'anchor-depth-estimate';}
  const contactCounts={};for(const id of[17,18,19,20,21,22])contactCounts[id]=world[id]&&Math.abs(world[id][1])<.10?(previous?.contacts?.[id]||0)+1:0;
  return{camera,world,rootCameraPosition,worldRootPosition,depth:rootCameraPosition[2],rawDepth:rootCameraPosition[2],boneResidual:residual,bonesUsed:bones.length,uncertaintyM:Math.max(.15,rootCameraPosition[2]*(.10+residual)),quality:'monocular-ambiguous',method:'2d-ray-depth-bone-temporal-floor-fit',floorConstraint:contacts.size>0,contacts:contactCounts,contactSamples:Object.fromEntries([17,18,19,20,21,22].map(i=>[i,{x:points[i].x,y:points[i].y}])),depthAmbiguous:true};
}
