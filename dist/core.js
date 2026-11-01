// Pure geometry and decoding. MMPose SimCC3DLabel: input_size=(288,384,288), split=2.
export const POSE = Object.freeze({width:288,height:384,depth:288,count:133,split:2,zRange:2.1744869});
export const MEAN=[123.675,116.28,103.53], STD=[58.395,57.12,57.375];
export const BODY_EDGES=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,18],[15,19],[17,18],[16,20],[16,21],[16,22],[20,21]];
export const HAND_EDGES=[];
for(const start of [91,112])for(let f=0;f<5;f++){HAND_EDGES.push([start,start+1+f*4]);for(let j=0;j<3;j++)HAND_EDGES.push([start+1+f*4+j,start+2+f*4+j]);}
export const FACE_EDGES=[];
for(const [a,b,loop]of [[23,39,false],[40,44,false],[45,49,false],[50,53,false],[54,58,false],[59,64,true],[65,70,true],[71,82,true],[83,90,true]]){for(let i=a;i<b;i++)FACE_EDGES.push([i,i+1]);if(loop)FACE_EDGES.push([b,a]);}
export const ALL_EDGES=[...BODY_EDGES,...HAND_EDGES,...FACE_EDGES];
export const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
export function makeROI(bbox,width,height,padding=1.25){
  let [x1,y1,x2,y2]=bbox;
  if(![x1,y1,x2,y2].every(Number.isFinite)||x2<=x1||y2<=y1)throw Error('人物範囲が不正です');
  const cx=(x1+x2)/2,cy=(y1+y2)/2;
  let w=(x2-x1)*padding,h=(y2-y1)*padding;
  if(w/h>width/height)h=w*height/width;else w=h*width/height;
  return {x:cx-w/2,y:cy-h/2,width:w,height:h};
}
export function normalizeRGBA(rgba,width,height){
  const n=width*height,out=new Float32Array(3*n);
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)out[c*n+i]=(rgba[i*4+c]-MEAN[c])/STD[c];
  return out;
}
function maximum(tensor,k){
  const bins=tensor.dims.at(-1),offset=k*bins;let max=-Infinity,idx=0;
  for(let i=0;i<bins;i++){const v=tensor.data[offset+i];if(v>max){max=v;idx=i;}}
  return {index:idx,value:max};
}
export function decodeSimCC(x,y,z,roi,sourceWidth,sourceHeight){
  for(const [axis,t]of Object.entries({x,y,z}))if(!t||t.dims.length!==3||t.dims[0]!==1||t.dims[1]!==POSE.count||t.dims[2]!==({x:576,y:768,z:576})[axis])throw Error(`SimCC ${axis}形状不一致: ${t?.dims}`);
  const points=[],pixelToM=POSE.zRange/(POSE.depth/2);
  for(let i=0;i<POSE.count;i++){
    const ax=maximum(x,i),ay=maximum(y,i),az=maximum(z,i),score=Math.min(ax.value,ay.value);
    const valid=Number.isFinite(score)&&score>0;
    const px=valid?ax.index/POSE.split:-.5,py=valid?ay.index/POSE.split:-.5;
    const u=roi.x+px/POSE.width*roi.width,v=roi.y+py/POSE.height*roi.height;
    const dz=valid?(az.index/POSE.split/(POSE.depth/2)-1)*POSE.zRange:null;
    points.push({id:i,x:u,y:v,relativeZ:dz,rawScore:score,score:valid?clamp(score,0,1):0,
      inFrame:valid&&u>=0&&u<sourceWidth&&v>=0&&v<sourceHeight,
      relativeModel:[(px-POSE.width/2)*pixelToM,-(py-POSE.height/2)*pixelToM,dz]});
  }
  // Preserve decoder output exactly; root-center only the display coordinates.
  const hip=[11,12].map(i=>points[i]);
  const root=[0,1,2].map(c=>(hip[0].relativeModel[c]+hip[1].relativeModel[c])/2);
  for(const p of points)p.relativeDisplay=p.relativeModel.map((v,c)=>v===null?null:v-root[c]);
  return points;
}
export function bboxFromPoints(points,threshold,sourceWidth,sourceHeight){
  // Body/feet + hands define ROI; face landmarks do not outweigh a lost body.
  const visible=points.filter(p=>(p.id<23||p.id>=91)&&p.inFrame&&p.score>=threshold);
  if(visible.length<8||points[11].score<threshold||points[12].score<threshold)return null;
  const xs=visible.map(p=>p.x),ys=visible.map(p=>p.y);
  const x1=Math.min(...xs),x2=Math.max(...xs),y1=Math.min(...ys),y2=Math.max(...ys);
  if(x2-x1<12||y2-y1<24)return null;
  // Include a head margin from shoulder/eye observations without cumulative padding.
  return [clamp(x1,0,sourceWidth),clamp(y1-(y2-y1)*.06,0,sourceHeight),clamp(x2,0,sourceWidth),clamp(y2+(y2-y1)*.025,0,sourceHeight)];
}
export function selectBox(boxes,previous){
  if(!boxes.length)return null;
  const area=b=>(b[2]-b[0])*(b[3]-b[1]);
  if(!previous)return boxes.toSorted((a,b)=>area(b)-area(a))[0];
  const overlap=b=>{const i=Math.max(0,Math.min(b[2],previous[2])-Math.max(b[0],previous[0]))*Math.max(0,Math.min(b[3],previous[3])-Math.max(b[1],previous[1]));return i/(area(b)+area(previous)-i);};
  const sorted=boxes.toSorted((a,b)=>overlap(b)-overlap(a));
  return overlap(sorted[0])>.05?sorted[0]:boxes.toSorted((a,b)=>area(b)-area(a))[0];
}
export function calibrationFromValues(values,width,height){
  const heightM=values.height/100,fov=values.fov*Math.PI/180;
  const fx=values.fx||width/(2*Math.tan(fov/2)),fy=values.fy||fx;
  const c={...values,heightM,fx,fy,cx:values.cx??width/2,cy:values.cy??height/2,width,height,
    pitch:(values.cameraPitch||0)*Math.PI/180,cameraHeight:values.cameraHeight??1.2,
    kind:values.fx&&values.fy?'intrinsics-entered':'fov-approximation',
    bones:[[5,6,heightM*.23],[11,12,heightM*.17],[5,11,heightM*.30],[6,12,heightM*.30],
      [5,7,heightM*.186],[6,8,heightM*.186],[7,9,heightM*.146],[8,10,heightM*.146],
      [11,13,values.thigh?values.thigh/100:heightM*.245],[12,14,values.thigh?values.thigh/100:heightM*.245],
      [13,15,values.shin?values.shin/100:heightM*.246],[14,16,values.shin?values.shin/100:heightM*.246]]};
  if(![fx,fy,heightM,c.cameraHeight,c.pitch,c.cx,c.cy].every(Number.isFinite)||fx<=0||fy<=0||heightM<.8||heightM>2.3)throw Error('校正値の範囲を確認してください');
  return c;
}
export function cameraPoint(p,rootDepth,c){const z=rootDepth+p.relativeZ;return [(p.x-c.cx)*z/c.fx,(p.y-c.cy)*z/c.fy,z];}
export function worldPoint(p,c){const[x,y,z]=p,cp=Math.cos(c.pitch),sp=Math.sin(c.pitch);return [x,c.cameraHeight-cp*y-sp*z,cp*z-sp*y];}
const norm=v=>Math.hypot(...v),sub=(a,b)=>a.map((v,i)=>v-b[i]);
export function estimateDistance(points,c,threshold,previous=null,deltaTime=1/60){
  if(!c||!points||points[11].score<threshold||points[12].score<threshold||!points[11].inFrame||!points[12].inFrame)return null;
  const bones=c.bones.filter(([a,b])=>points[a].inFrame&&points[b].inFrame&&Math.min(points[a].score,points[b].score)>=threshold&&points[a].relativeZ!==null&&points[b].relativeZ!==null);
  if(bones.length<4)return null;
  const error=z=>{let sum=0,weights=0;for(const[a,b,length]of bones){
    const pa=cameraPoint(points[a],z,c),pb=cameraPoint(points[b],z,c);
    if(pa[2]<=.05||pb[2]<=.05)return Infinity;
    const relativeError=(norm(sub(pa,pb))-length)/length,w=Math.min(points[a].score,points[b].score);
    // Huber loss protects against an occluded, shortened limb.
    const e=Math.abs(relativeError),h=e<.15?.5*e*e:.15*(e-.075);sum+=w*h;weights+=w;
  }return sum/weights;};
  let best=.5,bestLoss=Infinity;
  for(let z=.5;z<=15;z+=.1){const e=error(z);if(e<bestLoss){bestLoss=e;best=z;}}
  let lo=Math.max(.3,best-.15),hi=Math.min(15,best+.15);
  for(let i=0;i<24;i++){const a=lo+(hi-lo)/3,b=hi-(hi-lo)/3;if(error(a)<error(b))hi=b;else lo=a;}
  const rawDepth=(lo+hi)/2;
  if(!Number.isFinite(bestLoss)||bestLoss>.035||rawDepth<.55||rawDepth>14.8)return null;
  const depth=previous&&deltaTime>0&&deltaTime<.5?previous.depth+(rawDepth-previous.depth)*(1-Math.exp(-deltaTime/.10)):rawDepth;
  const camera=points.map(p=>p.score>=threshold&&p.inFrame?cameraPoint(p,depth,c):null);
  const world=camera.map(p=>p?worldPoint(p,c):null);
  const rootCamera=camera[11].map((v,i)=>(v+camera[12][i])/2),rootWorld=worldPoint(rootCamera,c);
  const relativeResidual=Math.sqrt(bestLoss*2),uncertainty=Math.max(.08,depth*(relativeResidual+(c.kind==='fov-approximation'?.10:.05)));
  return {depth,rawDepth,rootCameraPosition:rootCamera,worldRootPosition:rootWorld,camera,world,boneResidual:relativeResidual,bonesUsed:bones.length,
    uncertaintyM:uncertainty,quality:c.kind==='fov-approximation'?'approximate':'model-dependent',method:'relative-z-and-bone-length-fit',floorConstraint:false};
}
export class OneEuro{
  constructor(){this.states=new Map();}
  reset(){this.states.clear();}
  filter(key,value,time){
    const old=this.states.get(key);if(!old||time<=old.time||time-old.time>.5){this.states.set(key,{value,raw:value,dx:0,time});return value;}
    const dt=time-old.time,alpha=f=>1/(1+1/(2*Math.PI*f*dt));
    const dx=old.dx+alpha(1)*((value-old.raw)/dt-old.dx),cutoff=1.5+.07*Math.abs(dx);
    const out=old.value+alpha(cutoff)*(value-old.value);this.states.set(key,{value:out,raw:value,dx,time});return out;
  }
}
export function percentile(values,p){if(!values.length)return null;const s=values.toSorted((a,b)=>a-b);return s[Math.min(s.length-1,Math.floor((s.length-1)*p))];}
