import {makeROI,clamp,estimateDistance as fitDistance,cameraPoint,worldPoint}from './core.js';
export const POSE=Object.freeze({width:224,height:224,count:70});
// Native MHR70 ordering; ankle, toe and heel IDs are not COCO-WholeBody IDs.
export const JOINT_NAMES=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel',
  ...['right','left'].flatMap(side=>[...['thumb','index','middle','ring','pinky'].flatMap(finger=>['tip','distal','intermediate','base'].map(part=>`${side}_${finger}_${part}`)),`${side}_wrist`]),
  'left_olecranon','right_olecranon','left_cubital_fossa','right_cubital_fossa','left_acromion','right_acromion','neck'];
export const FACE_IDS=new Set([0,1,2,3,4]),HAND_IDS=new Set(Array.from({length:42},(_,i)=>21+i));
export const BODY_IDS=new Set(Array.from({length:70},(_,i)=>i).filter(i=>!FACE_IDS.has(i)&&!HAND_IDS.has(i)));
export const groupOf=id=>FACE_IDS.has(id)?'face':HAND_IDS.has(id)?'hand':'body';
export const BODY_EDGES=[[69,5],[69,6],[5,6],[5,7],[7,62],[6,8],[8,41],[5,9],[6,10],[9,10],[9,11],[11,13],[10,12],[12,14],[13,15],[13,16],[13,17],[15,16],[14,18],[14,19],[14,20],[18,19],[69,0],[7,63],[8,64],[7,65],[8,66],[5,67],[6,68]];
export const FACE_EDGES=[[0,1],[0,2],[1,3],[2,4]];
export const HAND_EDGES=[];
for(const[start,wrist]of [[21,41],[42,62]])for(let f=0;f<5;f++){const tip=start+4*f;HAND_EDGES.push([wrist,tip+3],[tip+3,tip+2],[tip+2,tip+1],[tip+1,tip]);}
export const COCO_TO_MHR=[0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14];
export function personROI(bbox,padding=1.2){return {...makeROI(bbox,224,224,padding),bbox:Array.from(bbox).slice(0,4)};}
export function cliffCondition(bbox,width,height){
  const[x1,y1,x2,y2]=bbox;
  if(width<=0||height<=0||![x1,y1,x2,y2].every(Number.isFinite)||x2<=x1||y2<=y1)throw Error('CLIFF人物範囲が不正です');
  return new Float32Array([(x1+x2)/width-1,(y1+y2)/height-1,Math.max(x2-x1,y2-y1)/Math.max(width,height)]);
}
export function halfToFloat(bits){const sign=bits&0x8000?-1:1,exponent=bits>>10&31,mantissa=bits&1023;return exponent===31?mantissa?NaN:sign*Infinity:exponent===0?sign*2**-14*(mantissa/1024):sign*2**(exponent-15)*(1+mantissa/1024);}
export function floatToHalf(v){
  const f=new Float32Array([v]),b=new Uint32Array(f.buffer)[0],sign=b>>>16&0x8000;
  let exp=(b>>>23&255)-127+15,m=b&0x7fffff;
  if(exp>=31)return sign|0x7c00|(Number.isNaN(v)?0x200:0);
  if(exp<=0){if(exp< -10)return sign;m=(m|0x800000)>>(1-exp);return sign|((m+0xfff+(m>>13&1))>>13);}
  m+=0xfff+(m>>13&1);if(m&0x800000){m=0;exp++;}return sign|(exp<<10)|(m>>13);
}
export function inputTensor(ort,data,dims,type='tensor(float)'){
  if(type==='tensor(float16)')return new ort.Tensor('float16',Uint16Array.from(data,floatToHalf),dims);
  if(type!=='tensor(float)')throw Error(`未対応のモデル入力型: ${type}`);
  return new ort.Tensor('float32',data,dims);
}
export function floats(tensor){
  if(tensor.type==='float16'&&tensor.data instanceof Uint16Array)return Float32Array.from(tensor.data,halfToFloat);
  return Float32Array.from(tensor.data,Number);
}
export function validateOutputs(outputs){
  const shapes={mhr_params:[1,204],shape_params:[1,45],cam_trans:[1,3],joints_2d:[1,70,2],joints_3d:[1,70,3]},values={};
  for(const[name,dims]of Object.entries(shapes)){
    const tensor=outputs[name];
    if(!tensor||tensor.dims.join(',')!==dims.join(','))throw Error(`InstantHMR ${name}形状不一致: ${tensor?.dims}`);
    const data=floats(tensor);
    if(data.length!==dims.reduce((a,b)=>a*b,1)||data.some(v=>!Number.isFinite(v)))throw Error(`InstantHMR ${name}に非有限値または欠損があります`);
    values[name]=data;
  }
  return values;
}
export function decodeInstant(outputs,roi,width,height){
  const a=validateOutputs(outputs),xy=a.joints_2d,xyz=a.joints_3d;
  const root=[0,1,2].map(c=>(xyz[9*3+c]+xyz[10*3+c])/2);
  const points=Array.from({length:70},(_,id)=>{
    const x=roi.x+(xy[2*id]+1)*roi.width/2,y=roi.y+(xy[2*id+1]+1)*roi.height/2;
    const local=Array.from(xyz.slice(id*3,id*3+3)),relative=local.map((v,c)=>v-root[c]);
    return {id,name:JOINT_NAMES[id],x,y,relativeZ:relative[2],relativeModel:local,
      relativeDisplay:[relative[0],-relative[1],relative[2]],predictedCameraPosition:local.map((v,c)=>v+a.cam_trans[c]),
      rawScore:null,confidenceAvailable:false,valid:true,score:1,inFrame:x>=0&&x<width&&y>=0&&y<height};
  });
  return {points,mhrParams:Array.from(a.mhr_params),shapeParams:Array.from(a.shape_params),camTrans:Array.from(a.cam_trans)};
}
export function probeInput(){
  const n=224*224,out=new Float32Array(n*3),mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375];
  for(let c=0;c<3;c++)for(let y=0;y<224;y++)for(let x=0;x<224;x++)out[c*n+y*224+x]=(((x+3*y+17*c)%256)-mean[c])/std[c];
  return out;
}
export function checkProbe(outputs,reference){
  const a=validateOutputs(outputs),deltas={};
  for(const[name,tolerance]of Object.entries(reference.tolerance)){
    const expected=reference.values[name];if(!expected||expected.length!==a[name].length)throw Error('数値照合用のモデル参照が不正です');
    const max=Math.max(...a[name].map((v,i)=>Math.abs(v-expected[i])));deltas[name]=max;
    if(max>tolerance)throw Error(`${name}がCPU参照と一致しません（最大差 ${max.toFixed(5)}、許容 ${tolerance}）。GPUの数値異常を検出しました`);
  }
  return deltas;
}
export function bboxFromPoints(points,threshold,width,height){
  const visible=points.filter(p=>p.valid&&p.inFrame&&!HAND_IDS.has(p.id));
  if(visible.length<12||!points[9].inFrame||!points[10].inFrame)return null;
  const xs=visible.map(p=>p.x),ys=visible.map(p=>p.y),x1=Math.min(...xs),x2=Math.max(...xs),y1=Math.min(...ys),y2=Math.max(...ys);
  if(x2-x1<12||y2-y1<24)return null;
  return[clamp(x1,0,width),clamp(y1-.06*(y2-y1),0,height),clamp(x2,0,width),clamp(y2+.025*(y2-y1),0,height)];
}
export function estimateDistance(points,calibration,threshold,previous=null,dt=1/60){
  if(!points||!calibration)return null;
  const fitted=fitDistance(COCO_TO_MHR.map(id=>points[id]),calibration,threshold,previous,dt);
  if(!fitted)return null;
  const camera=points.map(p=>p.valid&&p.inFrame?cameraPoint(p,fitted.depth,calibration):null),world=camera.map(p=>p?worldPoint(p,calibration):null);
  return {...fitted,camera,world};
}
