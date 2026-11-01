import {MEAN,STD,clamp,normalizeRGBA,selectBox,calibrationFromValues,OneEuro,percentile}from './core.js';
import {POSE,configurePose,locate3D,BODY_EDGES,HAND_EDGES,FACE_EDGES,FACE_IDS,HAND_IDS,BODY_IDS,groupOf,personROI,decode2D,lift2D,bboxFromPoints}from './rtmw2d.js';
import {initRuntime as loadRuntime,RUNTIME_VERSION}from './runtime.js';
import {MODELS,MODEL_VARIANTS,MODEL_CACHE,loadModel,cacheSummary,validateModel}from './model-cache.js';
import {PoseView}from './view3d.js';
import {createExecution}from './gpu-execution.js';

const $=id=>document.getElementById(id),video=$('source-video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
const view=new PoseView($('pose3d')),crop=document.createElement('canvas'),cropCtx=crop.getContext('2d',{willReadFrequently:true});
crop.width=POSE.width;crop.height=POSE.height;
const detectionCanvas=document.createElement('canvas'),detectionCtx=detectionCanvas.getContext('2d',{willReadFrequently:true});
detectionCanvas.width=detectionCanvas.height=640;
const filter=new OneEuro();
const state={ort:null,session:null,runner:null,detector:null,backend:'webgpu',busy:false,loading:false,running:false,sourceURL:null,videoVersion:0,token:0,
  bbox:null,manual:null,roi:null,points:null,distance:null,previousDistance:null,calibration:null,worldOrigin:null,
  records:[],timings:[],modelInfo:null,detectorInfo:null,frame:0,frameTimes:[],lastMediaTime:null,threshold:.3,
  startTime:0,wallMs:0,mediaSeconds:0,warmupMs:null,warmupShape:null,logs:[],detectorError:null,localModel:null,needsReload:false,
  reference:null,probeDeltas:null,layout:'NHWC',prediction:null,renderFrame:null,waitFrame:null,previewTime:null};
const num=id=>$(id).value.trim()===''?null:Number($(id).value);
const formatTime=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const mb=n=>`${(n/1e6).toFixed(1)} MB`;
function log(message){state.logs.push(`${new Date().toISOString()} ${message}`);if(state.logs.length>100)state.logs.shift();updateDiagnostics();}
function error(message,title='処理を確認してください'){$('error-title').textContent=title;$('error-message').textContent=message;$('error-panel').hidden=false;log(`ERROR ${message}`);}
$('dismiss-error').onclick=()=>$('error-panel').hidden=true;
function updateDiagnostics(){
  $('diagnostics').textContent=[`Model: ${MODELS.pose.repo}@${MODELS.pose.revision}`,`Backend: ${state.backend} / ${RUNTIME_VERSION} / ${state.layout}`,
    `WebGPU: ${navigator.gpu?'APIあり':'APIなし'} / secureContext: ${isSecureContext}`,
    `GPU buffer reuse: ${!!state.runner} / Graph Capture: false`,
    `Source: ${video.videoWidth||0} × ${video.videoHeight||0} / pixel coordinates before crop`,
    state.modelInfo?`SHA-256: ${state.modelInfo.sha256}\nCache source: ${state.modelInfo.source}`:'',
    state.session?`Inputs: ${state.session.inputNames.join(', ')}\nOutputs: ${state.session.outputNames.join(', ')}\nOutput shape: ${JSON.stringify(state.warmupShape)}\nWarm-up: ${state.warmupMs?.toFixed(1)} ms (計測から除外)`:'',
    `Calibration: ${state.calibration?.kind||'none'} / no fixed root depth / no floor clamping`,
    `Depth: ${POSE.is3D?'model relative-Z + calibrated root depth':'2D + constrained pseudo-3D'} / COCO133`,
    `Latest pose lag: ${state.points?Math.max(0,video.currentTime-(state.poseMediaTime??video.currentTime)).toFixed(3)+' s':'none'} / one inference in flight`,
    `Records: ${state.records.length} / GPU memory: browser APIから未計測`,
    ...state.logs.slice(-12)].filter(Boolean).join('\n');
}
async function refreshCache(){const s=await cacheSummary();$('cache-summary').textContent=s.error?'キャッシュを開けません':`${s.count}モデル保存済み${s.storage?.usage?` · 使用 ${mb(s.storage.usage)}`:''}`;return s;}
function enableControls(){const loaded=!!video.videoWidth,ready=!!state.session&&!state.loading&&!state.needsReload;$('run').disabled=!(loaded&&ready)||state.busy&&!state.running;$('step').disabled=!(loaded&&ready)||state.running||state.busy;$('play').disabled=!loaded;$('timeline').disabled=!loaded;$('load-model').disabled=state.loading||state.running||state.busy;$('wasm-mode').disabled=state.loading||state.running||state.busy;$('local-model').disabled=state.loading||state.running||state.busy;$('model-select').disabled=state.loading||state.running||state.busy;}

async function initRuntime(){
  if(state.ort)return state.ort;
  state.ort=await loadRuntime();return state.ort;
}
function progress(label,p){
  $('model-status').textContent=p.phase==='cache'?`${label} · 保存済みモデルを検証`:p.phase==='verify'?`${label} · SHA-256照合中`:`${label} · ダウンロード中`;
  $('download-size').textContent=`${mb(p.received)} / ${mb(p.total)}`;$('model-progress').value=Math.min(100,p.received/p.total*100);
}
async function prepareModel(){
  if(state.loading||state.running||state.busy)return;state.loading=true;enableControls();$('error-panel').hidden=true;
  try{
    state.backend=$('wasm-mode').checked?'wasm':'webgpu';$('backend').textContent=state.backend==='webgpu'?'WebGPU':'WASM · CPU診断';
    if(state.backend==='webgpu'){
      if(!navigator.gpu)throw Error('WebGPUがありません。PCのChrome / Edgeを使用してください。CPUで確認する場合は「WASM診断モード」を選んでモデルを準備してください。');
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
      if(!adapter)throw Error('WebGPUのアダプターを取得できません。ブラウザのハードウェアアクセラレーションを確認してください。');
      log(`GPU adapter: ${adapter.info?.vendor||'unknown'} / ${adapter.info?.architecture||'unknown'} / fallback=${adapter.isFallbackAdapter??'unknown'}`);
    }
    const ort=await initRuntime();
    await releasePose();
    if(state.detector){await state.detector.release();state.detector=null;}
    let model;
    if(state.localModel){const bytes=new Uint8Array(await state.localModel.arrayBuffer());model={bytes,sha256:await validateModel(bytes,MODELS.pose),source:'local-file'};}
    else model=await loadModel(MODELS.pose,p=>progress(MODELS.pose.name,p));
    state.modelInfo={sha256:model.sha256,source:model.source,bytes:model.bytes.byteLength,warning:model.warning};
    const layouts=state.backend==='webgpu'?['NHWC','NCHW']:['NCHW'];let lastError;
    for(const layout of layouts){
      state.layout=layout;$('model-status').textContent='実モデルの初回推論を実行中';
      try{
        if(MODELS.pose.gpuIO&&state.backend==='webgpu'){
          state.runner=await createExecution(ort,model.bytes,{layout,gpuIO:true,capture:false});state.session=state.runner.session;
        }else state.session=await ort.InferenceSession.create(model.bytes,{executionProviders:state.backend==='webgpu'?[{name:'webgpu',preferredLayout:layout}]:['wasm'],graphOptimizationLevel:'all'});
        const started=performance.now();let out;
        try{out=await runPose(new Float32Array(3*POSE.width*POSE.height));decode2D(out,{x:0,y:0,width:POSE.width,height:POSE.height},POSE.width,POSE.height);state.warmupMs=performance.now()-started;state.warmupShape=Object.fromEntries(Object.entries(out).map(([k,v])=>[k,v.dims]));}
        finally{if(out)Object.values(out).forEach(t=>t.dispose());}
        lastError=null;break;
      }catch(e){lastError=e;log(layout+': '+e.message);await releasePose().catch(()=>{});}
    }
    model.bytes=null;if(lastError)throw lastError;log('Actual RTMW warm-up succeeded');
    state.needsReload=false;$('model-status').textContent=`推論確認済み · ${state.backend==='webgpu'?'WebGPU':'WASM（CPU診断）'}`;$('download-size').textContent='SHA-256 一致';$('model-progress').value=100;$('load-model').innerHTML='再読み込み <span>↻</span>';
    if(state.modelInfo.warning)error(state.modelInfo.warning,'推論は実行できます');
    if(video.videoWidth)await inferFrame(false);
  }catch(e){
    await releasePose().catch(()=>{});
    $('model-status').textContent='準備中に問題が発生';error(`${String(e.message||e)}\n再試行、または同じモデルのローカルONNXを指定できます。`,'モデル準備を完了できませんでした');
  }finally{state.loading=false;enableControls();await refreshCache();updateROIHint();updateDiagnostics();}
}
async function prepareDetector(){
  if(state.detector)return;
  const ort=await initRuntime(),model=await loadModel(MODELS.detector,p=>progress('RTMDet-s',p));
  state.detectorInfo={sha256:model.sha256,source:model.source,bytes:model.bytes.byteLength};
  // RTMDet's exported NMS is executed in WASM; pose inference stays on the selected GPU.
  state.detector=await ort.InferenceSession.create(model.bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  model.bytes=null;state.detectorError=null;log(`Detector loaded: ${state.detector.inputNames} → ${state.detector.outputNames} (WASM / NMS)`);
}
function poseFeeds(image){return{input:new state.ort.Tensor('float32',image,[1,3,POSE.height,POSE.width])};}
async function releasePose(){const runner=state.runner,session=state.session;state.runner=null;state.session=null;if(runner)await runner.release();else if(session)await session.release();}
async function runPose(data){
  if(state.runner){const r=await state.runner.run(data);return{keypoints:{dims:[1,133,3],data:r.points,dispose(){}},scores:{dims:[1,133],data:r.scores,dispose(){}}};}
  const feeds=poseFeeds(data);try{return await state.session.run(feeds);}finally{Object.values(feeds).forEach(t=>t.dispose());}
}
function solvePose(points,dt=1/60,previous=state.previousDistance){
  const c=state.calibration||calibrationFromValues({height:num('height'),fov:num('fov'),cameraHeight:num('camera-height'),cameraPitch:num('camera-pitch'),thigh:num('thigh'),shin:num('shin')},video.videoWidth,video.videoHeight);
  return POSE.is3D?locate3D(points,c,state.threshold,previous,dt):lift2D(points,c,state.threshold,previous,dt,num('depth-direction'));
}
async function detectPerson(){
  if(!state.detector){await prepareDetector();$('model-status').textContent='推論確認済み · 自動切り抜き有効';}
  const w=video.videoWidth,h=video.videoHeight,scale=Math.min(640/w,640/h),dw=Math.round(w*scale),dh=Math.round(h*scale);
  // MMDeploy RTMDet uses top-left letterboxing, padding 114, BGR normalization.
  detectionCtx.fillStyle='rgb(114,114,114)';detectionCtx.fillRect(0,0,640,640);detectionCtx.drawImage(video,0,0,w,h,0,0,dw,dh);
  const px=detectionCtx.getImageData(0,0,640,640).data,n=640*640,data=new Float32Array(3*n);
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=(px[4*i+2-c]-MEAN[2-c])/STD[2-c];
  const input=new state.ort.Tensor('float32',data,[1,3,640,640]);
  const out=await state.detector.run({[state.detector.inputNames[0]]:input});input.dispose();
  const values=Object.values(out),det=values.find(t=>t.dims.at(-1)===5),labels=values.find(t=>t!==det&&t.data.length===det?.data.length/5);
  if(!det)throw Error(`人物検出器の出力形式が不正: ${values.map(t=>t.dims)}`);
  const boxes=[];
  for(let i=0;i<det.data.length/5;i++){
    const a=det.data,score=a[i*5+4];if(score<.35||(labels&&Number(labels.data[i])!==0))continue;
    boxes.push([clamp(a[i*5]/scale,0,w),clamp(a[i*5+1]/scale,0,h),clamp(a[i*5+2]/scale,0,w),clamp(a[i*5+3]/scale,0,h),score]);
  }
  values.forEach(t=>t.dispose());return selectBox(boxes,state.bbox);
}
async function chooseROI(){
  const mode=$('roi-mode').value,w=video.videoWidth,h=video.videoHeight;
  if(mode==='manual')return state.manual?personROI(state.manual,num('roi-padding')):null;
  if(mode==='full')return personROI([0,0,w,h],num('roi-padding'));
  if(!state.bbox||state.frame%num('detect-interval')===0){
    const found=await detectPerson();if(found)state.bbox=found;else if(!state.bbox)return null;
  }
  return state.bbox?personROI(state.bbox,num('roi-padding')):null;
}
function drawInput(){
  if(!video.videoWidth)return;
  // Preview is bounded; the 224px inference crop still samples the original video.
  const scale=Math.min(1,1280/Math.max(video.videoWidth,video.videoHeight)),w=Math.round(video.videoWidth*scale),h=Math.round(video.videoHeight*scale);
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);ctx.drawImage(video,0,0,w,h);ctx.setTransform(w/video.videoWidth,0,0,h/video.videoHeight,0,0);
  const s=Math.max(video.videoWidth,video.videoHeight)/1000;
  if(state.roi){const r=state.roi;ctx.strokeStyle='#d7ef91';ctx.lineWidth=1.4*s;ctx.setLineDash([7*s,5*s]);ctx.strokeRect(r.x,r.y,r.width,r.height);ctx.setLineDash([]);}
  if(state.manual&&$('roi-mode').value==='manual'&&!state.points){const b=state.manual;ctx.strokeStyle='#cde9a0';ctx.lineWidth=2*s;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
  const points=displayPoints();if(!points)return;
  const valid=i=>points[i]?.score>=state.threshold&&points[i].inFrame;
  for(const[edges,color,width]of [[BODY_EDGES,'#d6f69b',2.2],[HAND_EDGES,'#70e4c6',1.2],[FACE_EDGES,'#efcda1',.7]]){
    if(edges===HAND_EDGES&&!$('show-hands').checked||edges===FACE_EDGES&&!$('show-face').checked)continue;
    ctx.strokeStyle=color;ctx.lineWidth=Math.max(1,width*s);
    for(const[a,b]of edges){if(!valid(a)||!valid(b))continue;ctx.beginPath();ctx.moveTo(points[a].x,points[a].y);ctx.lineTo(points[b].x,points[b].y);ctx.stroke();}
  }
  for(const p of points){if(!valid(p.id)||FACE_IDS.has(p.id)&&!$('show-face').checked||HAND_IDS.has(p.id)&&!$('show-hands').checked)continue;const group=groupOf(p.id);ctx.fillStyle=group==='body'?'#e4f9b9':group==='face'?'#f3d6b0':'#91f3d7';ctx.beginPath();ctx.arc(p.x,p.y,Math.max(1,(group==='body'?3:1.5)*s),0,Math.PI*2);ctx.fill();}
}
function displayPoints(){return state.displayPoints||state.points;}
function updatePoseUI(){
  const p=state.points;drawInput();
  view.update(displayPoints(),state.distance,{threshold:state.threshold,showHands:$('show-hands').checked,showFace:$('show-face').checked,worldOrigin:state.worldOrigin});
  $('pose-empty').hidden=!!p;const count=ids=>p?p.filter(p=>p.inFrame&&p.valid&&p.score>=state.threshold&&(!ids||ids.has(p.id))).length:0;
  $('joint-count').textContent=p?count():'—';$('body-count').textContent=`${p?count(BODY_IDS):'—'} / 23`;$('face-count').textContent=`${p?count(FACE_IDS):'—'} / 68`;$('hand-count').textContent=`${p?count(HAND_IDS):'—'} / 42`;
  $('confidence-info').textContent=POSE.is3D?'モデルの相対Z出力 · 絶対距離は別途推定':'2D信頼度の閾値以上 · 奥行きは模擬推定';
  $('distance').textContent=state.distance?state.distance.depth.toFixed(2):'—';
  $('distance-quality').textContent=state.distance?`${POSE.is3D?'モデル相対Z':'模擬3D'} · 骨長残差 ${(state.distance.boneResidual*100).toFixed(1)}% · ${state.distance.floorConstraint?'接地拘束あり':'接地拘束なし'}`:(state.calibration?'有効な骨が不足 / 推定保留':'絶対距離は未校正');
  $('tracking-state').hidden=!video.videoWidth;$('tracking-state').textContent=p?`${count()} / 133 点`:$('roi-mode').value==='manual'&&!state.manual?'ROI指定待ち':'推論待ち';
}
async function inferFrame(record=true){
  if(state.busy||!state.session||!video.videoWidth||video.readyState<2)return null;
  state.busy=true;enableControls();const start=performance.now(),version=state.videoVersion;
  try{
    const roi=await chooseROI(),afterDetect=performance.now();if(version!==state.videoVersion)return null;
    if(!roi){state.points=null;state.displayPoints=null;state.roi=null;state.distance=null;updatePoseUI();return null;}
    const mediaTime=video.currentTime;state.roi=roi;cropCtx.fillStyle='#000';cropCtx.fillRect(0,0,POSE.width,POSE.height);
    // drawImage clips the source at boundaries; a transform preserves virtual, offscreen ROI padding.
    cropCtx.save();cropCtx.setTransform(POSE.width/roi.width,0,0,POSE.height/roi.height,-roi.x*POSE.width/roi.width,-roi.y*POSE.height/roi.height);cropCtx.drawImage(video,0,0);cropCtx.restore();
    const data=normalizeRGBA(cropCtx.getImageData(0,0,POSE.width,POSE.height).data,POSE.width,POSE.height);
    const afterPre=performance.now();
    const out=await runPose(data);
    const afterInfer=performance.now();
    let prediction;try{prediction={points:decode2D(out,roi,video.videoWidth,video.videoHeight)};}finally{Object.values(out).forEach(t=>t.dispose());}const points=prediction.points;
    if(version!==state.videoVersion)return null;
    const dt=state.lastMediaTime===null?1/60:mediaTime-state.lastMediaTime;
    if(dt<=0||dt>.5){filter.reset();state.previousDistance=null;}
    const solution=solvePose(points,dt);state.previousDistance=solution;state.distance=state.calibration?solution:null;
    state.points=points;state.prediction=prediction;state.poseMediaTime=mediaTime;
    state.displayPoints=$('smooth').checked?points.map(p=>({...p,x:filter.filter(`${p.id}x`,p.x,mediaTime),y:filter.filter(`${p.id}y`,p.y,mediaTime),relativeDisplay:p.relativeDisplay.map((v,c)=>v===null?null:filter.filter(`${p.id}r${c}`,v,mediaTime))})):points;
    if(state.distance&&state.worldOrigin===null)state.worldOrigin=state.distance.worldRootPosition[2];
    if($('roi-mode').value==='auto'){const next=bboxFromPoints(points,state.threshold,video.videoWidth,video.videoHeight);state.bbox=next;}
    state.lastMediaTime=mediaTime;state.frame++;updatePoseUI();
    const end=performance.now(),timing={frame:state.frame,mediaTime,backend:state.backend,detect:afterDetect-start,preprocess:afterPre-afterDetect,pose:afterInfer-afterPre,postprocess:end-afterInfer,total:end-start};
    if(record){
      state.timings.push(timing);state.records.push({frame:state.frame,timeSeconds:mediaTime,source:{width:video.videoWidth,height:video.videoHeight},roi:{...roi},backend:state.backend,
        depthMethod:solution?.method??(POSE.is3D?'model-root-relative-z':null),modelDimensions:POSE.is3D?3:2,depthAmbiguous:true,scaleCalibrated:!!state.calibration,
        points:points.map(p=>({id:p.id,name:p.name,x:p.x,y:p.y,jointRelativeZ:p.relativeZ,relativeModel:p.relativeModel,relativeDisplay:p.relativeDisplay,predictedCameraPosition:p.predictedCameraPosition,
          cameraPosition:state.distance?.camera[p.id]??null,worldPosition:state.distance?.world[p.id]??null,score:p.rawScore,confidenceAvailable:true,depthSource:p.depthSource,inFrame:p.inFrame,missing:!p.inFrame||!p.valid||p.score<state.threshold})),
        rootCameraPosition:state.distance?.rootCameraPosition??null,worldRootPosition:state.distance?.worldRootPosition??null,
        calibration:state.calibration?{kind:state.calibration.kind,quality:state.distance?.quality??'missing',uncertaintyM:state.distance?.uncertaintyM??null}:null,timing});
      updateMetrics(timing);
    }
    return timing;
  }finally{state.busy=false;enableControls();}
}
function updateMetrics(t){
  state.frameTimes.push(t.total);if(state.frameTimes.length>60)state.frameTimes.shift();
  const now=performance.now();state.wallMs=now-state.startTime;
  const avg=state.frameTimes.reduce((a,b)=>a+b,0)/state.frameTimes.length;
  const throughput=1000/avg,elapsed=state.wallMs/1000;
  const liveFPS=state.records.length/Math.max(.001,elapsed);
  const measured=state.running?liveFPS:throughput;
  $('fps').textContent=measured.toFixed(1);$('latency').textContent=t.pose.toFixed(1);
  $('timing-detail').textContent=`全体 ${t.total.toFixed(1)} · 検出 ${t.detect.toFixed(1)} · 前処理 ${t.preprocess.toFixed(1)} ms`;
  $('realtime-badge').textContent=state.backend==='wasm'?'CPU 診断':measured>=59.5?'60 fps相当':`目標 60 fps`;
  $('record-count').textContent=state.records.length.toLocaleString();$('export-json').disabled=$('export-csv').disabled=false;
  const spark=$('fps-spark'),c=spark.getContext('2d');c.clearRect(0,0,spark.width,spark.height);c.beginPath();c.strokeStyle='#9cb67c';c.lineWidth=1.2;
  state.frameTimes.forEach((v,i)=>{const x=i/Math.max(1,state.frameTimes.length-1)*spark.width,y=spark.height-clamp(1000/v/100,0,1)*spark.height;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
}
function resetTracking(){state.videoVersion++;state.bbox=null;state.roi=null;state.points=null;state.prediction=null;state.poseMediaTime=null;state.displayPoints=null;state.distance=null;state.previousDistance=null;state.lastMediaTime=null;filter.reset();updatePoseUI();}
function stop(){state.running=false;state.token++;state.waitFrame?.();video.pause();$('run').innerHTML='推論を開始 <span>▶</span>';document.body.classList.remove('running');enableControls();updateDiagnostics();}
function waitForVideoFrame(){
  return new Promise(resolve=>{
    let id,timer;const done=()=>{if(state.waitFrame!==done)return;state.waitFrame=null;clearTimeout(timer);if(video.cancelVideoFrameCallback&&id!==undefined)video.cancelVideoFrameCallback(id);resolve();};state.waitFrame=done;
    if(video.requestVideoFrameCallback)id=video.requestVideoFrameCallback(done);
    // Resolve on paused/ended sources and browsers without frame callbacks, too.
    timer=setTimeout(done,video.requestVideoFrameCallback?100:8);
  });
}
function presentVideo(){
  if(state.renderFrame!==null)return;
  const tick=()=>{state.renderFrame=null;if(!video.videoWidth||video.paused||video.ended)return;
    if(state.previewTime!==video.currentTime){state.previewTime=video.currentTime;drawInput();}
    state.renderFrame=requestAnimationFrame(tick);
  };state.renderFrame=requestAnimationFrame(tick);
}
async function run(){
  if(state.running){stop();return;}if(!state.session||!video.videoWidth||state.busy)return;
  if($('roi-mode').value==='manual'&&!state.manual){error('入力映像で人物を囲むようにドラッグしてください。','人物の範囲を指定してください');return;}
  state.running=true;state.token++;const token=state.token;state.records=[];state.timings=[];state.frameTimes=[];state.frame=0;state.startTime=performance.now();state.wallMs=0;
  document.body.classList.add('running');$('run').innerHTML='推論を停止 <span>■</span>';enableControls();$('run').disabled=false;
  if(video.currentTime>=video.duration-.03)video.currentTime=0;
  const offline=$('execution-mode').value==='offline';
  try{
    if(offline){
      video.pause();const fps=num('target-fps'),start=video.currentTime,end=video.duration;
      for(let t=start;t<end-.0001&&state.running&&token===state.token;t+=1/fps){await seekVideo(t);await inferFrame();state.mediaSeconds=video.currentTime-start;await new Promise(r=>setTimeout(r,0));}
    }else{
      await video.play();let previous=-1;
      while(state.running&&token===state.token&&!video.ended){
        if(video.paused)break;
        if(video.currentTime!==previous){previous=video.currentTime;await inferFrame();}
        else await waitForVideoFrame();
        // No unconditional display-frame delay after inference; never queue old frames.
      }
    }
  }catch(e){error(String(e.message||e),'推論中に問題が発生しました');}
  finally{if(token===state.token)stop();}
}
async function seekVideo(t){
  const target=clamp(t,0,Math.max(0,video.duration-.000001));
  if(Math.abs(video.currentTime-target)<.00001&&video.readyState>=2)return;
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{cleanup();reject(Error('動画のシークがタイムアウトしました'));},15000);const cleanup=()=>{clearTimeout(timer);video.removeEventListener('seeked',done);video.removeEventListener('error',bad);};const done=()=>{cleanup();resolve();};const bad=()=>{cleanup();reject(Error('この動画をデコードできません'));};video.addEventListener('seeked',done,{once:true});video.addEventListener('error',bad,{once:true});video.currentTime=target;});
}
async function openVideo(file){
  stop();state.videoVersion++;state.manual=null;resetTracking();state.records=[];state.timings=[];state.calibration=null;state.worldOrigin=null;view.world=false;$('relative-view').classList.add('active');$('world-view').classList.remove('active');$('calibration-badge').textContent='未校正';$('record-count').textContent='0';$('export-json').disabled=$('export-csv').disabled=true;
  if(state.sourceURL)URL.revokeObjectURL(state.sourceURL);state.sourceURL=URL.createObjectURL(file);video.src=state.sourceURL;
  await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=()=>reject(Error('この動画をブラウザでデコードできません。H.264 MP4またはWebMで保存し直してください（MOV/HEVC対応はOSに依存します）。'));video.load();});
  $('dropzone').hidden=true;$('source-tag').hidden=false;$('filename').textContent=file.name;$('source-resolution').textContent=`${video.videoWidth} × ${video.videoHeight}`;canvas.classList.add('loaded');$('timeline').max=video.duration;$('duration').textContent=formatTime(video.duration);drawInput();updateROIHint();enableControls();
  log(`Video loaded ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(3)}s`);
  if(state.session)await inferFrame(false);
}
function updateROIHint(){$('roi-hint').hidden=$('roi-mode').value!=='manual'||!video.videoWidth;}
$('video-file').onchange=async e=>{if(e.target.files[0])try{await openVideo(e.target.files[0]);}catch(e){error(e.message);}e.target.value='';};
for(const event of['dragover','dragenter'])$('video-stage').addEventListener(event,e=>{e.preventDefault();$('video-stage').classList.add('dragover');});
for(const event of['dragleave','drop'])$('video-stage').addEventListener(event,e=>{e.preventDefault();$('video-stage').classList.remove('dragover');});
$('video-stage').addEventListener('drop',async e=>{const file=e.dataTransfer.files[0];if(file)try{await openVideo(file);}catch(e){error(e.message);}});
$('sample-button').onclick=async()=>{
  const b=$('sample-button');b.disabled=true;b.textContent='サンプルを取得中…';
  try{const r=await fetch('./sample.mp4');if(!r.ok)throw Error(`HTTP ${r.status}`);await openVideo(new File([await r.blob()],'サンプル動画.mp4',{type:'video/mp4'}));}
  catch(e){error(`サンプル動画を取得できません: ${e.message}`);}finally{b.disabled=false;b.textContent='サンプル動画で試す ↗';}
};
$('load-model').onclick=prepareModel;$('run').onclick=run;
$('step').onclick=async()=>{video.pause();try{await seekVideo(video.currentTime+1/num('target-fps'));if(!state.startTime)state.startTime=performance.now();await inferFrame();}catch(e){error(e.message);}};
$('play').onclick=async()=>{if(video.paused){try{await video.play();}catch(e){error(e.message);}}else{if(state.running)stop();else video.pause();}};
video.addEventListener('play',()=>{$('play').textContent='Ⅱ';$('play').ariaLabel='動画を一時停止';presentVideo();});video.addEventListener('pause',()=>{$('play').textContent='▶';$('play').ariaLabel='動画を再生';state.waitFrame?.();drawInput();});
video.addEventListener('timeupdate',()=>{$('current-time').textContent=formatTime(video.currentTime);$('timeline').value=video.currentTime;if(!state.running&&!state.busy)drawInput();});
video.addEventListener('ended',()=>{if(state.running)stop();});
$('timeline').oninput=()=>{if(state.running)stop();video.currentTime=Number($('timeline').value);resetTracking();};
video.addEventListener('seeked',()=>{if(!state.running){resetTracking();drawInput();}});
$('speed').onchange=()=>video.playbackRate=num('speed');
$('roi-mode').onchange=()=>{resetTracking();updateROIHint();};
for(const id of['roi-padding','detect-interval'])$(id).onchange=resetTracking;
$('confidence').oninput=()=>{state.threshold=num('confidence');$('confidence-value').textContent=state.threshold.toFixed(2);updatePoseUI();};
for(const id of['show-face','show-hands','smooth'])$(id).onchange=updatePoseUI;
$('wasm-mode').onchange=()=>{state.needsReload=true;$('backend').textContent=$('wasm-mode').checked?'WASM · 再準備が必要':'WebGPU · 再準備が必要';enableControls();$('model-status').textContent='実行方式の変更後はモデルを再準備してください';};
$('local-model').onchange=async e=>{if(e.target.files[0]){state.localModel=e.target.files[0];await prepareModel();}e.target.value='';};
$('clear-cache').onclick=async()=>{if(state.running||state.loading){error('キャッシュの削除は推論を停止してから行ってください。');return;}await caches.delete(MODEL_CACHE);await refreshCache();log('Model cache cleared by user');};
$('calibrate').onclick=()=>{
  if(!video.videoWidth){error('校正する動画を先に選択してください。');return;}
  try{state.calibration=calibrationFromValues({height:num('height'),fov:num('fov'),fx:num('fx'),fy:num('fy'),cx:num('cx'),cy:num('cy'),cameraHeight:num('camera-height'),cameraPitch:num('camera-pitch'),thigh:num('thigh'),shin:num('shin')},video.videoWidth,video.videoHeight);
    state.previousDistance=null;state.worldOrigin=null;$('calibration-badge').textContent='単眼・概算';$('calibration-note').textContent='骨長と相対Zから推定。床座標は入力したカメラ位置・角度に依存します。';
    if(state.points){state.distance=solvePose(state.points,1/60,null);state.previousDistance=state.distance;state.displayPoints=null;if(state.distance)state.worldOrigin=state.distance.worldRootPosition[2];}updatePoseUI();log(`Calibration enabled: ${state.calibration.kind}`);
  }catch(e){error(e.message);}
};
for(const id of['height','fov','fx','fy','cx','cy','camera-height','camera-pitch','thigh','shin'])$(id).addEventListener('input',()=>{if(state.calibration){state.calibration=null;state.distance=null;state.previousDistance=null;$('calibration-badge').textContent='再校正が必要';updatePoseUI();}});
$('relative-view').onclick=()=>{view.world=false;$('relative-view').classList.add('active');$('world-view').classList.remove('active');$('depth-state').textContent='骨盤基準の相対奥行き';view.draw();};
$('world-view').onclick=()=>{if(!state.calibration||!state.distance){error('有効な距離推定が必要です。身長と画角を設定して校正してください。');return;}view.world=true;$('relative-view').classList.remove('active');$('world-view').classList.add('active');$('depth-state').textContent='校正した床の座標 · 単眼概算';view.draw();};
$('reset-view').onclick=()=>view.reset();
let dragStart=null;
function imagePosition(e){const r=canvas.getBoundingClientRect();return[clamp((e.clientX-r.left)/r.width*video.videoWidth,0,video.videoWidth),clamp((e.clientY-r.top)/r.height*video.videoHeight,0,video.videoHeight)];}
canvas.addEventListener('pointerdown',e=>{if($('roi-mode').value!=='manual'||!video.videoWidth)return;if(state.running)stop();resetTracking();dragStart=imagePosition(e);canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragStart)return;const p=imagePosition(e);state.manual=[Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])];state.points=null;state.displayPoints=null;state.roi=null;drawInput();});
canvas.addEventListener('pointerup',async()=>{if(!dragStart)return;dragStart=null;if(state.manual&&state.manual[2]-state.manual[0]>8&&state.manual[3]-state.manual[1]>8){try{await inferFrame(false);}catch(e){error(e.message);}}else{state.manual=null;error('人物の範囲を少し大きく指定してください。');}});
canvas.addEventListener('pointercancel',()=>dragStart=null);
function download(name,body,type){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function benchmark(){const ts=state.timings;return{schema:'vrc-rtmw2d-benchmark/v2',createdAt:new Date().toISOString(),backend:state.backend,mode:$('execution-mode').value,
  model:{...MODELS.pose,actual:state.modelInfo},jointLayout:'COCO-WholeBody133',jointNames:JOINT_NAMES_FOR_EXPORT(),detector:state.detectorInfo,runtime:RUNTIME_VERSION,preferredLayout:state.layout,gpuBufferReuse:!!state.runner,graphCapture:false,numericalProbe:state.probeDeltas,modelDimensions:POSE.is3D?3:2,relativeDepthSource:POSE.is3D?'model':'constrained-estimation',source:{width:video.videoWidth,height:video.videoHeight,durationSeconds:video.duration||null},
  frames:ts.length,wallMs:state.wallMs,processedFPS:ts.length/(state.wallMs/1000||1),poseMs:{p50:percentile(ts.map(t=>t.pose),.5),p95:percentile(ts.map(t=>t.pose),.95)},
  totalMs:{p50:percentile(ts.map(t=>t.total),.5),p95:percentile(ts.map(t=>t.total),.95)},warmupExcludedMs:state.warmupMs,outputs:state.warmupShape,calibration:state.calibration,
  notes:['GPU memory not measured','RTX 5060 Ti 8GB not validated by this browser benchmark','Depth source follows selected model: RTMW 2D uses constrained estimation, RTMW3D uses model relative-Z; absolute distance remains estimated','Uncalibrated preview assumes entered height/FOV; absolute camera/world fields remain null','Only calibrated camera/world values use entered camera geometry','No camera API / no upload endpoint'],timings:ts};}
function JOINT_NAMES_FOR_EXPORT(){return state.points?.map(p=>p.name)??null;}
$('export-benchmark').onclick=()=>download('vrc-mmpose-benchmark.json',JSON.stringify(benchmark(),null,2),'application/json');
$('export-json').onclick=()=>download('vrc-rtmw2d-poses.json',JSON.stringify({schema:'vrc-rtmw2d-poses/v2',benchmark:benchmark(),frames:state.records}),'application/json');
$('export-csv').onclick=()=>{
  const lines=['frame,time_seconds,joint_id,joint_name,x_px,y_px,joint_relative_z_m,raw_score,confidence_available,in_frame,missing,model_local_x_m,model_local_y_m,model_local_z_m,predicted_camera_x_m,predicted_camera_y_m,predicted_camera_z_m,calibrated_camera_x_m,calibrated_camera_y_m,calibrated_camera_z_m,world_x_m,world_y_m,world_z_m,backend'];
  for(const f of state.records)for(const p of f.points)lines.push([f.frame,f.timeSeconds,p.id,p.name,p.x,p.y,p.jointRelativeZ,p.score,true,p.inFrame,p.missing,...p.relativeModel,...(p.predictedCameraPosition||['','','']),...(p.cameraPosition||['','','']),...(p.worldPosition||['','','']),f.backend].join(','));
  download('vrc-rtmw2d-poses.csv','\uFEFF'+lines.join('\r\n'),'text/csv;charset=utf-8');
};
window.addEventListener('unhandledrejection',e=>{error(String(e.reason?.message||e.reason));});
if('serviceWorker'in navigator)navigator.serviceWorker.register('./sw.js').catch(e=>log(`Service worker: ${e.message}`));
for(const variant of MODEL_VARIANTS){const option=document.createElement('option');option.value=variant.id;option.textContent=variant.name+' · '+mb(variant.size)+(variant.smallFixtureDriftPassed?'':' · 基準との差大');$('model-select').append(option);}
$('model-select').value=MODELS.pose.id;
$('model-select').onchange=()=>{stop();MODELS.pose=MODEL_VARIANTS.find(v=>v.id===$('model-select').value);configureSelectedModel();state.localModel=null;state.needsReload=true;resetTracking();$('model-name').textContent=MODELS.pose.name;$('model-description').textContent=`${POSE.height} × ${POSE.width} · 133点 · ${POSE.is3D?'モデル3D':'2D＋模擬3D'}`;$('model-status').textContent='モデルを準備してください';enableControls();};
$('depth-direction').onchange=()=>{state.previousDistance=null;};
function configureSelectedModel(){configurePose(MODELS.pose);crop.width=POSE.width;crop.height=POSE.height;$('model-name').textContent=MODELS.pose.name;$('model-description').textContent=`${POSE.height} × ${POSE.width} · 133点 · ${POSE.is3D?'モデル3D':'2D＋模擬3D'}`;$('download-size').textContent=mb(MODELS.pose.size);$('depth-direction').disabled=!!POSE.is3D;$('depth-state').textContent=POSE.is3D?'モデル出力の骨盤相対Z':'骨長・時系列からの模擬3D';}
configureSelectedModel();
updateDiagnostics();refreshCache();enableControls();
