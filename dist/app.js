import {POSE,MEAN,STD,BODY_EDGES,HAND_EDGES,FACE_EDGES,clamp,makeROI,normalizeRGBA,decodeSimCC,bboxFromPoints,selectBox,calibrationFromValues,estimateDistance,OneEuro,percentile}from './core.js';
import {MODELS,MODEL_CACHE,loadModel,cacheSummary,validateModel}from './model-cache.js';
import {PoseView}from './view3d.js';

const $=id=>document.getElementById(id),video=$('source-video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
const view=new PoseView($('pose3d')),crop=document.createElement('canvas'),cropCtx=crop.getContext('2d',{willReadFrequently:true});
crop.width=POSE.width;crop.height=POSE.height;
const detectionCanvas=document.createElement('canvas'),detectionCtx=detectionCanvas.getContext('2d',{willReadFrequently:true});
detectionCanvas.width=detectionCanvas.height=640;
const frameCanvas=document.createElement('canvas'),frameCtx=frameCanvas.getContext('2d');
const filter=new OneEuro();
const state={ort:null,session:null,detector:null,backend:'webgpu',busy:false,loading:false,running:false,sourceURL:null,videoVersion:0,token:0,
  bbox:null,manual:null,roi:null,points:null,distance:null,previousDistance:null,calibration:null,worldOrigin:null,
  records:[],timings:[],modelInfo:null,detectorInfo:null,frame:0,frameTimes:[],lastMediaTime:null,threshold:.3,
  startTime:0,wallMs:0,mediaSeconds:0,warmupMs:null,warmupShape:null,logs:[],detectorError:null,localModel:null,needsReload:false};
const num=id=>$(id).value.trim()===''?null:Number($(id).value);
const formatTime=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const mb=n=>`${(n/1e6).toFixed(1)} MB`;
function log(message){state.logs.push(`${new Date().toISOString()} ${message}`);if(state.logs.length>100)state.logs.shift();updateDiagnostics();}
function error(message,title='処理を確認してください'){$('error-title').textContent=title;$('error-message').textContent=message;$('error-panel').hidden=false;log(`ERROR ${message}`);}
$('dismiss-error').onclick=()=>$('error-panel').hidden=true;
function updateDiagnostics(){
  $('diagnostics').textContent=[`Model: ${MODELS.pose.repo}@${MODELS.pose.revision}`,`Backend: ${state.backend} / ORT 1.24.3`,
    `WebGPU: ${navigator.gpu?'APIあり':'APIなし'} / secureContext: ${isSecureContext}`,
    `Source: ${video.videoWidth||0} × ${video.videoHeight||0} / pixel coordinates before crop`,
    state.modelInfo?`SHA-256: ${state.modelInfo.sha256}\nCache source: ${state.modelInfo.source}`:'',
    state.session?`Inputs: ${state.session.inputNames.join(', ')}\nOutputs: ${state.session.outputNames.join(', ')}\nOutput shape: ${JSON.stringify(state.warmupShape)}\nWarm-up: ${state.warmupMs?.toFixed(1)} ms (計測から除外)`:'',
    `Calibration: ${state.calibration?.kind||'none'} / no fixed root depth / no floor clamping`,
    `Records: ${state.records.length} / GPU memory: browser APIから未計測`,
    ...state.logs.slice(-12)].filter(Boolean).join('\n');
}
async function refreshCache(){const s=await cacheSummary();$('cache-summary').textContent=s.error?'キャッシュを開けません':`${s.count}モデル保存済み${s.storage?.usage?` · 使用 ${mb(s.storage.usage)}`:''}`;return s;}
function enableControls(){const loaded=!!video.videoWidth,ready=!!state.session&&!state.loading&&!state.needsReload;$('run').disabled=!(loaded&&ready);$('step').disabled=!(loaded&&ready)||state.running||state.busy;$('play').disabled=!loaded;$('timeline').disabled=!loaded;$('load-model').disabled=state.loading||state.running||state.busy;$('wasm-mode').disabled=state.loading||state.running||state.busy;$('local-model').disabled=state.loading||state.running||state.busy;}

async function initRuntime(){
  if(state.ort)return state.ort;
  const ort=await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
  ort.env.wasm.numThreads=crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||1):1;
  ort.env.wasm.proxy=false;ort.env.logLevel='warning';state.ort=ort;return ort;
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
    if(state.session){await state.session.release();state.session=null;}
    if(state.detector){await state.detector.release();state.detector=null;}
    let model;
    if(state.localModel){const bytes=new Uint8Array(await state.localModel.arrayBuffer());model={bytes,sha256:await validateModel(bytes,MODELS.pose),source:'local-file'};}
    else model=await loadModel(MODELS.pose,p=>progress('RTMW3D-X',p));
    state.modelInfo={sha256:model.sha256,source:model.source,bytes:model.bytes.byteLength,warning:model.warning};
    $('model-status').textContent='WebGPUセッションを構築中';
    state.session=await ort.InferenceSession.create(model.bytes,{executionProviders:[state.backend],graphOptimizationLevel:'all',enableMemPattern:true});
    model.bytes=null;
    // Run a real inference, not just a download or session creation.
    $('model-status').textContent='実モデルで初回推論を確認中';
    const tensor=new ort.Tensor('float32',new Float32Array(3*POSE.width*POSE.height),[1,3,POSE.height,POSE.width]);
    const warmStart=performance.now(),out=await state.session.run({[state.session.inputNames[0]]:tensor});
    state.warmupMs=performance.now()-warmStart;state.warmupShape=Object.fromEntries(Object.entries(out).map(([k,v])=>[k,v.dims]));
    const axes=poseAxes(out);decodeSimCC(...axes,{x:0,y:0,width:288,height:384},288,384);
    if(Object.values(out).some(t=>t.data.some(v=>!Number.isFinite(v))))throw Error('初回推論で非有限値を検出しました');
    Object.values(out).forEach(t=>t.dispose());tensor.dispose();log(`Warm-up succeeded ${state.warmupMs.toFixed(1)}ms ${JSON.stringify(state.warmupShape)}`);
    try{await prepareDetector();}catch(e){state.detectorError=e.message;log(`Detector: ${e.message}`);$('roi-mode').value='manual';error(`姿勢モデルは推論確認済みです。自動人物検出器の準備に失敗しました: ${e.message}\n動画上で人物の範囲をドラッグすると推論できます。`,'手動ROIで続行できます');}
    state.needsReload=false;$('model-status').textContent=`推論確認済み · ${state.backend==='webgpu'?'WebGPU':'WASM（CPU診断）'}`;$('download-size').textContent='SHA-256 一致';$('model-progress').value=100;$('load-model').innerHTML='再読み込み <span>↻</span>';
    if(state.modelInfo.warning)error(state.modelInfo.warning,'推論は実行できます');
    if(video.videoWidth)await inferFrame(false);
  }catch(e){
    if(state.session){await state.session.release().catch(()=>{});state.session=null;}
    $('model-status').textContent='準備中に問題が発生';error(`${String(e.message||e)}\n再試行、または同じモデルのローカルONNXを指定できます。`,'モデル準備を完了できませんでした');
  }finally{state.loading=false;enableControls();await refreshCache();updateROIHint();updateDiagnostics();}
}
async function prepareDetector(){
  const ort=await initRuntime(),model=await loadModel(MODELS.detector,p=>progress('RTMDet-s',p));
  state.detectorInfo={sha256:model.sha256,source:model.source,bytes:model.bytes.byteLength};
  // RTMDet's exported NMS is executed in WASM; pose inference stays on the selected GPU.
  state.detector=await ort.InferenceSession.create(model.bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  model.bytes=null;state.detectorError=null;log(`Detector loaded: ${state.detector.inputNames} → ${state.detector.outputNames} (WASM / NMS)`);
}
function poseAxes(outputs){
  const entries=Object.entries(outputs),named=axis=>entries.find(([name])=>new RegExp(`(^|_)${axis}($|_)|simcc_${axis}|simcc${axis}`,'i').test(name))?.[1];
  const ordered=state.session.outputNames.map(n=>outputs[n]);
  return ['x','y','z'].map((a,i)=>named(a)||ordered[i]);
}
async function detectPerson(){
  if(!state.detector)throw Error('人物検出器を準備してください。手動ROIも使用できます。');
  const w=frameCanvas.width,h=frameCanvas.height,scale=Math.min(640/w,640/h),dw=Math.round(w*scale),dh=Math.round(h*scale);
  // MMDeploy RTMDet uses top-left letterboxing, padding 114, BGR normalization.
  detectionCtx.fillStyle='rgb(114,114,114)';detectionCtx.fillRect(0,0,640,640);detectionCtx.drawImage(frameCanvas,0,0,w,h,0,0,dw,dh);
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
  if(mode==='manual')return state.manual?makeROI(state.manual,POSE.width,POSE.height,num('roi-padding')):null;
  if(mode==='full')return makeROI([0,0,w,h],POSE.width,POSE.height,1);
  if(!state.bbox||state.frame%num('detect-interval')===0){
    const found=await detectPerson();if(found)state.bbox=found;else if(!state.bbox)return null;
  }
  return state.bbox?makeROI(state.bbox,POSE.width,POSE.height,num('roi-padding')):null;
}
function drawInput(){
  if(!video.videoWidth)return;
  if(canvas.width!==video.videoWidth||canvas.height!==video.videoHeight){canvas.width=video.videoWidth;canvas.height=video.videoHeight;frameCanvas.width=canvas.width;frameCanvas.height=canvas.height;}
  const source=state.running&&frameCanvas.width&&state.points?frameCanvas:video;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(source,0,0,canvas.width,canvas.height);
  const s=Math.max(canvas.width,canvas.height)/1000;
  if(state.roi){const r=state.roi;ctx.strokeStyle='#d7ef91';ctx.lineWidth=1.4*s;ctx.setLineDash([7*s,5*s]);ctx.strokeRect(r.x,r.y,r.width,r.height);ctx.setLineDash([]);}
  if(state.manual&&$('roi-mode').value==='manual'&&!state.points){const b=state.manual;ctx.strokeStyle='#cde9a0';ctx.lineWidth=2*s;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
  const points=displayPoints();if(!points)return;
  const valid=i=>points[i]?.score>=state.threshold&&points[i].inFrame;
  for(const[edges,color,width]of [[BODY_EDGES,'#d6f69b',2.2],[HAND_EDGES,'#70e4c6',1.2],[FACE_EDGES,'#efcda1',.7]]){
    if(edges===HAND_EDGES&&!$('show-hands').checked||edges===FACE_EDGES&&!$('show-face').checked)continue;
    ctx.strokeStyle=color;ctx.lineWidth=Math.max(1,width*s);
    for(const[a,b]of edges){if(!valid(a)||!valid(b))continue;ctx.beginPath();ctx.moveTo(points[a].x,points[a].y);ctx.lineTo(points[b].x,points[b].y);ctx.stroke();}
  }
  for(const p of points){if(!valid(p.id)||p.id>=23&&p.id<91&&!$('show-face').checked||p.id>=91&&!$('show-hands').checked)continue;ctx.fillStyle=p.id<23?'#e4f9b9':p.id<91?'#f3d6b0':'#91f3d7';ctx.beginPath();ctx.arc(p.x,p.y,Math.max(1,(p.id<23?3:1.5)*s),0,Math.PI*2);ctx.fill();}
}
function displayPoints(){return state.displayPoints||state.points;}
function updatePoseUI(){
  const p=state.points;drawInput();
  view.update(displayPoints(),state.distance,{threshold:state.threshold,showHands:$('show-hands').checked,showFace:$('show-face').checked,worldOrigin:state.worldOrigin});
  $('pose-empty').hidden=!!p;const count=(a,b)=>p?p.slice(a,b).filter(p=>p.inFrame&&p.score>=state.threshold).length:0;
  $('joint-count').textContent=p?count(0,133):'—';$('body-count').textContent=`${p?count(0,23):'—'} / 23`;$('face-count').textContent=`${p?count(23,91):'—'} / 68`;$('hand-count').textContent=`${p?count(91,133):'—'} / 42`;
  $('confidence-info').textContent=`閾値 ${state.threshold.toFixed(2)} · 欠測 ${p?133-count(0,133):'—'} 点`;
  $('distance').textContent=state.distance?state.distance.depth.toFixed(2):'—';
  $('distance-quality').textContent=state.distance?`概算 ±${state.distance.uncertaintyM.toFixed(2)} m · ${state.distance.bonesUsed}骨`:(state.calibration?'有効な骨が不足 / 推定保留':'絶対距離は未校正');
  $('tracking-state').hidden=!video.videoWidth;$('tracking-state').textContent=p?`${count(0,133)} / 133 点`:$('roi-mode').value==='manual'&&!state.manual?'ROI指定待ち':'人物を探索中';
}
async function inferFrame(record=true){
  if(state.busy||!state.session||!video.videoWidth||video.readyState<2)return null;
  state.busy=true;enableControls();const start=performance.now(),version=state.videoVersion,mediaTime=video.currentTime;
  try{
    if(frameCanvas.width!==video.videoWidth||frameCanvas.height!==video.videoHeight){frameCanvas.width=video.videoWidth;frameCanvas.height=video.videoHeight;}
    // Snapshot once; inference/overlay and recorded timestamps refer to this same decoded frame.
    frameCtx.drawImage(video,0,0);const roi=await chooseROI(),afterDetect=performance.now();
    if(!roi){state.points=null;state.displayPoints=null;state.roi=null;state.distance=null;updatePoseUI();return null;}
    state.roi=roi;cropCtx.fillStyle='#000';cropCtx.fillRect(0,0,POSE.width,POSE.height);
    // drawImage clips the source at boundaries; a transform preserves virtual, offscreen ROI padding.
    cropCtx.save();cropCtx.setTransform(POSE.width/roi.width,0,0,POSE.height/roi.height,-roi.x*POSE.width/roi.width,-roi.y*POSE.height/roi.height);cropCtx.drawImage(frameCanvas,0,0);cropCtx.restore();
    const data=normalizeRGBA(cropCtx.getImageData(0,0,POSE.width,POSE.height).data,POSE.width,POSE.height);
    const input=new state.ort.Tensor('float32',data,[1,3,POSE.height,POSE.width]),afterPre=performance.now();
    let out;try{out=await state.session.run({[state.session.inputNames[0]]:input});}finally{input.dispose();}
    const afterInfer=performance.now();
    let points;try{points=decodeSimCC(...poseAxes(out),roi,video.videoWidth,video.videoHeight);}finally{Object.values(out).forEach(t=>t.dispose());}
    if(version!==state.videoVersion)return null;
    const dt=state.lastMediaTime===null?1/60:mediaTime-state.lastMediaTime;
    if(dt<=0||dt>.5){filter.reset();state.previousDistance=null;}
    state.points=points;
    state.displayPoints=$('smooth').checked?points.map(p=>({...p,x:filter.filter(`${p.id}x`,p.x,mediaTime),y:filter.filter(`${p.id}y`,p.y,mediaTime),relativeDisplay:p.relativeDisplay.map((v,c)=>v===null?null:filter.filter(`${p.id}r${c}`,v,mediaTime))})):points;
    state.distance=estimateDistance(points,state.calibration,state.threshold,state.previousDistance,dt);state.previousDistance=state.distance;
    if(state.distance&&state.worldOrigin===null)state.worldOrigin=state.distance.worldRootPosition[2];
    if($('roi-mode').value==='auto'){const next=bboxFromPoints(points,state.threshold,video.videoWidth,video.videoHeight);state.bbox=next;}
    state.lastMediaTime=mediaTime;state.frame++;updatePoseUI();
    const end=performance.now(),timing={frame:state.frame,mediaTime,backend:state.backend,detect:afterDetect-start,preprocess:afterPre-afterDetect,pose:afterInfer-afterPre,postprocess:end-afterInfer,total:end-start};
    if(record){
      state.timings.push(timing);state.records.push({frame:state.frame,timeSeconds:mediaTime,source:{width:video.videoWidth,height:video.videoHeight},roi:{...roi},backend:state.backend,
        points:points.map(p=>({id:p.id,x:p.x,y:p.y,jointRelativeZ:p.relativeZ,relativeModel:p.relativeModel,relativeDisplay:p.relativeDisplay,
          cameraPosition:state.distance?.camera[p.id]??null,worldPosition:state.distance?.world[p.id]??null,score:p.rawScore,inFrame:p.inFrame,missing:!p.inFrame||p.score<state.threshold})),
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
function resetTracking(){state.bbox=null;state.roi=null;state.points=null;state.displayPoints=null;state.distance=null;state.previousDistance=null;state.lastMediaTime=null;filter.reset();updatePoseUI();}
function stop(){state.running=false;state.token++;video.pause();$('run').innerHTML='推論を開始 <span>▶</span>';document.body.classList.remove('running');enableControls();updateDiagnostics();}
async function run(){
  if(state.running){stop();return;}if(!state.session||!video.videoWidth)return;
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
        // Never queue frames: slow hardware skips decoded frames in real-time mode.
        await new Promise(r=>requestAnimationFrame(r));
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
  stop();state.videoVersion++;state.manual=null;resetTracking();state.records=[];state.timings=[];state.calibration=null;state.worldOrigin=null;$('calibration-badge').textContent='未校正';$('record-count').textContent='0';$('export-json').disabled=$('export-csv').disabled=true;
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
video.addEventListener('play',()=>{$('play').textContent='Ⅱ';$('play').ariaLabel='動画を一時停止';});video.addEventListener('pause',()=>{$('play').textContent='▶';$('play').ariaLabel='動画を再生';});
video.addEventListener('timeupdate',()=>{$('current-time').textContent=formatTime(video.currentTime);$('timeline').value=video.currentTime;if(!state.running&&!state.busy)drawInput();});
video.addEventListener('ended',()=>{if(state.running)stop();});
$('timeline').oninput=()=>{if(state.running)stop();video.currentTime=Number($('timeline').value);resetTracking();};
video.addEventListener('seeked',()=>{if(!state.running){resetTracking();drawInput();}});
$('speed').onchange=()=>video.playbackRate=num('speed');
$('roi-mode').onchange=()=>{resetTracking();updateROIHint();};
for(const id of['roi-padding','detect-interval'])$(id).onchange=()=>{state.bbox=null;state.roi=null;};
$('confidence').oninput=()=>{state.threshold=num('confidence');$('confidence-value').textContent=state.threshold.toFixed(2);updatePoseUI();};
for(const id of['show-face','show-hands','smooth'])$(id).onchange=updatePoseUI;
$('wasm-mode').onchange=()=>{state.needsReload=true;$('backend').textContent=$('wasm-mode').checked?'WASM · 再準備が必要':'WebGPU · 再準備が必要';enableControls();$('model-status').textContent='実行方式の変更後はモデルを再準備してください';};
$('local-model').onchange=async e=>{if(e.target.files[0]){state.localModel=e.target.files[0];await prepareModel();}e.target.value='';};
$('clear-cache').onclick=async()=>{if(state.running||state.loading){error('キャッシュの削除は推論を停止してから行ってください。');return;}await caches.delete(MODEL_CACHE);await refreshCache();log('Model cache cleared by user');};
$('calibrate').onclick=()=>{
  if(!video.videoWidth){error('校正する動画を先に選択してください。');return;}
  try{state.calibration=calibrationFromValues({height:num('height'),fov:num('fov'),fx:num('fx'),fy:num('fy'),cx:num('cx'),cy:num('cy'),cameraHeight:num('camera-height'),cameraPitch:num('camera-pitch'),thigh:num('thigh'),shin:num('shin')},video.videoWidth,video.videoHeight);
    state.previousDistance=null;state.worldOrigin=null;$('calibration-badge').textContent='単眼・概算';$('calibration-note').textContent='骨長と相対Zから推定。床座標は入力したカメラ位置・角度に依存します。';
    if(state.points){state.distance=estimateDistance(state.points,state.calibration,state.threshold);if(state.distance)state.worldOrigin=state.distance.worldRootPosition[2];}updatePoseUI();log(`Calibration enabled: ${state.calibration.kind}`);
  }catch(e){error(e.message);}
};
for(const id of['height','fov','fx','fy','cx','cy','camera-height','camera-pitch','thigh','shin'])$(id).addEventListener('input',()=>{if(state.calibration){state.calibration=null;state.distance=null;state.previousDistance=null;$('calibration-badge').textContent='再校正が必要';updatePoseUI();}});
$('relative-view').onclick=()=>{view.world=false;$('relative-view').classList.add('active');$('world-view').classList.remove('active');$('depth-state').textContent='骨盤基準の相対奥行き';view.draw();};
$('world-view').onclick=()=>{if(!state.calibration||!state.distance){error('有効な距離推定が必要です。身長と画角を設定して校正してください。');return;}view.world=true;$('relative-view').classList.remove('active');$('world-view').classList.add('active');$('depth-state').textContent='校正した床の座標 · 単眼概算';view.draw();};
$('reset-view').onclick=()=>view.reset();
let dragStart=null;
function imagePosition(e){const r=canvas.getBoundingClientRect();return[clamp((e.clientX-r.left)/r.width*canvas.width,0,canvas.width),clamp((e.clientY-r.top)/r.height*canvas.height,0,canvas.height)];}
canvas.addEventListener('pointerdown',e=>{if($('roi-mode').value!=='manual'||!video.videoWidth)return;if(state.running)stop();dragStart=imagePosition(e);canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragStart)return;const p=imagePosition(e);state.manual=[Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])];state.points=null;state.displayPoints=null;state.roi=null;drawInput();});
canvas.addEventListener('pointerup',async()=>{if(!dragStart)return;dragStart=null;if(state.manual&&state.manual[2]-state.manual[0]>8&&state.manual[3]-state.manual[1]>8){try{await inferFrame(false);}catch(e){error(e.message);}}else{state.manual=null;error('人物の範囲を少し大きく指定してください。');}});
canvas.addEventListener('pointercancel',()=>dragStart=null);
function download(name,body,type){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function benchmark(){const ts=state.timings;return{schema:'vrc-mmpose-benchmark/v1',createdAt:new Date().toISOString(),backend:state.backend,mode:$('execution-mode').value,
  model:{...MODELS.pose,actual:state.modelInfo},detector:state.detectorInfo,runtime:'onnxruntime-web@1.24.3',source:{width:video.videoWidth,height:video.videoHeight,durationSeconds:video.duration||null},
  frames:ts.length,wallMs:state.wallMs,processedFPS:ts.length/(state.wallMs/1000||1),poseMs:{p50:percentile(ts.map(t=>t.pose),.5),p95:percentile(ts.map(t=>t.pose),.95)},
  totalMs:{p50:percentile(ts.map(t=>t.total),.5),p95:percentile(ts.map(t=>t.total),.95)},warmupExcludedMs:state.warmupMs,outputs:state.warmupShape,calibration:state.calibration,
  notes:['GPU memory not measured','RTX 5060 Ti 8GB not validated by this browser benchmark','relativeZ is not absolute camera distance','No camera API / no upload endpoint'],timings:ts};}
$('export-benchmark').onclick=()=>download('vrc-mmpose-benchmark.json',JSON.stringify(benchmark(),null,2),'application/json');
$('export-json').onclick=()=>download('vrc-mmpose-poses.json',JSON.stringify({schema:'vrc-mmpose-poses/v1',benchmark:benchmark(),frames:state.records}),'application/json');
$('export-csv').onclick=()=>{
  const lines=['frame,time_seconds,joint_id,x_px,y_px,joint_relative_z_m,raw_score,in_frame,missing,root_camera_x_m,root_camera_y_m,root_camera_z_m,world_root_x_m,world_root_y_m,world_root_z_m,backend'];
  for(const f of state.records)for(const p of f.points)lines.push([f.frame,f.timeSeconds,p.id,p.x,p.y,p.jointRelativeZ,p.score,p.inFrame,p.missing,...(f.rootCameraPosition||['','','']),...(f.worldRootPosition||['','','']),f.backend].join(','));
  download('vrc-mmpose-poses.csv','\uFEFF'+lines.join('\r\n'),'text/csv;charset=utf-8');
};
window.addEventListener('unhandledrejection',e=>{error(String(e.reason?.message||e.reason));});
if('serviceWorker'in navigator)navigator.serviceWorker.register('./sw.js').catch(e=>log(`Service worker: ${e.message}`));
updateDiagnostics();refreshCache();enableControls();
