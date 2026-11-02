import {MODEL_VARIANTS}from './rtmw-models.js';
import {EXTRA_MODELS}from './extra-models.js';
MODEL_VARIANTS.push(...EXTRA_MODELS);
export {MODEL_VARIANTS};
export const MODEL_CACHE='vrc-mmpose-models-v1';
export const MODELS={
  pose:MODEL_VARIANTS.find(v=>v.id==='rtmpose-s-256x192-fp32'),
  detector:{name:'RTMDet-s',repo:'ziq/rtm',revision:'6bdd0110307167f5f003e92e34099a21bd61283c',path:'rtmdet-s.onnx',size:40309222,sha256:'3bb7246a8b5104dd94e9f9bbfcf92d2cd9e55d344a9a5e8b7b84ebcfeedafe14'}
};
export const modelURL=m=>`https://huggingface.co/${m.repo}/resolve/${m.revision}/${m.path}`;
export async function hashBytes(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');}
export async function validateModel(bytes,model){
  if(bytes.byteLength<1000000)throw Error(`${model.name}: 完全なONNXファイルではありません（${bytes.byteLength} bytes）`);
  if(model.size&&bytes.byteLength!==model.size)throw Error(`${model.name}: ファイル長が一致しません（${bytes.byteLength} / ${model.size} bytes）`);
  const sha256=await hashBytes(bytes);
  if(model.sha256&&sha256!==model.sha256)throw Error(`${model.name}: SHA-256が一致しません。キャッシュを削除して再取得してください`);
  return sha256;
}
export async function loadModel(model,onProgress=()=>{}){
  const key=modelURL(model);let cache=null,warning=null;
  try{cache=await caches.open(MODEL_CACHE);}catch(e){warning=`永続キャッシュを開けません: ${e.message}`;}
  if(cache){const hit=await cache.match(key);if(hit){
    onProgress({phase:'cache',received:model.size,total:model.size});
    const bytes=new Uint8Array(await hit.arrayBuffer());
    try{const sha256=await validateModel(bytes,model);return {bytes,sha256,source:'cache',warning};}
    catch{await cache.delete(key);onProgress({phase:'invalid-cache',received:0,total:model.size});}
  }}
  if(navigator.storage?.persist)await navigator.storage.persist().catch(()=>false);
  let response,lastError;
  for(let attempt=0;attempt<3;attempt++){
    try{response=await fetch(key,{mode:'cors',credentials:'omit',referrerPolicy:'no-referrer',signal:AbortSignal.timeout(300000)});if(!response.ok)throw Error(`HTTP ${response.status}`);break;}
    catch(e){lastError=e;if(attempt===2)throw Error(`${model.name}の取得に失敗: ${lastError.message}。接続を確認して再試行してください。`);await new Promise(r=>setTimeout(r,800*(attempt+1)));}
  }
  const advertised=Number(response.headers.get('content-length')),total=advertised>0?advertised:model.size;
  const reader=response.body.getReader(),chunks=[];let received=0;
  while(true){const{done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.byteLength;onProgress({phase:'download',received,total});}
  const bytes=new Uint8Array(received);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}chunks.length=0;
  onProgress({phase:'verify',received,total:received});
  const sha256=await validateModel(bytes,model);
  if(cache){try{await cache.put(key,new Response(bytes,{headers:{'Content-Type':'application/octet-stream','X-Model-SHA256':sha256}}));}catch(e){warning=`キャッシュ容量が不足しています。推論は継続します。${e.message}`;}}
  return {bytes,sha256,source:'network',warning};
}
export async function cacheSummary(){
  try{const c=await caches.open(MODEL_CACHE),keys=await c.keys();return {count:keys.length,models:keys.map(k=>k.url),storage:await navigator.storage?.estimate?.()};}catch(e){return {count:0,error:e.message};}
}
