const APP_CACHE='vrc-mmpose-app-rtmw2d-v3';
const LOCAL=['./','./index.html','./style.css','./app.js','./core.js','./rtmw2d.js','./rtmw-models.js','./runtime.js','./ort-patch.js','./model-cache.js','./view3d.js','./favicon.svg','./architecture.md'];
self.addEventListener('install',event=>{event.waitUntil(caches.open(APP_CACHE).then(c=>c.addAll(LOCAL)).then(()=>self.skipWaiting()));});
self.addEventListener('activate',event=>{event.waitUntil((async()=>{for(const name of await caches.keys())if(name.startsWith('vrc-mmpose-app-')&&name!==APP_CACHE)await caches.delete(name);await self.clients.claim();})());});
self.addEventListener('fetch',event=>{
  const request=event.request,url=new URL(request.url);
  if(request.method!=='GET'||url.protocol==='blob:')return;
  // Large ONNX files are hash-verified and persisted by model-cache.js.
  if(url.hostname==='huggingface.co'||url.hostname.endsWith('.huggingface.co')||url.pathname.endsWith('.onnx'))return;
  const runtime=url.hostname==='cdn.jsdelivr.net'&&url.pathname.startsWith('/npm/onnxruntime-web@1.24.3/');
  const own=url.origin===self.location.origin;
  if(!own&&!runtime)return;
  if(request.destination==='video'||url.pathname.endsWith('.mp4'))return;
  event.respondWith((async()=>{
    const c=await caches.open(APP_CACHE);
    if(runtime){const hit=await c.match(request);if(hit)return hit;const r=await fetch(request);if(r.ok)await c.put(request,r.clone());return r;}
    try{const r=await fetch(request);if(r.ok)await c.put(request,r.clone());return r;}catch(e){const hit=await c.match(request);if(hit)return hit;throw e;}
  })());
});
