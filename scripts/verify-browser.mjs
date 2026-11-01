// Numerical CI only: no screenshots, video recording, or visual review.
import {chromium}from 'playwright';
import {spawn}from 'node:child_process';
import {readFile,writeFile,mkdir}from 'node:fs/promises';
import assert from 'node:assert/strict';
const server=spawn(process.execPath,['serve.mjs'],{stdio:['ignore','pipe','inherit']});
await new Promise((resolve,reject)=>{server.stdout.once('data',resolve);server.once('error',reject);});
let browser;
try{
  browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan','--disable-vulkan-surface']});
  const page=await browser.newPage({acceptDownloads:true,serviceWorkers:'allow'}),errors=[];let modelRequests=0;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',request=>{if(request.url().includes('/momolesang/InstantHMR/resolve/')&&request.url().endsWith('/instanthmr.onnx'))modelRequests++;});
  await page.goto('http://127.0.0.1:4173/');
  const adapter=await page.evaluate(async()=>{const a=await navigator.gpu?.requestAdapter();return a?{info:a.info,features:[...a.features]}:null;});
  assert.ok(adapter,'CI WebGPU software adapter unavailable');
  console.log('CI software WebGPU adapter',JSON.stringify(adapter));
  async function prepare(){
    await page.locator('#load-model').click();
    await page.waitForFunction(()=>document.getElementById('model-status').textContent.includes('推論確認済み')||!document.getElementById('error-panel').hidden,{},{timeout:420000});
    const error=await page.locator('#error-panel').isVisible();
    if(error)throw Error(await page.locator('#error-message').textContent());
    assert.match(await page.locator('#model-status').textContent(),/推論確認済み/);
  }
  await prepare();
  const diagnostics=await page.locator('#diagnostics').textContent();
  assert.match(diagnostics,/joints_3d/);assert.match(diagnostics,/Numerical startup check:/);
  await page.locator('#video-file').setInputFiles('dist/sample.mp4');
  await page.waitForFunction(()=>!document.getElementById('step').disabled,{},{timeout:120000});
  await page.locator('#step').click();
  await page.waitForFunction(()=>Number(document.getElementById('record-count').textContent.replaceAll(',',''))>=1,{},{timeout:120000});
  const download=page.waitForEvent('download');await page.locator('#export-json').click();
  const poses=JSON.parse(await readFile(await (await download).path(),'utf8'));
  assert.equal(poses.schema,'vrc-instanthmr-poses/v2');assert.equal(poses.frames[0].points.length,70);
  for(const p of poses.frames[0].points){assert.ok(p.relativeModel.every(Number.isFinite));assert.equal(p.score,null);}
  assert.equal(poses.frames[0].points[13].name,'left_ankle');assert.equal(poses.frames[0].points[20].name,'right_heel');
  assert.equal(poses.benchmark.detector,null,'Default full-frame mode must not fetch the optional detector');
  const cache=await page.evaluate(async()=>{const c=await caches.open('vrc-mmpose-models-v1');return(await c.keys()).map(r=>r.url);});
  assert.ok(cache.some(url=>url.includes('/momolesang/InstantHMR/resolve/')));
  assert.equal(modelRequests,1);
  await page.reload();await prepare();assert.equal(modelRequests,1,'Reload should reuse SHA-verified Cache Storage model');
  assert.equal(errors.length,0,errors.join('\n'));
  const report={provider:'WebGPU via CI software adapter; not RTX performance',adapter,diagnostics,cacheReuse:true,modelNetworkRequests:modelRequests,
    source:'bundled public sample',frames:poses.frames.length,finiteJoints:70,ankleIds:[13,14],footIds:[15,16,17,18,19,20],pageErrors:errors,
    visualVerification:false,notes:['No screenshots or browser visual verification','No RTX performance claim']};
  await mkdir('verification/instant',{recursive:true});await writeFile('verification/instant/browser-numerical.json',JSON.stringify(report,null,2));
  console.log('INSTANT_BROWSER='+JSON.stringify(report));
}finally{await browser?.close();server.kill();}
