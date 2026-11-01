import {clamp}from './core.js';
import {BODY_EDGES,HAND_EDGES,FACE_EDGES,FACE_IDS,HAND_IDS,groupOf}from './rtmw2d.js';
// COCO133 renderer; depth is a constrained monocular estimate.
export class PoseView{
  constructor(canvas){
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.yaw=.22;this.pitch=.12;this.zoom=1;this.points=null;this.world=false;this.threshold=.3;this.showHands=true;this.showFace=true;
    let drag=null;canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(!drag)return;this.yaw+=(e.clientX-drag[0])*.008;this.pitch=clamp(this.pitch+(e.clientY-drag[1])*.006,-1.2,1.2);drag=[e.clientX,e.clientY];this.draw();});
    for(const name of['pointerup','pointercancel'])canvas.addEventListener(name,()=>drag=null);
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=clamp(this.zoom*Math.exp(-e.deltaY*.001),.4,3);this.draw();},{passive:false});
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);this.draw();
  }
  reset(){this.yaw=.22;this.pitch=.12;this.zoom=1;this.draw();}
  update(points,distance,options={}){this.points=points;this.distance=distance;Object.assign(this,options);this.draw();}
  cameraPlane(point){
    let[x,y,z]=point;
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const xx=x*cy+z*sy,zz=-x*sy+z*cy,yy=y*cp-zz*sp,depth=y*sp+zz*cp;
    const perspective=Math.max(.25,1+depth*.14);
    return [xx/perspective,yy/perspective];
  }
  project(point,w,h){
    const[x,y]=this.cameraPlane(point),fit=this.fit||{x:0,y:0,scale:Math.min(w,h)*.4};
    return [w/2+(x-fit.x)*fit.scale*this.zoom,h/2-(y-fit.y)*fit.scale*this.zoom];
  }
  draw(){
    const{canvas,ctx}=this,w=canvas.clientWidth||480,h=canvas.clientHeight||360,dpr=Math.min(devicePixelRatio||1,2);
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#102a23';ctx.fillRect(0,0,w,h);
    const drawLine=(a,b,color,width=1)=>{const aa=this.project(a,w,h),bb=this.project(b,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...aa);ctx.lineTo(...bb);ctx.stroke();};
    const coords=(this.points||[]).map((p,i)=>{
      if(this.world){const q=this.distance?.world[i];if(!q)return null;return[q[0],q[1]-.8,q[2]-(this.worldOrigin??0)];}
      return p.relativeDisplay;
    });
    // A regressed 3D joint remains drawable when its separate 2D head is off-image.
    const visible=i=>coords[i]&&coords[i].every(Number.isFinite)&&this.points[i].valid!==false&&this.points[i].score>=this.threshold;
    const enabled=i=>!(FACE_IDS.has(i)&&!this.showFace)&&!(HAND_IDS.has(i)&&!this.showHands);
    const plane=coords.flatMap((q,i)=>visible(i)&&enabled(i)?[this.cameraPlane(q)]:[]);
    if(plane.length){
      const xs=plane.map(p=>p[0]),ys=plane.map(p=>p[1]),left=Math.min(...xs),right=Math.max(...xs),bottom=Math.min(...ys),top=Math.max(...ys);
      this.fit={x:(left+right)/2,y:(bottom+top)/2,scale:Math.min(w*.82/Math.max(.5,right-left),h*.82/Math.max(1.2,top-bottom))};
    }else this.fit=null;
    // Keep the calibrated world viewport fixed so lateral/forward movement remains visible.
    if(this.world&&this.fit){
      if(this.worldFit&&this.worldFitOrigin===this.worldOrigin)this.fit=this.worldFit;
      else{this.worldFit={...this.fit};this.worldFitOrigin=this.worldOrigin;}
    }
    const floor=this.world?-.8:-.95;
    for(let i=-10;i<=10;i++){const p=i*.25;drawLine([p,floor,-2.5],[p,floor,2.5],i===0?'#3e5c47':'#244638');drawLine([-2.5,floor,p],[2.5,floor,p],i===0?'#3e5c47':'#244638');}
    if(!this.points)return;
    const edges=[...BODY_EDGES.map(e=>[e,'#c4e58f',2]),...(this.showHands?HAND_EDGES.map(e=>[e,'#69c9b2',1.2]):[]),...(this.showFace?FACE_EDGES.map(e=>[e,'#bdaf8f',.75]):[])];
    for(const[[a,b],color,width]of edges)if(visible(a)&&visible(b))drawLine(coords[a],coords[b],color,width);
    for(let i=0;i<coords.length;i++){
      if(!visible(i)||(FACE_IDS.has(i)&&!this.showFace)||(HAND_IDS.has(i)&&!this.showHands))continue;
      const group=groupOf(i),[x,y]=this.project(coords[i],w,h);ctx.beginPath();ctx.arc(x,y,group==='body'?3:1.3,0,Math.PI*2);ctx.fillStyle=group==='body'?'#def5af':group==='face'?'#cbbd9b':'#72dbc0';ctx.fill();
    }
  }
}
