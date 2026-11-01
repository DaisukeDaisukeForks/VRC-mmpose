import {BODY_EDGES,HAND_EDGES,FACE_EDGES,clamp}from './core.js';
// A perspective Canvas renderer keeps all 133 points visible without a second GPU session.
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
  project(point,w,h){
    let[x,y,z]=point;
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const xx=x*cy+z*sy,zz=-x*sy+z*cy,yy=y*cp-zz*sp,depth=y*sp+zz*cp;
    const scale=Math.min(w,h)*.68*this.zoom/(1+depth*.14);
    return [w/2+xx*scale,h*.48-yy*scale];
  }
  draw(){
    const{canvas,ctx}=this,w=canvas.clientWidth||480,h=canvas.clientHeight||360,dpr=Math.min(devicePixelRatio||1,2);
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#102a23';ctx.fillRect(0,0,w,h);
    const drawLine=(a,b,color,width=1)=>{const aa=this.project(a,w,h),bb=this.project(b,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...aa);ctx.lineTo(...bb);ctx.stroke();};
    const floor=this.world?-.8:-.95;
    for(let i=-10;i<=10;i++){const p=i*.25;drawLine([p,floor,-2.5],[p,floor,2.5],i===0?'#3e5c47':'#244638');drawLine([-2.5,floor,p],[2.5,floor,p],i===0?'#3e5c47':'#244638');}
    if(!this.points)return;
    const coords=this.points.map((p,i)=>{
      if(this.world){const q=this.distance?.world[i];if(!q)return null;return[q[0],q[1]-.8,q[2]-(this.worldOrigin??0)];}
      return p.relativeDisplay;
    });
    const visible=i=>coords[i]&&coords[i].every(Number.isFinite)&&this.points[i].score>=this.threshold&&this.points[i].inFrame;
    const edges=[...BODY_EDGES.map(e=>[e,'#c4e58f',2]),...(this.showHands?HAND_EDGES.map(e=>[e,'#69c9b2',1.2]):[]),...(this.showFace?FACE_EDGES.map(e=>[e,'#bdaf8f',.75]):[])];
    for(const[[a,b],color,width]of edges)if(visible(a)&&visible(b))drawLine(coords[a],coords[b],color,width);
    for(let i=0;i<coords.length;i++){
      if(!visible(i)||(i>=23&&i<91&&!this.showFace)||(i>=91&&!this.showHands))continue;
      const[x,y]=this.project(coords[i],w,h);ctx.beginPath();ctx.arc(x,y,i<23?3:1.3,0,Math.PI*2);ctx.fillStyle=i<23?'#def5af':i<91?'#cbbd9b':'#72dbc0';ctx.fill();
    }
  }
}
