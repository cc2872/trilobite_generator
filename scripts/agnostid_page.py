"""scripts/agnostid_page.py — writes web/agnostid.html, the agnostid's own plan editor (8 Nov 2026).

The page draws the whole animal in plan (canvas; the drawing code mirrors the builder's formulas) over Claire's
reference drawing (presets/agnostida_drawing.png), with draggable dots in the studio's sketch_controls pattern and a
slider for every parameter the agnostid uses. Its state is the agnostida preset's "drawing" block (see web/app.py,
/api/drawing/<preset>): on load the page fetches the saved block, BUILD sends the state through /api/drawing/agnostida/
build (the real pieces, STL downloads, the blueprint sheet), SAVE writes it back into presets/agnostida.json and the
studio's saved head/body agnostida-drawing. The embedded FULL/BASE are the block as it was when this script ran: the
fallback when the API is not there (the standalone copy, --standalone out.html, is the same page without the buttons).

    python scripts/agnostid_page.py                 -> web/agnostid.html
    python scripts/agnostid_page.py --standalone X  -> X (no BUILD/SAVE; COPY EDITS JSON only)
"""
import base64, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "web")); sys.path.insert(0, os.path.join(ROOT, "isopod_model"))
ORDER = "agnostida"


def data():
    """FULL (every resolved value the drawing reads) and BASE (the block's edits) from the preset's drawing block."""
    import app as A
    f, rec = A._preset_record(ORDER); D = rec["drawing"]; R = A._drawing_resolve(ORDER, D)
    full = dict(P={k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in R["P"].items()},
                outline=R["outline"], face=R["face"], rows=dict(rows=R["rows"]))
    base = dict(P=D["params"], outline=D["outline"], face=D["face"], rows=dict(rows=D["rows"]))
    return full, base, D.get("saved_at", "")


S = [
 ("Proportions","P","length","Body length (mm)",80,130,1),
 ("Proportions","P","width","Body width (mm)",40,70,1),
 ("Proportions","P","cephFrac","Head fraction",0.3,0.5,0.005),
 ("Proportions","P","pygFrac","Tail fraction",0.3,0.4,0.005),
 ("Head outline","outline","length","Head length",1.2,2.4,0.01),
 ("Head outline","outline","width","Head width",0.7,1.1,0.01),
 ("Head outline","outline","front_exp","Front roundness",1.6,3.5,0.05),
 ("Head outline","outline","widest","Widest at (0 rear .. 1 front)",0.0,0.8,0.01),
 ("Head outline","outline","rear_taper","Rear corner tuck",0.0,0.5,0.01),
 ("Head outline","outline","notch","Rear corners drop (notch)",0.0,0.6,0.01),
 ("Head outline","outline","corner","Rear corner rounding",0.0,0.5,0.01),
 ("Head outline","outline","side","Straight side run",0.0,0.6,0.01),
 ("Genal spines","outline","genal","Length",0.0,0.5,0.01),
 ("Genal spines","outline","genal_w","Root width",0.02,0.3,0.005),
 ("Genal spines","outline","genal_angle","Root angle (deg)",-90,120,1),
 ("Genal spines","outline","genal_kink","Kink at",0.05,0.9,0.01),
 ("Genal spines","outline","genal_kink_angle","Angle after kink",-120,120,1),
 ("Genal spines","outline","genal_tip","Tip: point-round",0.0,1.0,0.01),
 ("Genal spines","outline","genal_at","Root along rear margin (0 corner .. 1 inward)",-1.0,1.0,0.01),
 ("Genal spines","outline","genal_web","Web into corner",0.0,1.0,0.01),
 ("Genal spines","outline","genal_web_round","Web edge (-bulge..+hug)",-1.0,1.0,0.05),
 ("Head face","face","border_w","Border band width",0.03,0.3,0.005),
 ("Head face","P","glabInflate","Glabella width",0.3,1.3,0.01),
 ("Head face","face","glab_len","Glabella length",0.5,1.1,0.01),
 ("Head face","face","furrow","Furrow depth (mm)",0.2,2.5,0.05),
 ("Head face","face","node","Node (mm)",0.0,2.0,0.05),
 ("Head face","face","node_at","Node at (0 rear .. 1 front)",0.05,0.9,0.01),
 ("Head face","face","furrow_at","Glabellar furrow at (-1 = default)",-1.0,0.95,0.01),
 ("Head face","face","collar","Front collar (T) size",0.0,0.4,0.01),
 ("Head face","face","glab_round","Glabella vault (0 flat .. 1 round)",0.0,1.0,0.01),
 ("Head face","face","glab_bulge","Glabella side bulge",-0.15,0.3,0.01),
 ("Head face","face","bullae","Basal lobes size",0.0,0.25,0.005),
 ("Head face","face","bullae_at","Basal lobes at",0.0,0.9,0.01),
 ("Head face","face","bullae_rise","Basal lobes height",0.0,0.4,0.01),
 ("Segments","row0","width","Seg 1 half-width (model mm; rows spec)",10,35,0.1),
 ("Segments","row1","width","Seg 2 half-width (model mm; rows spec)",10,35,0.1),
 ("Segments","P","pleuraFlip","Flip pleura fore-aft",0,1,1),
 ("Segments","P","tipRound","Blunt round end",0.0,1.0,0.01),
 ("Segments","P","tipTaper","Tip taper",0.0,0.95,0.01),
 ("Segments","P","ringArch","Axial ring arch",0.0,0.8,0.01),
 ("Segments","P","pleuralLobes","Lobes per side",0,4,1),
 ("Segments","P","pleuralLobeCut","Lobe scallop",0.0,0.6,0.01),
 ("Segments","P","pleuralLobeRelief","Lobe relief",0.0,2.0,0.05),
 ("Segments","P","tipSweep","Tip sweep",0.0,2.0,0.01),
 ("Tail","P","pygWidth","Tail width / last segment",0.5,2.0,0.01),
 ("Tail","P","tailOutlineExp","Outline squareness",1.5,4.0,0.05),
 ("Tail","P","pygNeck","Neck pinch",0.0,0.5,0.01),
 ("Tail","P","pygNeckLen","Neck length",0.05,0.6,0.01),
 ("Tail","P","pygAxisFrac","Axis width",0.1,0.6,0.01),
 ("Tail","P","pygAxisLen","Axis length",0.25,0.97,0.01),
 ("Tail","P","pygRings","Axis rings",0,4,1),
 ("Tail","P","pygRingAt","First ring at (-1 = default)",-1.0,0.95,0.01),
 ("Tail","P","pygNode","Axis node",0.0,3.0,0.05),
 ("Tail","P","pygNodePos","Node at",0.05,0.95,0.01),
 ("Tail spines","P","pygSpine","Pair length",0.0,1.0,0.01),
 ("Tail spines","P","pygSplay","Pair splay (deg)",0,45,1),
 ("Tail spines","P","pygSpineSweep","Pair lean back (deg)",0,80,1),
 ("Tail spines","P","pygSpineWidth","Pair base width",0.3,2.0,0.05),
 ("Tail spines","P","pygLobePoint","Pair flank (straight-concave)",1.0,3.0,0.1),
 ("Tail spines","P","pygMarginal","Extra marginals (count)",0,6,1),
 ("Tail spines","P","pygMarginalLen","Marginal length",0.0,0.25,0.005),
 ("Overlay","view","refOpacity","Reference opacity",0.0,1.0,0.01),
 ("Overlay","view","refScale","Reference scale",0.7,1.4,0.005),
 ("Overlay","view","refY","Reference y-shift",-80,80,1),
]


def panels_html():
    groups, rows = [], {}
    for g, d, k, lab, lo, hi, st in S:
        if g not in groups: groups.append(g); rows[g] = ""
        rows[g] += (f'<div class="row"><label><span>{lab}</span><span class="key">{d}.{k}</span></label>'
                    f'<div class="ctl"><input type="range" data-d="{d}" data-k="{k}" min="{lo}" max="{hi}" step="{st}">'
                    f'<span class="val"></span></div></div>\n')
    return "".join(f'<details {"open" if g=="Overlay" else ""}><summary>{g}</summary>{rows[g]}</details>' for g in groups)


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agnostid plan editor</title>
<style>
 :root{--ink:#fff;--bg:#000;--dim:#9a9a9a;--grid:#1c1c1c;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;display:flex;min-height:100vh}
 #left{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:10px;gap:10px}
 canvas{background:var(--bg);max-width:100%;height:auto;border:1px solid var(--grid);touch-action:none}
 #right{width:330px;max-width:45vw;border-left:1px solid var(--grid);padding:10px 14px;overflow-y:auto;height:100vh}
 @media (max-width:700px){ body{flex-direction:column} #left{padding:6px}
  #right{width:100%;max-width:none;height:auto;border-left:none;border-top:1px solid var(--grid)} }
 h1{font-size:13px;letter-spacing:.14em;font-weight:400;margin:4px 0 2px}
 .sub{color:var(--dim);font-size:11px;margin-bottom:10px}
 .nav{display:flex;gap:6px;margin:0 0 8px} .nav a{flex:1;color:var(--dim);border:1px solid var(--grid);text-decoration:none;text-align:center;padding:5px 0;font-size:10px;letter-spacing:.08em;text-transform:uppercase} .nav a:hover{color:var(--ink);border-color:var(--ink)}
 details{border-top:1px solid var(--grid);padding:5px 0}
 summary{cursor:pointer;letter-spacing:.1em;padding:4px 0;text-transform:uppercase;font-size:11px}
 .row{margin:8px 0}
 label{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;margin-bottom:2px}
 .key{color:#555;font-size:10px}
 .ctl{display:flex;align-items:center;gap:8px}
 input[type=range]{flex:1;accent-color:#fff;height:26px;margin:0}
 .val{color:var(--ink);font-size:11px;min-width:42px;text-align:right}
 button{background:var(--bg);color:var(--ink);border:1px solid var(--ink);font:inherit;letter-spacing:.1em;padding:9px 10px;width:100%;margin:6px 0;cursor:pointer}
 button:hover{background:#181818} button:disabled{color:#555;border-color:#333;cursor:default}
 #json{width:100%;height:110px;background:#0a0a0a;color:var(--dim);border:1px solid var(--grid);font:10px/1.4 ui-monospace,monospace;display:none}
 #status{color:var(--dim);font-size:11px;min-height:16px;margin:4px 0} #status.err{color:#f66}
 #result{width:100%;max-width:680px;color:var(--dim);font-size:11px} #result table{border-collapse:collapse;width:100%} #result td{padding:2px 6px;border-bottom:1px solid var(--grid)} #result a{color:var(--ink)}
 #result img{width:100%;border:1px solid var(--grid);margin-top:8px}
</style></head><body>
<div id="left"><canvas id="cv" width="680" height="920"></canvas><div id="result"></div></div>
<div id="right">
 <h1>AGNOSTID &middot; PLAN EDITOR</h1>
 __NAV__
 <div class="sub">__SUB__</div>
 <div id="status"></div>
 __BUTTONS__
 __PANELS__
 <button id="copy">COPY EDITS JSON</button><span id="copied"></span>
 <button id="reset">RESET TO SAVED</button>
 <textarea id="json" readonly></textarea>
</div>
<script>
const SITE = __SITE__;                      // true on the site: load/build/save through /api/drawing/<order>
const ORDER = "__ORDER__";
const REF_IMG = "data:image/png;base64,__REF__";
let FULL = __FULL__;                        // every resolved parameter: the drawing reads these (the block as saved when this page was written)
let BASE = __CUR__;                         // the block's edits: COPY EDITS JSON = BASE + the editable keys
const state = {view:{refOpacity:0.35, refScale:1.0, refY:0}};
const copyOf=o=>JSON.parse(JSON.stringify(o));
function initState(full){ for(const d of ['P','outline','face']) state[d]=copyOf(full[d]);
  const rr=(full.rows&&full.rows.rows)||{}; for(const k in state){ if(/^row\d+$/.test(k)) delete state[k]; }
  for(const k in rr) state['row'+k]=copyOf(rr[k]); }
initState(FULL);
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const img = new Image(); img.src = REF_IMG; img.onload = ()=>safeDraw();
const clamp=(v,lo,hi)=>Math.min(Math.max(v,lo),hi);
const smooth=(a,b,x)=>{const t=clamp((x-a)/(b-a),0,1);return t*t*(3-2*t);};
function solveMono(f,target,lo,hi){ for(let i=0;i<24;i++){const m=(lo+hi)/2; (f(m)<target)?lo=m:hi=m;} return (lo+hi)/2; }
function segHalf(si,Wh){ const r=state['row'+si]; if(r&&r.width!==undefined) return r.width;   // the rows spec's half-width (model mm) ...
  const P=state.P; return (si===0?P.widthThoraxFront:P.widthThoraxRear)*Wh; }                   // ... or the width curve's
function rowsSpec(){ const out={}; const rr=(BASE.rows&&BASE.rows.rows)||{};
  for(const k in rr) out[k]=Object.assign({},rr[k],state['row'+k]||{});
  for(const k in state){ const m=k.match(/^row(\d+)$/); if(m&&!out[m[1]]) out[m[1]]=copyOf(state[k]); }
  return {rows:out}; }

let G={};                                    // geometry of the last draw (model mm + transform)
function geom(){
  const P=state.P,O=state.outline;
  const Wh=P.width/2;
  const A=O.width*Wh*1.02, Lh=O.length*Wh*0.92;
  const Lth=(1-P.cephFrac-P.pygFrac)*P.length, d=Lth/2;
  const Lp=P.pygFrac*P.length, wp=P.widthThoraxRear*Wh*P.pygWidth;
  const total=Lh+Lth+Lp+6;
  const sc=Math.min(cv.height*0.92/total, cv.width*0.9/(2*Math.max(A,wp,segHalf(1,Wh))+10));
  const X=cv.width/2, Y0=cv.height*0.04;
  return {Wh,A,Lh,Lth,d,Lp,wp,total,sc,X,Y0, yT:Lh+Lth+3};
}
const px=(x,y)=>[G.X+x*G.sc, G.Y0+y*G.sc];
const inv=(cxp,cyp)=>[(cxp-G.X)/G.sc,(cyp-G.Y0)/G.sc];

// ---- handles: the studio's pattern (sketch_controls.handles): name -> {pos, apply(x,y), word}
function handles(){
  const P=state.P,O=state.outline,F=state.face,g=G;
  const set=(d,k,v,lo,hi)=>{if(!state[d])state[d]={}; state[d][k]=clamp(v,lo,hi);};
  const tw=(1-O.widest)*g.Lh;
  const gw=0.30*g.A*P.glabInflate, glL=F.glab_len*g.Lh*0.92, gy0=g.Lh*0.995;
  const rc=[g.A*0.97*(1-O.rear_taper), g.Lh*0.995];
  const gl=O.genal*g.A, a1=O.genal_angle*Math.PI/180, a2=O.genal_kink_angle*Math.PI/180;
  const L1=gl*O.genal_kink, L2=gl*(1-O.genal_kink);
  const dir=an=>[Math.sin(an),Math.cos(an)];
  const d1=dir(a1), d2=dir(a2);
  const k1=[rc[0]+d1[0]*L1, rc[1]+d1[1]*L1];
  const gt=[k1[0]+d2[0]*L2, k1[1]+d2[1]*L2];
  const nO=P.tailOutlineExp, nk=P.pygNeck;
  const neckf=yy=>1-nk*(1-smooth(0,P.pygNeckLen,yy));
  const tx=ph=>{const by=g.Lp*Math.pow(Math.sin(ph),2/nO);
    return [g.wp*Math.pow(Math.cos(ph),2/nO)*neckf(by/g.Lp), g.yT+by];};
  const aw=P.pygAxisFrac*g.wp, al=P.pygAxisLen*g.Lp;
  const {pb,pt,radLean}=pairGeom(P,g);
  const segSet=(si,x)=>{ if(state['row'+si]) set('row'+si,'width',Math.abs(x),10,35); else set('P',si===0?'widthThoraxFront':'widthThoraxRear',Math.abs(x)/g.Wh,0.4,1.0); };
  const H={
   headLen:{pos:[0,0],word:'length',apply:(x,y)=>set('outline','length',(g.Lh-y)/(g.Wh*0.92),1.2,2.4)},
   headWide:{pos:[g.A,tw],word:'width / widest',apply:(x,y)=>{set('outline','width',Math.abs(x)/(g.Wh*1.02),0.7,1.1); set('outline','widest',1-y/g.Lh,0.0,0.8);}},
   front:{pos:(()=>{const yy=tw*0.45;return [g.A*Math.pow(1-Math.pow((tw-yy)/tw,O.front_exp),1/O.front_exp),yy];})(),word:'front',
     apply:(x,y)=>{const yy=tw*0.45, t=clamp(Math.abs(x)/g.A,0.05,0.98);
       set('outline','front_exp',solveMono(fe=>Math.pow(1-Math.pow((tw-yy)/tw,fe),1/fe),t,1.6,3.5),1.6,3.5);}},
   rear:{pos:[rc[0],rc[1]-2],word:'corner',apply:(x,y)=>set('outline','rear_taper',1-Math.abs(x)/(g.A*0.97),0,0.5)},
   genKink:{pos:k1,word:'spine root/kink',apply:(x,y)=>{const dx=Math.abs(x)-rc[0], dy=y-rc[1];
       set('outline','genal_angle',Math.atan2(dx,dy)*180/Math.PI,-90,120);
       set('outline','genal_kink',Math.hypot(dx,dy)/Math.max(gl,1e-3),0.05,0.9);}},
   genTip:{pos:gt,word:'spine tip',apply:(x,y)=>{const dx=Math.abs(x)-k1[0], dy=y-k1[1];
       set('outline','genal_kink_angle',Math.atan2(dx,dy)*180/Math.PI,-120,120);
       set('outline','genal',(L1+Math.hypot(dx,dy))/g.A,0,0.5);}},
   glabW:{pos:[gw,gy0-glL*0.5],word:'glabella',apply:(x,y)=>set('P','glabInflate',Math.abs(x)/(0.30*g.A),0.3,1.3)},
   glabLen:{pos:[0,gy0-glL],word:'glabella front',apply:(x,y)=>set('face','glab_len',(gy0-y)/(g.Lh*0.92),0.5,1.1)},
   border:{pos:[g.A-F.border_w*g.A,tw*1.25],word:'border',apply:(x,y)=>set('face','border_w',(g.A-Math.abs(x))/g.A,0.03,0.3)},
   bullae:{pos:[gw+F.bullae*g.A*0.75+2,gy0-F.bullae_at*glL],word:'basal lobe',apply:(x,y)=>{
       set('face','bullae',Math.max(Math.abs(x)-gw-2,0)/(g.A*0.75),0,0.25);
       set('face','bullae_at',(gy0-y)/Math.max(glL,1e-3),0,0.9);}},
   seg1:{pos:[segHalf(0,g.Wh),g.Lh+1.5+g.d*0.5],word:'seg 1',apply:(x,y)=>segSet(0,x)},
   seg2:{pos:[segHalf(1,g.Wh),g.Lh+1.5+g.d*1.5+P.tipSweep*g.d],word:'seg 2 / sweep',apply:(x,y)=>{
       segSet(1,x);
       set('P','tipSweep',(y-(g.Lh+1.5+g.d*1.5))/g.d,0,2);}},
   neck:{pos:[g.wp*neckf(0),g.yT+1],word:'neck',apply:(x,y)=>set('P','pygNeck',1-Math.abs(x)/g.wp,0,0.5)},
   tailWide:{pos:tx(Math.PI*0.22),word:'tail width',apply:(x,y)=>{}},
   tailShape:{pos:tx(Math.PI*0.37),word:'tail shape',apply:(x,y)=>{const t=clamp(Math.abs(x)/g.wp,0.05,0.98);
       set('P','tailOutlineExp',solveMono(n=>Math.pow(Math.cos(Math.PI*0.37),2/n)*neckf(Math.pow(Math.sin(Math.PI*0.37),2/n)),t,1.5,4.0),1.5,4.0);}},
   tailLen:{pos:[0,g.yT+g.Lp],word:'tail length',apply:(x,y)=>set('P','pygFrac',(y-g.yT)/P.length,0.3,0.4)},
   axis:{pos:[aw,g.yT+al],word:'tail axis',apply:(x,y)=>{set('P','pygAxisFrac',Math.abs(x)/g.wp,0.1,0.6);
       set('P','pygAxisLen',(y-g.yT)/g.Lp,0.25,0.97);}},
   node:{pos:[0,g.yT+P.pygNodePos*al],word:'node',apply:(x,y)=>set('P','pygNodePos',(y-g.yT)/Math.max(al,1e-3),0.05,0.95)},
   pairBase:{pos:pb,word:'tail spine root',apply:(x,y)=>{const ph=Math.atan2(clamp(y-g.yT,0.5,1e9),Math.max(Math.abs(x),0.5));
       set('P','pygSplay',90-Math.atan(Math.tan(ph)*g.Lp/g.wp)*180/Math.PI,0,45);}},
   pairTip:{pos:pt,word:'tail spine tip / lean',apply:(x,y)=>{const dx=Math.abs(x)-pb[0], dy=y-pb[1];
       set('P','pygSpine',Math.hypot(dx,dy)/g.Lp,0,1.0);
       const ang=Math.atan2(dx,dy)-Math.atan2(pb[0],pb[1]-g.yT);   // direction relative to the radial
       set('P','pygSpineSweep',-ang*180/Math.PI+radLean*180/Math.PI,0,80);}},
  };
  return H;
}

// tail-wide: direct and stable
function fixTailWide(H){ H.tailWide.apply=(x,y)=>{ const P=state.P;
  state.P.pygWidth=clamp(Math.abs(x)/(Math.pow(Math.cos(Math.PI*0.22),2/P.tailOutlineExp)*P.widthThoraxRear*(state.P.width/2)),0.5,2.0); }; return H; }

function pairGeom(P,g){                        // the paired pygidial spines (anatomy/tail.py: phi_f, sweep lean)
  const nO=P.tailOutlineExp, nk=P.pygNeck, sp=P.pygSplay*Math.PI/180;
  const neckf=yy=>1-nk*(1-smooth(0,P.pygNeckLen,yy));
  const phi=Math.atan(g.wp/g.Lp*Math.tan(Math.PI/2-sp));
  const by=g.Lp*Math.pow(Math.sin(phi),2/nO), bx=g.wp*Math.pow(Math.cos(phi),2/nO)*neckf(by/g.Lp);
  const pb=[bx,g.yT+by]; const rad=Math.atan2(bx,by);        // radial direction (angle from straight back)
  const radLean=rad; const lean=P.pygSpineSweep*Math.PI/180;
  const dirA=rad-lean;                                          // leaned toward the rear
  const L=P.pygSpine*g.Lp;
  const pt=[bx+Math.sin(dirA)*L, g.yT+by+Math.cos(dirA)*L];
  const hw=Math.hypot(bx,by)*(13*Math.PI/180)*P.pygSpineWidth*0.5;
  return {pb,pt,radLean,hw,dirA};
}
let hover=null, dragging=null;
function draw(){
  const P=state.P,O=state.outline,F=state.face,V=state.view;
  G=geom();
  cx.clearRect(0,0,cv.width,cv.height);
  if(V.refOpacity>0 && img.width){ cx.save(); cx.globalAlpha=V.refOpacity;
    const ih=G.total*G.sc*V.refScale, iw=ih*img.width/img.height;
    cx.drawImage(img, G.X-iw/2, G.Y0+V.refY, iw, ih); cx.restore(); }
  cx.strokeStyle='#fff'; cx.lineWidth=1.6; cx.lineJoin='round'; cx.lineCap='round';
  const M=pts=>{cx.beginPath();pts.forEach((p,i)=>{const q=px(p[0],p[1]);i?cx.lineTo(q[0],q[1]):cx.moveTo(q[0],q[1]);});cx.stroke();};
  const Mir=pts=>{M(pts);M(pts.map(p=>[-p[0],p[1]]));};
  const {A,Lh,d,Lp,wp,Wh,yT}=G;
  // head
  const fe=O.front_exp, tw=(1-O.widest)*Lh, rt=O.rear_taper, headPts=[];
  for(let i=0;i<=60;i++){const t=i/60, yy=t*Lh; let a;
    if(yy<=tw) a=A*Math.pow(1-Math.pow((tw-yy)/tw,fe),1/fe);
    else{const u=(yy-tw)/(Lh-tw); a=A*Math.pow(1-Math.pow(u,2.6),1/2.6)*(1-rt*u*u);}
    headPts.push([a,yy]);}
  Mir(headPts);
  const bw=F.border_w*A;
  Mir(headPts.map(p=>[Math.max(p[0]-bw,0),p[1]]).filter(p=>p[0]>1));
  const gw=0.30*A*P.glabInflate, glL=F.glab_len*Lh*0.92, gy0=Lh*0.995, cap=[];
  if(F.collar>0.005){const cw=F.collar*A; M([[-cw,bw+cw*0.6],[0,bw+1.2*cw],[cw,bw+cw*0.6]]); M([[0,bw+1.2*cw],[0,gy0-F.glab_len*Lh*0.92-2]]);}
  for(let i=0;i<=40;i++){const t=i/40, yy=gy0-t*glL;
    let w=gw*(1-0.12*t); const tt=smooth(0.78,1,t); w*=Math.sqrt(Math.max(1-tt*tt,1e-4));
    cap.push([w,yy]);}
  Mir(cap);
  const fAt=(F.furrow_at>=0?F.furrow_at:0.28); cx.setLineDash([4,4]); M([[-gw*0.92,gy0-glL*fAt],[gw*0.92,gy0-glL*fAt]]); cx.setLineDash([]);
  if(F.node>0.02){cx.setLineDash([2,3]);cx.beginPath();const n=px(0,gy0-(F.node_at!==undefined?F.node_at:0.21)*Lh);
    cx.arc(n[0],n[1],Math.max(F.node*2.2,3),0,6.283);cx.stroke();cx.setLineDash([]);}
  if(F.bullae>0.005){const Rb=F.bullae*A, fy=gy0-F.bullae_at*glL;
    [1,-1].forEach(m=>{cx.beginPath();const c=px(m*(gw+Rb*0.75),fy);
      cx.ellipse(c[0],c[1],Rb*G.sc*0.8,Rb*G.sc*1.25,m*0.5,0,6.283);cx.stroke();});}
  // genal
  const rc=[A*0.97*(1-rt),Lh*0.995], gl=O.genal*A, gwd=O.genal_w*A;
  const dirf=an=>[Math.sin(an),Math.cos(an)];
  const d1=dirf(O.genal_angle*Math.PI/180), d2=dirf(O.genal_kink_angle*Math.PI/180);
  const L1=gl*O.genal_kink, L2=gl*(1-O.genal_kink);
  const k1=[rc[0]+d1[0]*L1,rc[1]+d1[1]*L1], gt=[k1[0]+d2[0]*L2,k1[1]+d2[1]*L2];
  if(O.genal_web>0.01){ [1,-1].forEach(m=>{ const wa=px(m*(rc[0]-O.genal_web*g_A(A)),rc[1]), wb=px(m*k1[0],k1[1]);
      cx.beginPath(); cx.moveTo(wa[0],wa[1]); cx.lineTo(wb[0],wb[1]); cx.stroke(); }); }
  [1,-1].forEach(m=>{cx.beginPath();
    const q0=px(m*(rc[0]-gwd/2),rc[1]),q1=px(m*(k1[0]-gwd*0.3),k1[1]),q2=px(m*gt[0],gt[1]),
          q3=px(m*(k1[0]+gwd*0.3),k1[1]),q4=px(m*(rc[0]+gwd/2),rc[1]);
    cx.moveTo(q0[0],q0[1]);cx.quadraticCurveTo(q1[0],q1[1],q2[0],q2[1]);
    cx.quadraticCurveTo(q3[0],q3[1],q4[0],q4[1]);cx.stroke();});
  // segments
  const nL=Math.round(P.pleuralLobes), cut=P.pleuralLobeCut, sw=P.tipSweep*d;
  for(let si=0;si<2;si++){
    const wseg=segHalf(si,Wh), ytop=Lh+si*d+1.5;
    const edge=v=>{const pts=[];
      for(let i=0;i<=48;i++){const q=i/48; let hb=d*0.42;
        if(nL>0&&cut>0){const dip=Math.pow(1-Math.abs(Math.sin(Math.PI*nL*q)),1.2);
          hb*=1-cut*dip*smooth(0.08,0.28,q)*(1-smooth(0.82,1,q));}
        if(P.tipRound>0.005){const qq=clamp((q-(1-P.tipRound))/P.tipRound,0,1); hb=Math.max(hb*Math.sqrt(Math.max(1-qq*qq,0)),0.6);}
        hb*=1-P.tipTaper*Math.pow(q,2.5);
        const yc=ytop+d*0.5+(P.pleuraFlip>0.5?-1:1)*sw*Math.pow(q,1.6);
        pts.push([q*wseg,yc+v*hb]);}
      return pts;};
    Mir(edge(-1));Mir(edge(1));
    Mir([[P.pygAxisFrac*Wh*0.9,ytop+d*0.15],[P.pygAxisFrac*Wh*0.9,ytop+d*0.85]]);
  }
  // tail
  const nO=P.tailOutlineExp, nk=P.pygNeck;
  const neckf=yy=>1-nk*(1-smooth(0,P.pygNeckLen,yy));
  const tPts=[];
  for(let i=0;i<=70;i++){const ph=(i/70)*Math.PI/2;
    const xx=wp*Math.pow(Math.cos(ph),2/nO), yy=Lp*Math.pow(Math.sin(ph),2/nO);
    tPts.push([xx*neckf(yy/Lp),yT+yy]);}
  Mir(tPts);
  Mir(tPts.map(p=>[Math.max(p[0]-0.16*wp,0),p[1]]).filter(p=>p[0]>0.5));
  const aw=P.pygAxisFrac*wp, al=P.pygAxisLen*Lp, ax=[];
  for(let i=0;i<=30;i++){const t=i/30;
    ax.push([aw*(1-0.8*t)*Math.sqrt(Math.max(1-Math.pow(smooth(0.75,1,t),2),0.01)),yT+t*al]);}
  Mir(ax);
  for(let k=0;k<Math.round(P.pygRings);k++){const n_=Math.round(P.pygRings); const yy=yT+(P.pygRingAt>=0? Lp*(P.pygAxisLen/0.85)*(P.pygRingAt+k/(n_+1.5)) : al*(k+1)/(n_+1.5));
    Mir([[aw*(1-0.8*((yy-yT)/al))*0.95,yy],[0,yy]]);}
  if(P.pygNode>0.05){cx.setLineDash([2,3]);cx.beginPath();
    const n=px(0,yT+P.pygNodePos*al);cx.arc(n[0],n[1],Math.max(P.pygNode*1.4,3),0,6.283);cx.stroke();cx.setLineDash([]);}
  if(P.pygSpine>0.01){ const pg=pairGeom(P,G); const nx=Math.cos(pg.dirA), ny=-Math.sin(pg.dirA);
    [1,-1].forEach(m=>{cx.beginPath();
      const q0=px(m*(pg.pb[0]-nx*pg.hw),pg.pb[1]-ny*pg.hw), q1=px(m*pg.pt[0],pg.pt[1]), q2=px(m*(pg.pb[0]+nx*pg.hw),pg.pb[1]+ny*pg.hw);
      cx.moveTo(q0[0],q0[1]);cx.lineTo(q1[0],q1[1]);cx.lineTo(q2[0],q2[1]);cx.stroke();});}
  const mf=P.pygMarginalFrom*Math.PI/180, mt=P.pygMarginalTo*Math.PI/180, ml=P.pygMarginalLen*Lp, mw=P.pygSpineWidth;
  if(Math.round(P.pygMarginal)>0) [0,1].forEach(k=>{const ph=mf+(mt-mf)*(k+0.5)/2;
    const bx=wp*Math.pow(Math.abs(Math.cos(ph)),2/nO)*Math.sign(Math.cos(ph));
    const by=Lp*Math.pow(Math.sin(ph),2/nO);
    const x0=Math.abs(bx)*neckf(by/Lp), y0=yT+by, r=Math.hypot(x0,by)||1, ux=x0/r, uy=by/r;
    [1,-1].forEach(m=>{cx.beginPath();
      const q0=px(m*(x0-2.4*mw),y0),q1=px(m*(x0+ux*ml),y0+uy*ml*1.2),q2=px(m*(x0+2.4*mw),y0-1);
      cx.moveTo(q0[0],q0[1]);cx.lineTo(q1[0],q1[1]);cx.lineTo(q2[0],q2[1]);cx.stroke();});});
  // handles (right side only, like the studio sketch)
  const H=fixTailWide(handles());
  for(const name in H){const h=H[name], q=px(h.pos[0],h.pos[1]);
    cx.beginPath(); cx.arc(q[0],q[1],(name===hover||name===dragging)?8:5.5,0,6.283);
    cx.fillStyle=(name===hover||name===dragging)?'#fff':'#000';
    cx.fill(); cx.strokeStyle='#fff'; cx.lineWidth=1.4; cx.stroke();}
  if(hover||dragging){const h=H[dragging||hover], q=px(h.pos[0],h.pos[1]);
    cx.fillStyle='#9a9a9a'; cx.font='12px monospace'; cx.fillText(h.word, q[0]+12, q[1]-10);}
  cx.strokeStyle='#fff';
}
function g_A(A){return A*0.6;}
function safeDraw(){try{draw();}catch(e){cx.setTransform(1,0,0,1,0,0);cx.fillStyle='#f66';
  cx.font='12px monospace';cx.fillText('draw error: '+e.message,12,20);}}

// ---- pointer: nearest handle within tolerance (the studio's _nearest/_motion)
function canvasPt(ev){const r=cv.getBoundingClientRect();
  return [(ev.clientX-r.left)*cv.width/r.width,(ev.clientY-r.top)*cv.height/r.height];}
function nearest(cp){const H=fixTailWide(handles()); let best=null,bd=26*cv.width/cv.getBoundingClientRect().width;
  for(const name in H){const q=px(H[name].pos[0],H[name].pos[1]);
    const dd=Math.hypot(cp[0]-q[0],cp[1]-q[1]); const dm=Math.hypot(cp[0]-(2*G.X-q[0]),cp[1]-q[1]);
    const dmin=Math.min(dd,dm);
    if(dmin<bd){bd=dmin;best=name;}}
  return best;}
cv.addEventListener('pointerdown',ev=>{ev.preventDefault();const cp=canvasPt(ev);
  dragging=nearest(cp); if(dragging){cv.setPointerCapture(ev.pointerId); safeDraw();}});
cv.addEventListener('pointermove',ev=>{const cp=canvasPt(ev);
  if(dragging){const H=fixTailWide(handles()); const m=inv(cp[0],cp[1]);
    H[dragging].apply(m[0], m[1]);
    syncSliders(); safeDraw();}
  else{const n=nearest(cp); if(n!==hover){hover=n; safeDraw();}}});
['pointerup','pointercancel'].forEach(t=>cv.addEventListener(t,()=>{dragging=null; safeDraw();}));

// ---- sliders
function get(d,k){return (state[d]&&state[d][k]!==undefined)?state[d][k]:0;}
const sliders=document.querySelectorAll('input[type=range]');
function syncSliders(){sliders.forEach(s=>{const v=get(s.dataset.d,s.dataset.k);
  s.value=v; s.parentElement.querySelector('.val').textContent=(+v).toFixed(+s.step>=1?0:(+s.step>=0.1?1:2));});}
sliders.forEach(s=>{const d=s.dataset.d,k=s.dataset.k;
  s.addEventListener('input',()=>{if(!state[d])state[d]={}; state[d][k]=+s.value;
    s.parentElement.querySelector('.val').textContent=(+s.value).toFixed(+s.step>=1?0:(+s.step>=0.1?1:2)); safeDraw();});});
syncSliders();

function editsJSON(){
  const out=copyOf(BASE); delete out.rows;
  sliders.forEach(s=>{const d=s.dataset.d,k=s.dataset.k; if(d==='view'||/^row\d+$/.test(d))return;
    if(!out[d])out[d]={}; out[d][k]=state[d][k];});
  out.rows=rowsSpec();
  return out;
}
document.getElementById('copy').addEventListener('click',()=>{
  const txt=JSON.stringify(editsJSON(),null,1);
  const ta=document.getElementById('json'); ta.value=txt; ta.style.display='block';
  const done=()=>{document.getElementById('copied').textContent=' copied ✓';
    setTimeout(()=>document.getElementById('copied').textContent='',2000);};
  try{navigator.clipboard.writeText(txt).then(done,()=>{ta.select();});}catch(e){ta.select();}
});
document.getElementById('reset').addEventListener('click',()=>{ initState(FULL); syncSliders(); safeDraw(); });

// ---- the site: load the saved block, build it, save it (web/app.py /api/drawing/<order>)
const $=id=>document.getElementById(id);
function status(t,err){ const s=$('status'); s.textContent=t||''; s.className=err?'err':''; }
function payload(){ return {P:state.P, outline:state.outline, face:state.face, rows:rowsSpec().rows}; }
async function post(url,body){ const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  let j={}; try{ j=await r.json(); }catch(_){ } if(!r.ok) throw new Error(j.error||('HTTP '+r.status)); return j; }
if(SITE){
  (async()=>{ try{ const r=await fetch('/api/drawing/'+ORDER); if(!r.ok) throw new Error((await r.json()).error||('HTTP '+r.status));
      const j=await r.json(); FULL=j.full; BASE={P:j.drawing.params, outline:j.drawing.outline, face:j.drawing.face, rows:j.full.rows};
      initState(FULL); syncSliders(); safeDraw(); status('saved drawing loaded ('+(j.drawing.saved_at||'').slice(0,16).replace('T',' ')+')'); }
    catch(e){ status('could not load the saved drawing ('+e.message+'); showing the page’s copy',true); } })();
  let poll=null;
  $('build').onclick=async()=>{ const b=$('build'), s=$('save'); b.disabled=s.disabled=true; $('result').innerHTML=''; status('building…');
    poll=setInterval(async()=>{ try{ const p=await (await fetch('/api/progress')).json(); if(p.total) status(`building… ${p.done}/${p.total} pieces`); }catch(_){ } },1500);
    try{ const body=payload(); const m=await post('/api/drawing/'+ORDER+'/build',body); clearInterval(poll);
      const parts=(m.parts||[]); const bad=parts.filter(p=>p.error);
      const pr=m.print||{}; const touching=(pr.touching||[]).map(t=>`${t.a}–${t.b}`).join(', ');
      let h=`<table>${parts.map(p=>`<tr><td>${p.name}</td><td>${p.error?('<span style="color:#f66">'+p.error+'</span>'):((p.volume||0).toFixed(0)+' mm³')}</td><td>${p.watertight?'watertight ✓':''}</td><td>${p.stl?`<a href="${p.stl}">stl</a>`:''}</td></tr>`).join('')}</table>`;
      h+=`<div style="margin-top:6px">print ${pr.print_valid?'valid ✓':'—'} · rest contacts ${touching||'none'} · ${m.build_seconds||'?'} s` + (m.key?` · <a href="/api/stl/${m.key}">download STL (all pieces, print mm)</a>`:'') + `</div>`;
      $('result').innerHTML=h; status(bad.length?`built with ${bad.length} failed piece(s)`:'built ✓ — drawing the sheet…', bad.length>0);
      try{ const sh=await post('/api/drawing/'+ORDER+'/sheet',body); if(sh.url){ const im=document.createElement('img'); im.src=sh.url; $('result').appendChild(im); } status(bad.length?`built with ${bad.length} failed piece(s)`:'built ✓', bad.length>0); }
      catch(e){ status('built ✓ — sheet failed: '+e.message,true); }
    }catch(e){ clearInterval(poll); status('build failed: '+e.message,true); }
    b.disabled=s.disabled=false; };
  $('save').onclick=async()=>{ const b=$('build'), s=$('save'); b.disabled=s.disabled=true; status('saving…');
    try{ const j=await post('/api/drawing/'+ORDER,payload()); FULL=j.full; BASE={P:j.saved.params, outline:j.saved.outline, face:j.saved.face, rows:j.full.rows};
      initState(FULL); syncSliders(); safeDraw();
      status('saved ✓ into presets/'+ORDER+'.json'+(j.mirrored&&j.mirrored.head?` · studio head/body ${j.mirrored.head}`:(j.mirrored&&j.mirrored.error?` · studio mirror failed: ${j.mirrored.error}`:''))); }
    catch(e){ status('save failed: '+e.message,true); }
    b.disabled=s.disabled=false; };
}
safeDraw();
</script></body></html>
"""

SUB_SITE = ("Drag the dots on the drawing (right side) &mdash; each is a generator parameter, clamped to its real range. Sliders "
            "reach everything else. BUILD makes the real pieces (STL links and the blueprint sheet appear under the drawing). "
            "SAVE writes the state into the agnostida preset&rsquo;s drawing block and the studio&rsquo;s saved head/body "
            "<i>agnostida-drawing</i>. COPY EDITS JSON is the same state as an edits file.")
SUB_ALONE = ("Drag the dots on the drawing (right side) &mdash; each is a generator parameter, clamped to its real range. Sliders "
             "below reach everything else. COPY EDITS JSON &rarr; paste to Claude (or the site&rsquo;s /agnostid page) &rarr; real 3D build.")
NAV = '<div class="nav"><a href="/">studio</a><a href="/head">head</a><a href="/body">body</a><a href="/tree">tree</a></div>'
BUTTONS = '<button id="build">BUILD (real pieces + sheet)</button><button id="save">SAVE TO THE PRESET</button>'


def write(out, standalone=False):
    full, base, saved_at = data()
    ref = base64.b64encode(open(os.path.join(ROOT, "presets", f"{ORDER}_drawing.png"), "rb").read()).decode()
    html = (HTML.replace("__PANELS__", panels_html()).replace("__REF__", ref).replace("__ORDER__", ORDER)
                .replace("__CUR__", json.dumps(base)).replace("__FULL__", json.dumps(full))
                .replace("__SITE__", "false" if standalone else "true")
                .replace("__SUB__", SUB_ALONE if standalone else SUB_SITE)
                .replace("__NAV__", "" if standalone else NAV)
                .replace("__BUTTONS__", "" if standalone else BUTTONS))
    open(out, "w", encoding="utf-8").write(html)
    print("written", out, len(html), "bytes; block saved_at", saved_at)


if __name__ == "__main__":
    if "--standalone" in sys.argv: write(sys.argv[sys.argv.index("--standalone") + 1], standalone=True)
    else: write(os.path.join(ROOT, "web", "agnostid.html"))
