import json

with open("payload.json") as f:
    payload = f.read()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VARX Permutation Study Console</title>
<style>
:root{
  --paper:#ecf1f6; --panel:#ffffff; --ink:#14233c; --muted:#7b8aa0;
  --line:#d5deea; --cobalt:#2455c3; --cobalt-soft:#e3ebfb; --crimson:#c0392b;
  --good:#1d7a4f; --bad:#a03a2e; --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{background:var(--paper);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  line-height:1.45;padding:22px clamp(12px,3vw,40px) 40px}
a{color:var(--cobalt)}
header{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:14px}
h1{font-size:1.25rem;font-weight:650;letter-spacing:-.01em}
.meta{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.eq-panel{background:var(--ink);color:#dfe8f5;border-radius:8px;padding:14px 18px;margin-bottom:14px;
  font-family:var(--mono);font-size:.86rem;overflow-x:auto;white-space:nowrap}
.eq-panel .lhs{color:#fff;font-weight:600}
.eq-panel .tok{color:#9fb8e8}
.eq-panel .shock{color:#f0b429}
.eq-panel .purge{color:#e98d80;font-size:.72rem;vertical-align:super}
.eq-panel .dim{color:#5f7396}
.eq-caption{font-size:.68rem;color:#8ba0c2;margin-top:6px;font-family:var(--mono);white-space:normal}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin-bottom:14px}
.ctl{display:flex;flex-direction:column;gap:3px}
.ctl label{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:var(--panel);border:none;padding:5px 11px;font-size:.78rem;cursor:pointer;
  color:var(--ink);border-right:1px solid var(--line);font-family:var(--mono)}
.seg button:last-child{border-right:none}
.seg button[aria-pressed="true"]{background:var(--cobalt);color:#fff}
.seg button:focus-visible{outline:2px solid var(--cobalt);outline-offset:-2px}
.chk{display:flex;align-items:center;gap:6px;font-size:.78rem;margin-top:16px}
.grid{display:grid;grid-template-columns:minmax(430px,7fr) minmax(330px,5fr);gap:14px}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.panel h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:8px;font-weight:650}
table{width:100%;border-collapse:collapse;font-size:.76rem}
th{font-size:.64rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}
th.num,td.num{text-align:right;font-family:var(--mono)}
td{padding:4px 6px;border-bottom:1px solid #eef2f7;white-space:nowrap}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--cobalt-soft)}
tbody tr[aria-selected="true"]{background:var(--cobalt-soft);box-shadow:inset 3px 0 0 var(--cobalt)}
.bar{position:relative;background:#eef2f7;border-radius:3px;height:8px;min-width:70px}
.bar i{position:absolute;left:0;top:0;bottom:0;background:var(--cobalt);border-radius:3px}
.tag{display:inline-block;font-family:var(--mono);font-size:.66rem;background:#eef2f7;
  border-radius:4px;padding:1px 5px;margin-right:3px;color:#43536b}
.tag.s{background:#fdf3d7;color:#7a5b12}
.detail-head{font-family:var(--mono);font-size:.8rem;margin-bottom:8px;line-height:1.7}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
.kpi{border:1px solid var(--line);border-radius:6px;padding:7px 9px}
.kpi b{display:block;font-family:var(--mono);font-size:1.02rem;font-weight:600}
.kpi span{font-size:.62rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.kpi b.pos{color:var(--good)} .kpi b.neg{color:var(--bad)}
svg text{font-family:var(--mono);font-size:9px;fill:var(--muted)}
.legend{display:flex;gap:14px;font-size:.7rem;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.note{font-size:.72rem;color:var(--muted);margin-top:8px}
footer{margin-top:22px;font-size:.7rem;color:var(--muted);font-family:var(--mono)}
.diag{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:10px}
.diag b{color:var(--ink)}
@media (prefers-reduced-motion:no-preference){ tbody tr,.seg button{transition:background .12s} }
</style>
</head>
<body>
<header>
  <h1>VARX permutation study console</h1>
  <div class="meta" id="meta"></div>
</header>

<div class="eq-panel" id="eq"></div>

<div class="controls">
  <div class="ctl"><label>Forecast target</label><div class="seg" id="c-target"></div></div>
  <div class="ctl"><label>Horizon</label><div class="seg" id="c-h"></div></div>
  <div class="ctl"><label>Shock path</label><div class="seg" id="c-path"></div></div>
  <div class="ctl"><label>Evaluation window</label><div class="seg" id="c-sample"></div></div>
  <div class="ctl"><label>Lag rule</label><div class="seg" id="c-rule"></div></div>
  <div class="ctl"><label>Shock treatment</label><div class="seg" id="c-treat"></div></div>
  <div class="ctl chk"><input type="checkbox" id="c-valid" checked><label for="c-valid" style="text-transform:none;font-size:.78rem;letter-spacing:0;color:var(--ink)">Validity gate</label></div>
</div>

<div class="grid">
  <div class="panel">
    <h2 id="lb-title">Leaderboard</h2>
    <table id="tbl">
      <thead><tr>
        <th>Add-ons</th><th>Shocks</th><th>Treat</th><th>Rule</th><th class="num">p</th>
        <th class="num" data-k="rmse">RMSE &#9662;</th><th></th>
        <th class="num" data-k="mae">MAE</th>
        <th class="num" data-k="r2">R2oos</th>
        <th class="num" data-k="dm">DM p</th>
      </tr></thead>
      <tbody id="tb"></tbody>
    </table>
    <div class="note" id="lb-note"></div>
  </div>
  <div class="panel">
    <h2>Selected model</h2>
    <div class="detail-head" id="d-head">Select a row to inspect a model.</div>
    <div class="kpis" id="d-kpis"></div>
    <div id="d-chart"></div>
    <div class="legend" id="d-legend"></div>
    <div class="diag" id="d-diag"></div>
  </div>
</div>

<footer>Nicholas Hong | Built for educational and research purposes. Not financial advice.</footer>

<script>
const DATA = __PAYLOAD__;

const TL = {gdp:"GDP growth", inf:"Core inflation", stance:"Policy stance", fx:"Broad dollar"};
const SHOCKSYM = {oil:"oil", inv:"inv", wealth:"nw", fiscal:"gov"};
const TREATL = {raw:"raw", gdp_resid:"gdp purge", full_resid:"full purge", none:""};
const state = {target:"gdp", h:1, path:"uncond", sample:"all", rule:"all", treat:"all",
               validOnly:true, sortKey:"rmse", selected:null};

document.getElementById("meta").textContent =
  DATA.meta.sample[0] + " to " + DATA.meta.sample[1] + " | train end " + DATA.meta.train_end +
  " | " + DATA.meta.n_origins + " origins | " + DATA.specs.length + " spec rows";

function seg(id, opts, key, fmt){
  const el = document.getElementById(id);
  opts.forEach(o=>{
    const b = document.createElement("button");
    b.textContent = fmt ? fmt(o) : o;
    b.dataset.v = o;
    b.setAttribute("aria-pressed", String(state[key])===String(o));
    b.onclick = ()=>{ state[key] = (key==="h") ? +o : o;
      el.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed", x.dataset.v==String(o)));
      state.selected = null; render(); };
    el.appendChild(b);
  });
}
seg("c-target", ["gdp","inf","stance","fx"], "target", o=>TL[o]);
seg("c-h", ["1","4","8"], "h", o=>"h="+o);
seg("c-path", ["uncond","realized"], "path");
seg("c-sample", ["all","excovid"], "sample", o=>o==="all"?"full":"ex COVID");
seg("c-rule", ["all","aic","bic","hqic"], "rule");
seg("c-treat", ["all","raw","gdp_resid","full_resid"], "treat", o=>o==="all"?"all":TREATL[o]);
document.getElementById("c-valid").onchange = e=>{ state.validOnly = e.target.checked; render(); };

function metric(sid){
  const m = DATA.metrics[String(sid)];
  if(!m) return null;
  const key = state.path + "_" + state.target + "_" + state.h + "_" + state.sample;
  const alt = "uncond_" + state.target + "_" + state.h + "_" + state.sample;
  return m[key] || m[alt] || null;   // shockless specs only store uncond
}

function rows(){
  let out = [];
  for(const s of DATA.specs){
    if(state.rule!=="all" && s.rule!==state.rule) continue;
    if(state.treat!=="all" && s.treat!==state.treat && !(state.treat==="raw" && s.treat==="none")) continue;
    if(state.path==="realized" && s.shocks.length===0) continue;
    const m = metric(s.sid);
    if(!m) continue;
    if(state.validOnly && !m[4]) continue;
    out.push({s, rmse:m[0], mae:m[1], r2:m[2], dm:m[3], valid:m[4]});
  }
  if(state.rule==="all"){                    // collapse identical models across rules
    const seen = new Map();
    for(const r of out){
      const k = r.s.mid;
      if(seen.has(k)) seen.get(k).rules.push(r.s.rule);
      else { r.rules=[r.s.rule]; seen.set(k,r); }
    }
    out = [...seen.values()];
  } else out.forEach(r=>r.rules=[r.s.rule]);
  const k = state.sortKey;
  out.sort((a,b)=> k==="r2" ? (b.r2-a.r2) : ((a[k]??9e9)-(b[k]??9e9)));
  return out;
}

function fmtShocks(s){
  if(!s.shocks.length) return '<span class="tag">no shocks</span>';
  return s.shocks.map(x=>'<span class="tag s">'+SHOCKSYM[x]+'</span>').join("") +
    (s.treat!=="raw" && s.treat!=="none" ? ' <span class="tag">'+TREATL[s.treat]+'</span>' : "");
}

function render(){
  const rs = rows();
  document.getElementById("lb-title").textContent =
    "Leaderboard: " + TL[state.target] + ", h=" + state.h + ", " +
    (state.path==="uncond"?"unconditional":"realized shock path") +
    ", " + (state.sample==="all"?"full window":"ex COVID");
  const top = rs.slice(0, 30);
  if(state.selected===null && top.length) state.selected = top[0].s.sid;
  const maxR = Math.max(...top.map(r=>r.rmse), 1e-9);
  const tb = document.getElementById("tb");
  tb.innerHTML = top.map(r=>{
    const s = r.s;
    return '<tr data-sid="'+s.sid+'" aria-selected="'+(s.sid===state.selected)+'">'+
      '<td>'+(s.addons.length?s.addons.join(", "):"core")+'</td>'+
      '<td>'+fmtShocks(s)+'</td>'+
      '<td>'+(TREATL[s.treat]||"")+'</td>'+
      '<td>'+r.rules.join(",")+'</td>'+
      '<td class="num">'+s.p+'</td>'+
      '<td class="num">'+r.rmse.toFixed(3)+'</td>'+
      '<td><div class="bar"><i style="width:'+(100*r.rmse/maxR).toFixed(1)+'%"></i></div></td>'+
      '<td class="num">'+r.mae.toFixed(3)+'</td>'+
      '<td class="num" style="color:'+(r.r2>0?"var(--good)":"var(--bad)")+'">'+r.r2.toFixed(3)+'</td>'+
      '<td class="num">'+(r.dm==null?"":r.dm.toFixed(3))+'</td></tr>';
  }).join("");
  tb.querySelectorAll("tr").forEach(tr=> tr.onclick = ()=>{ state.selected=+tr.dataset.sid; render(); });
  document.getElementById("lb-note").textContent =
    rs.length + " specs match. Top 30 shown. RMSE and MAE in annualized points. " +
    "R2oos is against an AR(BIC) benchmark; DM p tests squared error against the core VAR.";
  renderEq(); renderDetail(rs);
}

function renderEq(){
  const sel = DATA.specs.find(x=>x.sid===state.selected);
  const endog = sel ? sel.endog : ["gdp","inf","stance"];
  const shocks = sel ? sel.shocks : [];
  const treat = sel ? sel.treat : "none";
  const p = sel ? sel.p : 2;
  const yv = "[" + endog.map(e=>'<span class="tok">'+e+'</span>').join(", ") + "]";
  let xpart = "";
  if(shocks.length){
    const purge = treat==="gdp_resid" ? '<span class="purge">gdp purge</span>' :
                  treat==="full_resid" ? '<span class="purge">full purge</span>' : "";
    const xv = "[" + shocks.map(x=>'<span class="shock">'+SHOCKSYM[x]+'</span>').join(", ") + "]" + purge;
    xpart = ' + B<sub>0</sub>'+xv+'<sub>t</sub> + B<sub>1</sub>'+xv+'<sub>t-1</sub>';
  }
  document.getElementById("eq").innerHTML =
    '<span class="lhs">Y<sub>t</sub></span> <span class="dim">=</span> c ' +
    '<span class="dim">+</span> &Sigma;<sub>i=1..'+p+'</sub> A<sub>i</sub> Y<sub>t-i</sub>' +
    xpart + ' <span class="dim">+</span> &Gamma;D<sub>t</sub> <span class="dim">+</span> &epsilon;<sub>t</sub>' +
    '<span class="dim">, &nbsp; Y = </span>' + yv +
    '<div class="eq-caption">The equation tracks the selected model. Shocks in amber enter with lags 0 and 1; ' +
    'D holds the 2020Q2 and 2020Q3 dummies; purge superscripts mark training-window residualization.</div>';
}

function renderDetail(rs){
  const r = rs.find(x=>x.s.sid===state.selected) ||
            (()=>{ const s = DATA.specs.find(x=>x.sid===state.selected);
                   if(!s) return null; const m = metric(s.sid);
                   return m ? {s, rmse:m[0], mae:m[1], r2:m[2], dm:m[3], rules:[s.rule]} : null; })();
  const head = document.getElementById("d-head"), kp = document.getElementById("d-kpis"),
        ch = document.getElementById("d-chart"), lg = document.getElementById("d-legend"),
        dg = document.getElementById("d-diag");
  if(!r){ head.textContent="No model matches the current filters."; kp.innerHTML=ch.innerHTML=lg.innerHTML=dg.innerHTML=""; return; }
  const s = r.s;
  head.innerHTML = "endog: <b>"+s.endog.join(" + ")+"</b><br>shocks: <b>"+
    (s.shocks.length?s.shocks.join(" + ")+" ("+ (TREATL[s.treat]||"raw") +")":"none")+
    "</b><br>lag rule "+r.rules.join(",")+" &rarr; p="+s.p;
  kp.innerHTML =
    '<div class="kpi"><b>'+r.rmse.toFixed(3)+'</b><span>RMSE</span></div>'+
    '<div class="kpi"><b>'+r.mae.toFixed(3)+'</b><span>MAE</span></div>'+
    '<div class="kpi"><b class="'+(r.r2>0?"pos":"neg")+'">'+r.r2.toFixed(3)+'</b><span>R2 oos</span></div>'+
    '<div class="kpi"><b>'+(r.dm==null?"n/a":r.dm.toFixed(3))+'</b><span>DM p</span></div>';
  const chartH = state.h===8 ? 4 : state.h;
  const key = state.path+"_"+state.target+"_"+chartH;
  const altKey = "uncond_"+state.target+"_"+chartH;
  const ser = (DATA.series[String(s.mid)]||{})[key] || (DATA.series[String(s.mid)]||{})[altKey];
  const act = DATA.actual[state.target+"_"+chartH];
  if(ser && act){
    ch.innerHTML = lineChart(act.dates, act.y, ser, state.path==="realized" ? "#c0392b" : "#2455c3");
    lg.innerHTML = '<span><i style="background:#14233c"></i>actual</span>'+
      '<span><i style="background:'+(state.path==="realized"?"#c0392b":"#2455c3")+'"></i>forecast, h='+chartH+
      (state.h===8?' (h=8 series not stored; metrics above are h=8)':'')+'</span>';
  } else { ch.innerHTML=""; lg.innerHTML=""; }
  dg.innerHTML = "diagnostics on the initial window: max eigenvalue <b>"+s.stab+
    "</b>, min Ljung-Box p <b>"+s.lb+"</b>, obs per parameter <b>"+s.dof+"</b>";
}

function lineChart(dates, a, f, fcColor){
  const W=440,H=190,L=34,R=6,T=10,B=22;
  const n=Math.min(a.length,f.length);
  const ys=a.slice(0,n).concat(f.slice(0,n));
  let lo=Math.min(...ys), hi=Math.max(...ys);
  const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
  const x=i=>L+(W-L-R)*i/(n-1), y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const path=arr=>arr.slice(0,n).map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1)).join(" ");
  let ticks="";
  for(let k=0;k<4;k++){
    const v=lo+(hi-lo)*k/3, yy=y(v);
    ticks+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+yy+'" y2="'+yy+'" stroke="#e3e9f1" stroke-width="1"/>'+
           '<text x="'+(L-4)+'" y="'+(yy+3)+'" text-anchor="end">'+v.toFixed(1)+'</text>';
  }
  let xt="";
  [0, Math.floor(n/2), n-1].forEach(i=>{
    xt+='<text x="'+x(i)+'" y="'+(H-6)+'" text-anchor="middle">'+dates[i]+'</text>';
  });
  return '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Forecast against actual">'+ticks+xt+
    '<path d="'+path(a)+'" fill="none" stroke="#14233c" stroke-width="1.4"/>'+
    '<path d="'+path(f)+'" fill="none" stroke="'+fcColor+'" stroke-width="1.4"/></svg>';
}

document.querySelectorAll("th[data-k]").forEach(th=>{
  th.onclick=()=>{ state.sortKey=th.dataset.k; render(); };
});

render();
</script>
</body>
</html>
"""

html = HTML.replace("__PAYLOAD__", payload)
with open("dashboard.html", "w") as f:
    f.write(html)
import os
print("dashboard MB", round(os.path.getsize("dashboard.html") / 1e6, 2))
