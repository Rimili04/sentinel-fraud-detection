/* Sentinel dashboard - vanilla JS, custom SVG charts (no external libraries) */
const $ = (s, r = document) => r.querySelector(s);
const fmt = n => Number(n).toLocaleString('en-US');
const money = n => '$' + Math.round(n).toLocaleString('en-US');
const money2 = n => '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const moneyS = n => n >= 1e6 ? '$' + (n / 1e6).toFixed(2) + 'M' : n >= 1e4 ? '$' + (n / 1e3).toFixed(0) + 'K' : money(n);
const pctAx = v => (+(v * 100).toFixed(1)) + '%';
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COL = { High: '#ff2e44', Medium: '#ff7a59', Low: '#3a82ff' };
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

let JOB = null, S = null, MODEL = null, TAB = 'landing';
const T = { tier: 'suspicious', q: '', sort: 'risk_score', dir: 'desc', page: 1 };
let uid = 0;

/* ------------------------------------------------------------------ tooltip */
const tip = $('#tip');
document.addEventListener('mousemove', e => {
  const el = e.target.closest ? e.target.closest('[data-tip]') : null;
  if (!el) { tip.classList.remove('on'); return; }
  tip.innerHTML = el.getAttribute('data-tip');
  tip.classList.add('on');
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});

/* ------------------------------------------------------------------- charts */
function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const f = v => Math.max(0, Math.min(255, Math.round(v + (amt > 0 ? (255 - v) * amt : v * amt))));
  return '#' + [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(f).map(v => v.toString(16).padStart(2, '0')).join('');
}
function grad(id, color) {
  return `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${shade(color, .38)}"/><stop offset=".55" stop-color="${color}"/><stop offset="1" stop-color="${shade(color, -.35)}"/></linearGradient>`;
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

function barChart(el, o) {
  const W = el.clientWidth || 600, H = o.h || 230, L = o.left || 40, R = 8, Tp = 12, B = 26;
  const pw = W - L - R, ph = H - Tp - B, n = o.labels.length, id = 'g' + (++uid);
  const sums = o.labels.map((_, i) => o.stacked ? o.series.reduce((a, s) => a + s.values[i], 0) : Math.max(...o.series.map(s => s.values[i])));
  const max = o.max || niceMax(Math.max(...sums, 0.0001));
  const yf = v => o.sqrt ? Math.sqrt(v / max) : v / max;
  const yfmt = o.yfmt || fmt;
  let defs = o.series.map((s, k) => grad(id + k, s.color)).join('');
  let svg = '', ticks = o.sqrt ? [0, max / 16, max / 4, max] : [0, .25, .5, .75, 1].map(t => t * max);
  ticks.forEach(v => {
    const y = Tp + ph * (1 - yf(v));
    svg += `<line class="grid-l" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="ax" x="${L - 6}" y="${y + 3.5}" text-anchor="end">${yfmt(v)}</text>`;
  });
  const slot = pw / n, bw = Math.max(2, slot * (o.gap || .68));
  const step = Math.max(1, Math.ceil(n * (o.labelW || 34) / pw));
  o.labels.forEach((lab, i) => {
    const x = L + slot * i + (slot - bw) / 2;
    let acc = 0;
    const tt = `<b>${esc(lab)}</b>` + o.series.map(s => `<br><span class="dot" style="background:${s.color}"></span> ${esc(s.name)}: <b>${(o.vfmt || fmt)(s.values[i])}</b>`).join('') + (o.extra ? '<br>' + o.extra(i) : '');
    o.series.forEach((s, k) => {
      const v = s.values[i];
      if (v <= 0) return;
      const base = o.stacked ? acc : 0;
      const y1 = Tp + ph * (1 - yf(base + v)), y0 = Tp + ph * (1 - yf(base));
      const x2 = o.stacked ? x : x + bw / o.series.length * k, w2 = o.stacked ? bw : bw / o.series.length - 1;
      svg += `<rect class="bar" x="${x2.toFixed(1)}" y="${y1.toFixed(1)}" width="${Math.max(w2, 1).toFixed(1)}" height="${Math.max(y0 - y1, 1).toFixed(1)}" rx="2" fill="url(#${id}${k})" data-tip="${esc(tt)}"/>`;
      if (o.stacked) acc += v;
    });
    // invisible hit area for thin/empty bars
    svg += `<rect x="${(L + slot * i).toFixed(1)}" y="${Tp}" width="${slot.toFixed(1)}" height="${ph}" fill="transparent" data-tip="${esc(tt)}"/>`;
    if (i % step === 0) svg += `<text class="ax" x="${(x + bw / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle">${esc(o.xfmt ? o.xfmt(lab, i) : lab)}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || 'chart')}"><defs>${defs}</defs>${svg}</svg>`;
}

function lineChart(el, o) {
  const W = el.clientWidth || 600, H = o.h || 200, L = 44, R = 10, Tp = 12, B = 26;
  const pw = W - L - R, ph = H - Tp - B, n = o.labels.length, id = 'l' + (++uid);
  const max = niceMax(Math.max(...o.values, 0.0001));
  const X = i => L + (n === 1 ? pw / 2 : pw * i / (n - 1)), Y = v => Tp + ph * (1 - v / max);
  let svg = '';
  [0, .25, .5, .75, 1].forEach(t => {
    const y = Tp + ph * (1 - t);
    svg += `<line class="grid-l" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="ax" x="${L - 6}" y="${y + 3.5}" text-anchor="end">${o.yfmt(t * max)}</text>`;
  });
  const pts = o.values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  svg += `<path d="M${X(0)},${Tp + ph} L${pts.join(' L')} L${X(n - 1)},${Tp + ph}Z" fill="url(#${id})" opacity=".9"/>`;
  svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${o.color}" stroke-width="2.4" stroke-linejoin="round" style="filter:drop-shadow(0 0 6px ${o.color})"/>`;
  const step = Math.max(1, Math.ceil(n * 30 / pw));
  o.labels.forEach((lab, i) => {
    svg += `<circle cx="${X(i).toFixed(1)}" cy="${Y(o.values[i]).toFixed(1)}" r="4" fill="${o.color}" opacity="0" data-tip="${esc(o.tip(i))}" style="cursor:default"/>`;
    svg += `<rect x="${(X(i) - pw / n / 2).toFixed(1)}" y="${Tp}" width="${(pw / n).toFixed(1)}" height="${ph}" fill="transparent" data-tip="${esc(o.tip(i))}"/>`;
    if (i % step === 0) svg += `<text class="ax" x="${X(i)}" y="${H - 8}" text-anchor="middle">${esc(lab)}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${o.color}" stop-opacity=".5"/><stop offset="1" stop-color="${o.color}" stop-opacity="0"/></linearGradient></defs>${svg}</svg>`;
}

function donut(el, parts, center, sub) {
  const R = 70, C = 2 * Math.PI * R, total = parts.reduce((a, p) => a + p.value, 0) || 1, id = 'd' + (++uid);
  let off = 0, svg = '', defs = '';
  parts.forEach((p, k) => {
    defs += grad(id + k, p.color);
    const len = Math.max(p.value / total * C - 2, 0);
    if (p.value > 0) svg += `<circle cx="90" cy="90" r="${R}" fill="none" stroke="url(#${id}${k})" stroke-width="22" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 90 90)" data-tip="<b>${esc(p.label)}</b><br>${fmt(p.value)} (${pct(p.value / total)})"/>`;
    off += p.value / total * C;
  });
  el.innerHTML = `<div class="donut-wrap"><svg viewBox="0 0 180 180"><defs>${defs}</defs><circle cx="90" cy="90" r="${R}" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="22"/>${svg}<text class="donut-c" x="90" y="92">${esc(center)}</text><text class="donut-s" x="90" y="110">${esc(sub)}</text></svg>
  <div class="legend" style="flex-direction:column;gap:10px">${parts.map(p => `<span><i class="dot" style="background:${p.color}"></i>${esc(p.label)} <b style="color:#fff">${fmt(p.value)}</b> <span style="color:var(--muted)">${pct(p.value / total, p.value / total < .1 ? 1 : 0)}</span></span>`).join('')}</div></div>`;
}

function hbars(el, rows, o = {}) {
  if (!rows.length) { el.innerHTML = '<div class="empty">No data</div>'; return; }
  const key = o.flag || 'flagged', tk = o.tot || 'total';
  const max = Math.max(...rows.map(r => r[tk]), 1);
  el.innerHTML = rows.map(r => {
    const lab = o.label ? o.label(r) : String(r.label).replace(/_/g, ' ');
    const val = o.value ? o.value(r) : `<b>${fmt(r[key])}</b> / ${fmt(r[tk])} &middot; ${pct(r[tk] ? r[key] / r[tk] : 0)}`;
    const tt = o.tip ? o.tip(r) : `<b>${esc(lab)}</b><br>${fmt(r[key])} flagged of ${fmt(r[tk])}<br>Flag rate ${pct(r[tk] ? r[key] / r[tk] : 0)}`;
    return `<div class="hb ${o.pos ? 'pos' : ''}" data-tip="${esc(tt)}"><div class="lab" title="${esc(lab)}">${esc(lab)}</div><div class="track"><div class="tot" style="width:${o.pos ? 0 : r[tk] / max * 100}%"></div><div class="fl" style="width:${(o.pos ? r[key] / max : r[key] / max) * 100}%"></div></div><div class="val">${val}</div></div>`;
  }).join('');
}

function heatmap(el, m, tot) {
  const max = Math.max(...m.flat(), 1);
  let h = '<div></div>' + [...Array(24).keys()].map(i => `<div class="h">${i % 3 === 0 ? String(i).padStart(2, '0') : ''}</div>`).join('');
  m.forEach((row, d) => {
    h += `<div class="d">${DAYS[d]}</div>`;
    row.forEach((v, hr) => {
      const a = v / max, bg = v === 0 ? 'rgba(58,130,255,.09)' : `rgba(255,${Math.round(46 + (1 - a) * 80)},${Math.round(68 + (1 - a) * 60)},${(.25 + a * .75).toFixed(2)})`;
      h += `<div class="cell" style="background:${bg}" data-tip="<b>${DAYS[d]} ${String(hr).padStart(2, '0')}:00</b><br>${v} flagged of ${tot[d][hr]}"></div>`;
    });
  });
  el.innerHTML = `<div class="heat">${h}</div>`;
}

function gauge(score, tier) {
  const c = COL[tier];
  return `<svg viewBox="0 0 140 92"><defs>${grad('gg', c)}</defs><path d="M12 76 A58 58 0 0 1 128 76" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="13" stroke-linecap="round"/>
  <path d="M12 76 A58 58 0 0 1 128 76" pathLength="100" fill="none" stroke="url(#gg)" stroke-width="13" stroke-linecap="round" stroke-dasharray="${Math.max(score, .5)} 100" style="filter:drop-shadow(0 0 8px ${c})"/>
  <text x="70" y="70" text-anchor="middle" style="font-family:var(--display);font-size:30px;font-weight:600;fill:#fff">${Math.round(score)}</text><text x="70" y="86" text-anchor="middle" class="ax">risk score</text></svg>`;
}

/* --------------------------------------------------------------------- data */
async function api(url, opts) {
  const r = await fetch(url, opts);
  let j = {};
  try { j = await r.json(); } catch (e) { }
  if (!r.ok) throw new Error(j.error || 'Request failed (' + r.status + ')');
  return j;
}
function loading(on, msg) { $('#loader').hidden = !on; if (msg) $('#loadmsg').textContent = msg; }

async function runAnalysis(promise) {
  $('#landing-err').innerHTML = '';
  loading(true);
  try {
    const d = await promise;
    JOB = d.job_id; S = d.summary;
    Object.assign(T, { tier: 'suspicious', q: '', sort: 'risk_score', dir: 'desc', page: 1 });
    $('#dl-sus').href = `/api/job/${JOB}/download/suspicious.csv`;
    $('#dl-full').href = `/api/job/${JOB}/download/full.csv`;
    $('#actions').hidden = false;
    document.querySelectorAll('#nav button').forEach(b => b.disabled = false);
    setTab('overview');
  } catch (e) {
    setTab('landing');
    $('#landing-err').innerHTML = `<div class="alert">${esc(e.message)}</div>`;
  } finally { loading(false); }
}
function upload(file) {
  if (!file) return;
  const fd = new FormData(); fd.append('file', file);
  runAnalysis(api('/api/analyze', { method: 'POST', body: fd }));
}

/* -------------------------------------------------------------------- tabs */
const TITLES = {
  landing: ['Batch fraud analytics', 'Upload a transaction file and score every row in one pass.'],
  overview: ['Fraud overview', ''], patterns: ['Fraud patterns', 'Where, when and how suspicious activity clusters in this batch.'],
  transactions: ['Suspicious transactions', 'Search, sort and open any flagged transaction for a full explanation.'],
  model: ['Model', 'How scores are produced and how well they perform on unseen data.'],
};
function setTab(t) {
  TAB = t;
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  $('#landing').hidden = t !== 'landing';
  $('#dash').hidden = !['overview', 'patterns', 'transactions'].includes(t);
  ['overview', 'patterns', 'transactions'].forEach(x => $('#tab-' + x).hidden = x !== t);
  $('#tab-model').hidden = t !== 'model';
  let [a, b] = TITLES[t];
  if (t === 'overview' && S) b = `${S.source} \u00b7 ${fmt(S.n_transactions)} transactions \u00b7 ${S.date_from.slice(0, 10)} to ${S.date_to.slice(0, 10)}`;
  $('#title').textContent = a; $('#subtitle').textContent = b;
  renderTab();
  scrollTo({ top: 0 });
}
function renderTab() {
  if (TAB === 'overview' && S) renderOverview();
  else if (TAB === 'patterns' && S) renderPatterns();
  else if (TAB === 'transactions' && S) renderTransactions();
  else if (TAB === 'model') renderModel();
}
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(renderTab, 150); });

/* ---------------------------------------------------------------- overview */
function renderOverview() {
  const s = S, c = s.charts, el = $('#tab-overview');
  const hAmt = s.high_amount, mAmt = s.flagged_amount - s.high_amount, lAmt = s.total_amount - s.flagged_amount;
  const ev = s.evaluation;
  el.innerHTML = `
  ${s.notes.map(n => `<div class="alert" style="margin:0 0 14px;background:linear-gradient(180deg,rgba(58,130,255,.22),rgba(58,130,255,.08));box-shadow:inset 0 0 0 1px rgba(58,130,255,.5);color:#d4e3ff">${esc(n)}</div>`).join('')}
  <div class="grid g-hero">
    <div class="panel">
      <h3>Exposure under review</h3><div class="sub">Value of transactions scored medium or high risk</div>
      <div class="hero-num">${money(s.flagged_amount)}</div>
      <div class="hero-sub">${pct(s.flagged_amount / Math.max(s.total_amount, 1))} of ${moneyS(s.total_amount)} processed volume</div>
      <div class="tierbar" role="img" aria-label="Share of transaction value by risk tier">
        <i class="c-high" style="width:${hAmt / s.total_amount * 100}%" data-tip="<b>High risk</b><br>${money(hAmt)}"></i>
        <i class="c-med" style="width:${mAmt / s.total_amount * 100}%" data-tip="<b>Medium risk</b><br>${money(mAmt)}"></i>
        <i class="c-low" style="width:${lAmt / s.total_amount * 100}%" data-tip="<b>Low risk</b><br>${money(lAmt)}"></i></div>
      <div class="legend"><span><i class="dot c-high"></i>High ${money(hAmt)}</span><span><i class="dot c-med"></i>Medium ${money(mAmt)}</span><span><i class="dot c-low"></i>Low ${moneyS(lAmt)}</span></div>
      <div class="strip">
        <div><b>${fmt(s.n_transactions)}</b><span>transactions scored</span></div>
        <div><b>${fmt(s.n_flagged)}</b><span>flagged (${pct(s.flag_rate)})</span></div>
        <div><b style="color:#ff6173">${fmt(s.n_high)}</b><span>high risk</span></div>
        <div><b>${s.seconds}s</b><span>batch scoring time</span></div>
      </div>
    </div>
    <div class="panel"><h3>Risk score distribution</h3><div class="sub">Every transaction by score, square-root scale so the tail stays visible</div><div class="chart" id="c-hist"></div>
      <div class="legend" style="margin-top:6px"><span><i class="dot c-low"></i>Low (&lt;${(s.thresholds.medium * 100).toFixed(0)})</span><span><i class="dot c-med"></i>Medium (${(s.thresholds.medium * 100).toFixed(0)}&ndash;${(s.thresholds.high * 100).toFixed(0)})</span><span><i class="dot c-high"></i>High (${(s.thresholds.high * 100).toFixed(0)}+)</span></div></div>
  </div>
  <div class="grid g-7-5">
    <div class="panel"><h3>Flagged transactions per day</h3><div class="sub">High and medium risk, stacked</div><div class="chart" id="c-daily"></div></div>
    <div class="panel"><h3>Fraud vs normal</h3><div class="sub">Share of all transactions by predicted class</div><div id="c-donut"></div>
      <p class="note">${fmt(s.n_users)} users in this batch. Average risk score ${s.avg_score.toFixed(1)}.</p></div>
  </div>
  ${ev ? `<div class="panel" style="margin-bottom:16px"><h3>Check against your labels</h3><div class="sub">Your file includes an is_fraud column, so predictions can be graded</div>
    <div class="mgrid">
      <div class="mtile"><b>${fmt(ev.actual_fraud)}</b><span>actual fraud in file (${money(ev.fraud_amount)})</span></div>
      <div class="mtile"><b style="color:#7fb0ff">${pct(ev.medium_plus.recall)}</b><span>caught when flagging medium + high (${fmt(ev.medium_plus.tp)})</span></div>
      <div class="mtile"><b style="color:#7fb0ff">${pct(ev.medium_plus.precision)}</b><span>of flagged are real fraud (${fmt(ev.medium_plus.fp)} false alarms)</span></div>
      <div class="mtile"><b style="color:#ff6173">${pct(ev.high_only.precision)}</b><span>precision on high-risk only, recall ${pct(ev.high_only.recall)}</span></div>
    </div>
    ${ev.by_type ? `<div style="margin-top:16px" id="c-evtype"></div>` : ''}</div>` : ''}
  <div class="grid g-7-5">
    <div class="panel"><h3>Key findings</h3><div class="sub">Generated from this batch for the management summary</div><ul class="ins">${s.insights.map(i => `<li>${esc(i)}</li>`).join('')}</ul></div>
    <div class="panel"><h3>Most exposed users</h3><div class="sub">By flagged value</div><div class="tw mini"><table><thead><tr><th>User</th><th class="num">Flagged</th><th class="num">Value</th><th class="num">Top score</th></tr></thead><tbody>
      ${c.users.slice(0, 8).map(u => `<tr data-user="${esc(u.user_id)}"><td class="mono">${esc(u.user_id)}</td><td class="num">${u.flagged} / ${u.total}</td><td class="num">${money(u.flagged_amount)}</td><td class="num"><b>${Math.round(u.max_score)}</b></td></tr>`).join('')}</tbody></table></div>
      <p class="note">Click a user to search their flagged transactions.</p></div>
  </div>`;
  barChart($('#c-hist'), {
    labels: c.histogram.map(b => b.lo + '-' + b.hi), xfmt: l => l.split('-')[0], stacked: true, sqrt: true, h: 240, gap: .78, labelW: 26,
    series: [{ name: 'Low', color: COL.Low, values: c.histogram.map(b => b.Low) }, { name: 'Medium', color: COL.Medium, values: c.histogram.map(b => b.Medium) }, { name: 'High', color: COL.High, values: c.histogram.map(b => b.High) }],
    left: 46, aria: 'Risk score histogram'
  });
  barChart($('#c-daily'), {
    labels: c.daily.map(d => d.label), xfmt: l => l.slice(5), stacked: true, h: 250,
    series: [{ name: 'High', color: COL.High, values: c.daily.map(d => d.high) }, { name: 'Medium', color: COL.Medium, values: c.daily.map(d => d.flagged - d.high) }],
    extra: i => `${fmt(c.daily[i].total)} transactions that day`
  });
  donut($('#c-donut'), [{ label: 'High risk', value: s.n_high, color: COL.High }, { label: 'Medium risk', value: s.n_medium, color: COL.Medium }, { label: 'Normal', value: s.n_low, color: COL.Low }], pct(s.flag_rate), 'flagged');
  if (ev && ev.by_type) hbars($('#c-evtype'), ev.by_type.map(r => ({ label: r.label, flagged: r.caught, total: r.count })), { value: r => `<b>${r.flagged}</b> / ${r.total} caught &middot; ${pct(r.flagged / r.total, 0)}` });
  el.querySelectorAll('tr[data-user]').forEach(tr => tr.onclick = () => { T.q = tr.dataset.user; T.page = 1; T.tier = 'suspicious'; setTab('transactions'); });
}

/* ---------------------------------------------------------------- patterns */
function renderPatterns() {
  const c = S.charts, el = $('#tab-patterns');
  el.innerHTML = `
  <div class="grid g2">
    <div class="panel"><h3>Fraud by hour of day</h3><div class="sub">Flagged transactions, stacked by tier</div><div class="chart" id="p-hours"></div></div>
    <div class="panel"><h3>Flag rate by hour</h3><div class="sub">Share of each hour's transactions that were flagged</div><div class="chart" id="p-hrate"></div></div>
  </div>
  <div class="panel" style="margin-bottom:16px"><h3>When it happens</h3><div class="sub">Flagged transactions by weekday and hour</div><div id="p-heat"></div></div>
  <div class="grid g2">
    <div class="panel"><h3>Location: cities</h3><div class="sub">Red = flagged, blue = all transactions</div><div id="p-city"></div></div>
    <div class="panel"><h3>Location: countries</h3><div class="sub">Red = flagged, blue = all transactions</div><div id="p-country"></div></div>
  </div>
  <div class="grid g3">
    <div class="panel"><h3>Transaction type</h3><div class="sub">Channel risk</div><div id="p-type"></div></div>
    <div class="panel"><h3>Merchant category</h3><div class="sub">Top categories by flagged count</div><div id="p-cat"></div></div>
    <div class="panel"><h3>Device</h3><div class="sub">Device type behind the transaction</div><div id="p-dev"></div></div>
  </div>
  <div class="grid g-7-5">
    <div class="panel"><h3>Amount distribution</h3><div class="sub">Flag rate by transaction amount band</div><div class="chart" id="p-amt"></div></div>
    <div class="panel"><h3>Behaviour signals</h3><div class="sub">Flag rate climbs with failed attempts and bursts of activity</div>
      <div class="sub2">Failed attempts in 24h</div><div class="chart" id="p-fail"></div>
      <div class="sub2" style="margin-top:10px">User transactions within one hour</div><div class="chart" id="p-vel"></div></div>
  </div>`;
  barChart($('#p-hours'), {
    labels: c.hours.map(h => String(h.hour).padStart(2, '0')), stacked: true, h: 230, labelW: 22,
    series: [{ name: 'High', color: COL.High, values: c.hours.map(h => h.high) }, { name: 'Medium', color: COL.Medium, values: c.hours.map(h => h.flagged - h.high) }],
    extra: i => `${fmt(c.hours[i].total)} total transactions`
  });
  lineChart($('#p-hrate'), {
    labels: c.hours.map(h => String(h.hour).padStart(2, '0')), values: c.hours.map(h => h.rate), color: '#ff2e44', h: 230, yfmt: pctAx,
    tip: i => `<b>${String(c.hours[i].hour).padStart(2, '0')}:00</b><br>Flag rate <b>${pct(c.hours[i].rate)}</b><br>${c.hours[i].flagged} of ${fmt(c.hours[i].total)}`
  });
  heatmap($('#p-heat'), c.heatmap, c.heatmap_total);
  hbars($('#p-city'), c.cities); hbars($('#p-country'), c.countries);
  hbars($('#p-type'), c.types); hbars($('#p-cat'), c.categories); hbars($('#p-dev'), c.devices);
  const rateBars = (id, rows, color, h) => barChart($(id), {
    labels: rows.map(r => r.label), series: [{ name: 'Flag rate', color, values: rows.map(r => r.rate) }], h, left: 44, labelW: 50,
    vfmt: v => pct(v), yfmt: pctAx, gap: .6, extra: i => `${fmt(rows[i].flagged)} flagged of ${fmt(rows[i].total)}`
  });
  rateBars('#p-amt', c.amount, COL.High, 250); rateBars('#p-fail', c.failed, COL.Medium, 130); rateBars('#p-vel', c.velocity, COL.High, 130);
}

/* ------------------------------------------------------------ transactions */
function renderTransactions() {
  const el = $('#tab-transactions');
  if (!el.dataset.ready) {
    el.dataset.ready = '1';
    el.innerHTML = `<div class="panel">
      <div class="tools">
        <input type="search" id="t-q" placeholder="Search ID, user, city, merchant, reason..." aria-label="Search suspicious transactions">
        <div class="seg" id="t-tier"><button data-t="suspicious">Suspicious</button><button data-t="High">High</button><button data-t="Medium">Medium</button><button data-t="all">All</button></div>
        <span class="sp" id="t-count"></span></div>
      <div class="tw"><table><thead><tr id="t-head"></tr></thead><tbody id="t-body"></tbody></table></div>
      <div class="pager"><button class="btn" id="t-prev">Previous</button><span id="t-page"></span><button class="btn" id="t-next">Next</button></div></div>`;
    let dbt;
    $('#t-q').addEventListener('input', e => { clearTimeout(dbt); dbt = setTimeout(() => { T.q = e.target.value; T.page = 1; loadTable(); }, 250); });
    $('#t-tier').onclick = e => { const b = e.target.closest('button'); if (!b) return; T.tier = b.dataset.t; T.page = 1; loadTable(); };
    $('#t-prev').onclick = () => { if (T.page > 1) { T.page--; loadTable(); } };
    $('#t-next').onclick = () => { T.page++; loadTable(); };
  }
  $('#t-q').value = T.q;
  loadTable();
}
const COLS = [['risk_score', 'Risk'], ['transaction_id', 'Transaction'], ['amount', 'Amount'], ['transaction_type', 'Type'],
  ['merchant_category', 'Merchant'], ['city', 'City'], ['device_type', 'Device'], ['failed_attempts_24h', 'Failed'], ['', 'Why it was flagged']];
async function loadTable() {
  $('#t-tier').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.t === T.tier));
  $('#t-head').innerHTML = COLS.map(([k, l]) => `<th data-k="${k}" class="${k === T.sort ? 'sorted' : ''}">${l}${k === T.sort ? (T.dir === 'asc' ? ' \u25B2' : ' \u25BC') : ''}</th>`).join('');
  $('#t-head').querySelectorAll('th').forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (!k) return;
    if (T.sort === k) T.dir = T.dir === 'asc' ? 'desc' : 'asc'; else { T.sort = k; T.dir = 'desc'; }
    T.page = 1; loadTable();
  });
  const qs = new URLSearchParams({ tier: T.tier, q: T.q, sort: T.sort, dir: T.dir, page: T.page, size: 25 });
  let d;
  try { d = await api(`/api/job/${JOB}/transactions?${qs}`); }
  catch (e) { $('#t-body').innerHTML = `<tr><td colspan="9" class="empty">${esc(e.message)}</td></tr>`; return; }
  $('#t-count').textContent = `${fmt(d.total)} transactions`;
  $('#t-page').textContent = `Page ${d.page} of ${d.pages}`;
  $('#t-prev').disabled = d.page <= 1; $('#t-next').disabled = d.page >= d.pages;
  if (!d.rows.length) { $('#t-body').innerHTML = `<tr><td colspan="9" class="empty">No transactions match. Clear the search or switch the tier filter.</td></tr>`; return; }
  $('#t-body').innerHTML = d.rows.map(r => `<tr data-i="${r.i}">
    <td><span class="pill"><span style="color:${COL[r.risk_tier]}">${Math.round(r.risk_score)}</span><span class="m"><i style="width:${r.risk_score}%;background:${COL[r.risk_tier]}"></i></span><span class="tier ${r.risk_tier}">${r.risk_tier}</span></span></td>
    <td class="mono">${esc(r.transaction_id)}<br><span style="color:var(--muted)">${esc(r.user_id)} &middot; ${esc(r.timestamp.slice(5, 16))}</span></td>
    <td class="num"><b>${money2(r.amount)}</b></td><td>${esc(r.transaction_type)}</td><td>${esc(String(r.merchant_category).replace(/_/g, ' '))}</td>
    <td>${esc(r.city)}</td><td>${esc(String(r.device_type).replace(/_/g, ' '))}</td><td class="num">${r.failed_attempts_24h}</td><td class="why">${esc(r.top_reasons) || '<span style="color:var(--muted)">No unusual factors</span>'}</td></tr>`).join('');
  $('#t-body').querySelectorAll('tr[data-i]').forEach(tr => tr.onclick = () => openDrawer(+tr.dataset.i));
}

/* ------------------------------------------------------------------ drawer */
const drawer = $('#drawer'), scrim = $('#scrim');
function closeDrawer() { drawer.classList.remove('on'); scrim.classList.remove('on'); }
scrim.onclick = closeDrawer;
addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
async function openDrawer(i) {
  drawer.innerHTML = '<div class="empty">Explaining this transaction...</div>';
  drawer.classList.add('on'); scrim.classList.add('on');
  let d;
  try { d = await api(`/api/job/${JOB}/transaction/${i}`); } catch (e) { drawer.innerHTML = `<div class="alert">${esc(e.message)}</div>`; return; }
  const x = d.details, mx = Math.max(...d.factors.map(f => Math.abs(f.points)), 1);
  const yn = v => v ? 'Yes' : 'No';
  const kv = [['Transaction', x.transaction_id], ['Time', x.timestamp], ['User', x.user_id], ['Amount', money2(x.amount)],
    ['Type', x.transaction_type], ['Merchant', String(x.merchant_category).replace(/_/g, ' ')], ['Location', x.city + ', ' + x.country], ['Device', String(x.device_type).replace(/_/g, ' ')],
    ['Failed attempts (24h)', x.failed_attempts_24h], ['Account age', x.account_age_days + ' days'], ['Txns in last hour', x.txn_count_1h], ['Amount vs user average', x.amount_to_user_mean + 'x'],
    ['New device', yn(x.new_device)], ['Unusual city', yn(x.city_mismatch)]];
  if ('is_fraud' in x) kv.push(['Actual label', x.is_fraud == 1 ? 'Fraud' + (x.fraud_type ? ' (' + x.fraud_type + ')' : '') : 'Legitimate']);
  drawer.innerHTML = `
  <div class="dr-h"><div><h2>${esc(x.transaction_id)}</h2><small>${esc(x.timestamp)} &middot; ${esc(x.user_id)}</small></div><button class="x" id="dr-x" aria-label="Close"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>
  <div class="gauge">${gauge(d.risk_score, d.risk_tier)}<div><span class="tier ${d.risk_tier}">${d.risk_tier} risk</span><p class="act" style="margin-top:8px">${esc(d.action)}</p></div></div>
  <div class="sec"><h4>What drove this score</h4>
    <p class="note" style="margin:-4px 0 8px">Points moved relative to a typical transaction (about ${d.baseline_score}). Red pushes risk up, blue pulls it down.</p>
    ${d.factors.length ? d.factors.map(f => { const w = Math.abs(f.points) / mx * 36; return `<div class="fx"><div class="t"><b>${esc(f.label)}</b>${esc(f.detail)}</div><div class="dv"><i class="${f.points >= 0 ? 'p' : 'n'}" style="width:${w}%"></i><em style="${f.points >= 0 ? 'left:calc(50% + ' + w + '% + 6px)' : 'right:calc(50% + ' + w + '% + 6px)'}">${f.points > 0 ? '+' : ''}${f.points}</em></div></div>`; }).join('') : '<p class="note">No single factor stands out; the score is close to a typical transaction.</p>'}</div>
  <div class="sec"><h4>Transaction details</h4><div class="kv">${kv.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div></div>
  <div class="sec"><h4>Same user, nearby activity (${d.user_summary.flagged} flagged of ${d.user_summary.transactions})</h4><div class="tw mini"><table><thead><tr><th>Time</th><th class="num">Amount</th><th>Type</th><th>City</th><th>Risk</th></tr></thead><tbody>
    ${d.history.map(h => `<tr class="${h.current ? 'cur' : ''}" ${h.current ? '' : `data-i="${h.i}" style="cursor:pointer"`}><td class="mono">${esc(h.timestamp.slice(5, 16))}</td><td class="num">${money2(h.amount)}</td><td>${esc(h.transaction_type)}</td><td>${esc(h.city)}</td><td><span style="color:${COL[h.risk_tier]};font-weight:800">${Math.round(h.risk_score)}</span></td></tr>`).join('')}</tbody></table></div></div>`;
  $('#dr-x').onclick = closeDrawer;
  drawer.querySelectorAll('tr[data-i]').forEach(tr => tr.onclick = () => openDrawer(+tr.dataset.i));
  drawer.scrollTop = 0;
}

/* ------------------------------------------------------------------- model */
async function loadModel() {
  if (MODEL) return MODEL;
  MODEL = await api('/api/model');
  const h = MODEL.holdout.flagged_medium_plus;
  $('#railfoot').innerHTML = `<b>Model ready</b><br>ROC-AUC ${MODEL.holdout.roc_auc.toFixed(3)}<br>Trained on ${fmt(MODEL.train_rows + MODEL.validation_rows)} rows`;
  return MODEL;
}
async function renderModel() {
  const el = $('#tab-model');
  try { await loadModel(); } catch (e) { el.innerHTML = `<div class="alert">${esc(e.message)}</div>`; return; }
  const m = MODEL, h = m.holdout, mp = h.flagged_medium_plus, ho = h.flagged_high_only, b = m.baseline_holdout;
  el.innerHTML = `
  <div class="panel" style="margin-bottom:16px"><h3>Accuracy on unseen data</h3><div class="sub">Scored on ${fmt(m.holdout_rows)} transactions the model never saw in training (${pct(m.holdout_fraud_rate)} fraud)</div>
    <div class="mgrid">
      <div class="mtile"><b>${h.roc_auc.toFixed(3)}</b><span>ROC-AUC (ranking quality)</span></div>
      <div class="mtile"><b>${h.pr_auc.toFixed(3)}</b><span>PR-AUC (fraud-focused)</span></div>
      <div class="mtile"><b style="color:#7fb0ff">${pct(mp.recall)}</b><span>fraud caught at medium+ (${mp.tp} of ${mp.tp + mp.fn})</span></div>
      <div class="mtile"><b style="color:#7fb0ff">${pct(mp.precision)}</b><span>flags that were real fraud</span></div>
    </div>
    <p class="note">High-risk only: ${pct(ho.precision)} precision, ${pct(ho.recall)} recall. Baseline logistic regression scored ROC-AUC ${b.roc_auc.toFixed(3)} and PR-AUC ${b.pr_auc.toFixed(3)}, so the boosted model is the one in use.</p></div>
  <div class="grid g2">
    <div class="panel"><h3>What the model relies on</h3><div class="sub">Permutation importance: how much accuracy drops when a signal is scrambled</div><div id="m-imp"></div></div>
    <div class="panel"><h3>Recall by fraud scenario</h3><div class="sub">Holdout fraud caught (medium or high), by simulated attack type</div><div id="m-type"></div>
      <p class="note">Stealth fraud mimics normal behaviour and is the hardest to catch, which is realistic.</p></div>
  </div>
  <div class="grid g2">
    <div class="panel"><h3>How a batch is scored</h3><div class="sub">&nbsp;</div><ol class="steps">
      <li><b>Validate.</b> Required columns are checked, missing optional columns get safe defaults.</li>
      <li><b>Build behaviour features.</b> Per user: velocity, time since last transaction, spend vs their norm, new device, unusual city.</li>
      <li><b>Score every row.</b> A gradient boosting model returns a fraud probability, shown as a 0&ndash;100 risk score.</li>
      <li><b>Tier.</b> High is ${(m.thresholds.high * 100).toFixed(0)}+, medium is ${(m.thresholds.medium * 100).toFixed(0)}&ndash;${(m.thresholds.high * 100 - 1).toFixed(0)}, low is below. Thresholds were tuned on validation data.</li>
      <li><b>Explain.</b> Flagged rows get reason codes; selecting one transaction opens a full factor breakdown.</li></ol></div>
    <div class="panel"><h3>Training setup</h3><div class="sub">Reproduce with <span class="code">python train.py</span></div>
      <div class="kv"><div><span>Model</span><b>Gradient boosting (${m.boosting_iterations} trees)</b></div><div><span>Training rows</span><b>${fmt(m.train_rows)}</b></div>
      <div><span>Validation rows</span><b>${fmt(m.validation_rows)}</b></div><div><span>Holdout rows</span><b>${fmt(m.holdout_rows)}</b></div>
      <div><span>Features</span><b>${m.features.length} signals</b></div><div><span>Trained</span><b>${esc(m.trained_at.slice(0, 10))}</b></div></div>
      <p class="note">The bundled model is trained on simulated card, transfer and ATM traffic. For production, retrain on your own labelled history by replacing the data step in train.py.</p></div>
  </div>`;
  hbars($('#m-imp'), m.importance.slice(0, 10).map(r => ({ label: r.label, flagged: r.importance, total: 1 })), { pos: true, value: r => `<b>${pct(r.flagged, 0)}</b>`, tip: r => `<b>${esc(r.label)}</b><br>${pct(r.flagged)} of total importance` });
  const bt = Object.entries(h.by_fraud_type).map(([k, v]) => ({ label: k, flagged: v.caught, total: v.count }));
  hbars($('#m-type'), bt, { value: r => `<b>${r.flagged}</b> / ${r.total} &middot; ${pct(r.flagged / r.total, 0)}` });
}

/* -------------------------------------------------------------------- init */
$('#nav').onclick = e => { const b = e.target.closest('button'); if (b && !b.disabled) setTab(b.dataset.tab); };
$('#btn-pick').onclick = e => { e.stopPropagation(); $('#file').click(); };
$('#btn-new').onclick = () => setTab('landing');
$('#file').onchange = e => { upload(e.target.files[0]); e.target.value = ''; };
$('#btn-sample').onclick = e => { e.stopPropagation(); runAnalysis(api('/api/analyze-sample', { method: 'POST' })); };
const drop = $('#drop');
drop.onclick = () => $('#file').click();
drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); } };
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => upload(e.dataTransfer.files[0]));
loadModel().catch(() => { $('#railfoot').textContent = 'Model not loaded. Run python train.py'; });
