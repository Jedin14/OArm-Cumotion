// OpenArm control page: state in over server-sent events, buttons and
// clicks out over POST. Everything the robot does is decided on beast.
'use strict';

const $ = (id) => document.getElementById(id);
const ALWAYS = new Set(['stop', 'markers', 'view', 'dummy_size', 'tip_check', 'abort', 'save', 'skip', 'up', 'down', 'left',
  'right', 'in', 'out', 'confirm', 'abort']);
let S = null;                 // the last state from beast
const NUDGES = ['left', 'right', 'up', 'down', 'in', 'out'];
let lastLog = 0;

// -- talking to beast -----------------------------------------------------------

async function post(path, body) {
  try {
    const r = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.ok) toast(j.message, true);
    return j;
  } catch (e) {
    toast('beast did not answer', true);
    return { ok: false };
  }
}
const act = (name, extra = {}) => post('/api/action', { name, ...extra });

function connect() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => { S = JSON.parse(e.data); render(S); };
  es.onerror = () => {
    $('status').textContent = 'Lost the connection to beast; retrying…';
    $('status').classList.add('error');
  };
}

// the picture: reconnect the stream if it drops
const video = $('video');
// ?still: single frames instead of the stream (screenshots, slow links)
if (new URLSearchParams(location.search).has('still')) {
  video.src = '/snapshot.jpg';
  setInterval(() => { video.src = '/snapshot.jpg?' + Date.now(); }, 1000);
} else {
  video.src = '/stream.mjpg';
}
video.addEventListener('error', () => setTimeout(() => {
  video.src = '/stream.mjpg?' + Date.now();
}, 1000));

// -- toasts ----------------------------------------------------------------------

function toast(text, bad = false) {
  if (!text) return;
  const t = document.createElement('div');
  t.className = 'toast' + (bad ? ' bad' : '');
  t.textContent = text;
  $('toasts').appendChild(t);
  setTimeout(() => {
    t.classList.add('leaving');
    setTimeout(() => t.remove(), 220);
  }, bad ? 3800 : 2200);
}

// -- buttons ---------------------------------------------------------------------

document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-action]');
  if (!b || b.disabled) return;
  const name = b.dataset.action;
  if (e.shiftKey && NUDGES.includes(name)) return act(name + '5');   // Shift-click: 5 mm
  if (name === 'view') return act('view', { value: b.dataset.value });
  if (name === 'pick_tool') {
    if (S && S.pick_tool === b.dataset.value) return;
    return act('pick_tool', { value: b.dataset.value });
  }
  if (name === 'tool') {
    return act('tool', { arm: b.dataset.arm, step: b.dataset.step ? Number(b.dataset.step) : 0 });
  }
  if (name === 'topdown') {
    // a segmented choice: only send when it changes something
    const want = b.dataset.when === 'top';
    if (S && S.top_down === want) return;
  }
  act(name);
});

// -- the picture: click a point ---------------------------------------------------

function imageRect() {
  // where the picture really is inside the <img> (object-fit: contain)
  const r = video.getBoundingClientRect();
  const nw = video.naturalWidth || 640, nh = video.naturalHeight || 480;
  const s = Math.min(r.width / nw, r.height / nh);
  const w = nw * s, h = nh * s;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
}

$('stage').addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || e.target.closest('.chrome, .sheet')) return;
  const r = imageRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  ripple(e.clientX, e.clientY);              // feedback on press, not on release
  post('/api/click', { x, y });
});

function ripple(cx, cy) {
  const box = $('ripples').getBoundingClientRect();
  const d = document.createElement('span');
  d.className = 'ripple';
  d.style.left = (cx - box.left) + 'px';
  d.style.top = (cy - box.top) + 'px';
  $('ripples').appendChild(d);
  requestAnimationFrame(() => requestAnimationFrame(() => d.classList.add('go')));
  setTimeout(() => d.remove(), 500);
}

// -- tabs ------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('.tab')];
function selectTab(name) {
  tabs.forEach((t, i) => {
    const on = t.dataset.tab === name;
    t.setAttribute('aria-selected', on);
    if (on) $('tabIndicator').style.transform = `translateX(${i * 100}%)`;
  });
  document.querySelectorAll('.pane').forEach((p) => { p.hidden = p.dataset.pane !== name; });
  try { localStorage.setItem('tab', name); } catch (_) { /* private window */ }
}
tabs.forEach((t) => t.addEventListener('click', () => selectTab(t.dataset.tab)));
{
  const asked = new URLSearchParams(location.search).get('tab');
  let saved = null;
  try { saved = localStorage.getItem('tab'); } catch (_) { /* private window */ }
  selectTab(['pick', 'calibrate', 'system'].includes(asked) ? asked : saved || 'pick');
}

$('logToggle').addEventListener('click', () => {
  const b = $('logToggle');
  b.setAttribute('aria-expanded', b.getAttribute('aria-expanded') !== 'true');
});

// -- keys (no animation on any of them) ---------------------------------------------

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest('input, textarea')) return;
  // a held key repeats: only the nudges may (holding I took 26 tip checks)
  if (e.repeat && !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-'].includes(e.key)) return;
  if (e.key === 'Escape') { e.preventDefault(); return act('stop'); }
  if (S && S.teaching) {
    const k = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
      '+': 'in', '=': 'in', '-': 'out', Enter: 'save', Backspace: 'skip' }[e.key];
    // Shift: 5 mm instead of 1 (not for save / skip)
    if (k) { e.preventDefault(); act(e.shiftKey && NUDGES.includes(k) ? k + '5' : k); }
    return;
  }
  if (S && S.confirming && (e.key === 'Enter' || e.key === 'y')) return act('confirm');
  const map = { m: 'move', ' ': S && S.click_only ? null : 'detect', r: 'recapture', t: 'teach', b: 'return',
    n: 'navigation', p: 'pre_pick', f: 'cal_arms', c: 'calibrate', o: 'overlay',
    k: 'markers', a: 'anchor', e: 'refit', i: 'tip_check' };
  const key = e.key.toLowerCase();
  if (key === 'd') { e.preventDefault(); return act('topdown'); }
  if (key === 'l') {
    return act('view', { value: S && S.view === 'live' ? 'captured' : 'live' });
  }
  if (map[key]) { e.preventDefault(); act(map[key]); }
});

// -- drawing the state ---------------------------------------------------------------

const TONE = {
  pre_pick: 'ok', navigation: 'ok', arm_out: 'warn', unknown: 'warn',
};
const fmt = (v, d = 3) => (v === null || v === undefined ? '–' : Number(v).toFixed(d));
const xyz = (p) => `x ${fmt(p[0])}  y ${fmt(p[1])}  z ${fmt(p[2])} m`;
const setPressed = (el, on) => el && el.setAttribute('aria-pressed', on ? 'true' : 'false');
const setText = (el, t) => { if (el && el.textContent !== t) el.textContent = t; };

function render(s) {
  // chrome over the camera
  const st = $('armState');
  setText(st, s.arm_state.replaceAll('_', ' '));
  st.dataset.tone = TONE[s.arm_state] || 'busy';
  setText($('viewChip'), s.calib_view ? 'CALIBRATION' : s.view === 'live' ? 'LIVE' : 'CAPTURED');
  setText($('approachChip'), s.top_down ? 'TOP DOWN' : 'FRONT');
  $('haltChip').hidden = !s.halted;
  $('clickChip').hidden = !s.click_only;
  document.querySelectorAll('.btn.model').forEach((b) => { b.hidden = s.click_only; });
  document.querySelector('[data-action="recapture"]').style.gridColumn = s.click_only ? '1 / -1' : '';
  $('viewLive').setAttribute('aria-selected', s.view === 'live');
  $('viewCaptured').setAttribute('aria-selected', s.view === 'captured');
  $('viewCaptured').disabled = !s.captured;
  setText($('status'), s.status || '');
  $('status').classList.toggle('error', !!s.error);
  setText($('detail'), s.detail || '');

  // sheets
  $('teachSheet').classList.toggle('open', s.teaching);
  $('teachSheet').setAttribute('aria-hidden', !s.teaching);
  setText($('teachTitle'), s.anchor_mode ? 'Anchor: put the tip exactly on the point'
    : 'Teach: nudge the tip onto the spot');
  const cal = s.calibrating || s.confirming;
  $('stage').classList.toggle('sheet-up', s.teaching || cal);
  $('calSheet').classList.toggle('open', cal && !s.teaching);
  $('calSheet').setAttribute('aria-hidden', !cal);
  $('calConfirm').hidden = !s.confirming;
  setText($('calTitle'), s.confirming ? 'Is the outline on marker 11?' : 'Calibrating');
  setText($('calSub'), s.confirming ? 'Confirm and the arms start; nothing has moved yet'
    : (s.status || 'Keep the view clear') + ' · Abort stops after the current move');
  setText($('teachSub'), `${s.status ? s.status + ' · ' : ''}1 mm a press, Shift 5 mm · arrows, + / −, Enter saves`);

  // pick
  setPressed($('modeNav'), s.mode === 'navigation');
  setPressed($('modePick'), s.mode === 'pick');
  setPressed($('apFront'), !s.top_down);
  setPressed($('apTop'), s.top_down);
  setPressed($('tlVacuum'), s.pick_tool === 'vacuum');
  setPressed($('tlFinger'), s.pick_tool === 'finger');
  // the box: the face the picker measured, matched to the known size
  const p = s.pick;
  const tg = s.target;
  const face = tg && tg.face_mm && tg.face_mm[0] ? tg.face_mm : null;
  const box = tg && tg.box_mm && tg.box_mm[0] ? tg.box_mm : null;
  setText($('targetBox'), !face || !p ? '' : `${tg.tool} · ${tg.arm} arm · ${tg.approach} face ${face[0]} × ${face[1]} mm`
    + (box ? ` = the ${box[0]} × ${box[1]} face, ${box[2]} mm deep` : s.object_size ? ' · no face of the box size matches' : ''));
  // the last pick: the camera's check
  const r = s.pick_result;
  $('resultCard').hidden = !r;
  if (r) {
    const rb = $('resultBadge');
    const rt = r.picked === true ? 'ok' : r.picked === false ? 'bad' : 'edge';
    setText(rb, r.picked === true ? 'picked' : r.picked === false ? `not picked · ${r.outcome}` : `can't tell · ${r.outcome || '?'}`);
    rb.dataset.tone = rt;
    setText($('resultWhy'), `${r.arm} arm · ${r.contact ? 'contact felt' : 'no contact'} · ${r.message || ''} · ${r.time}`);
  }
  const badge = $('targetBadge');
  if (!p) {
    setText(badge, 'none'); badge.dataset.tone = 'idle';
    setText($('targetXyz'), s.click_only ? 'Click the picture to pick a point'
      : 'Click the picture, or detect a box');
    setText($('targetWhy'), '');
  } else {
    const tone = p.ok === null ? 'busy' : !p.ok ? 'bad' : p.edge ? 'edge' : 'ok';
    setText(badge, { busy: 'checking', bad: 'not reachable', edge: 'edge of reach', ok: 'reachable' }[tone]);
    badge.dataset.tone = tone;
    setText($('targetXyz'), `${p.arm} arm · ${xyz(p.xyz)}`);
    setText($('targetWhy'), p.ok === false ? (p.why || '') : (p.label ? `plan: ${p.label}` : ''));
  }
  const move = $('btnMove');
  move.dataset.tone = !p ? '' : p.ok === false ? 'bad' : p.ok ? 'ok' : '';
  $('btnReturn').hidden = !s.last_touch;
  setPressed($('tgTeach'), s.teach && !s.anchor_mode);
  setPressed($('tgPieces'), s.pieces_on);
  for (const arm of ['left', 'right']) {
    const t = s.tools[arm];
    setPressed(document.querySelector(`.toggle[data-action="tool"][data-arm="${arm}"]`), t.on);
    setText($(arm === 'left' ? 'lenLeft' : 'lenRight'), t.on ? `${Math.round(t.length * 1000)} mm` : 'off');
  }

  // held object
  const heldArms = Object.keys(s.held || {});
  const hb = $('heldBadge');
  setText(hb, heldArms.length ? `${heldArms.join(' + ')} holding` : 'empty');
  hb.dataset.tone = heldArms.length ? 'busy' : 'idle';
  const sz = heldArms.length ? s.held[heldArms[0]] : s.object_size;
  setText($('heldSize'), sz ? `${heldArms.length ? 'Holding' : 'Next pick'}: ${sz[0]} × ${sz[1]} × ${sz[2]} mm`
    + (s.object_source && !heldArms.length ? ` · ${s.object_source}` : '')
    : `No size on ${s.ns}/held_object/size`);
  setPressed($('tgDummy'), s.dummy);
  // the size fields follow the server, except the one being typed in
  if (s.dummy_mm) {
    [['dimH', 0], ['dimW', 1], ['dimD', 2]].forEach(([id, i]) => {
      const el = $(id);
      if (document.activeElement !== el && Number(el.value) !== s.dummy_mm[i]) el.value = s.dummy_mm[i];
    });
  }

  // calibrate
  setPressed($('tgMarkers'), s.show_markers);
  setPressed($('tgAnchor'), s.anchor_mode);
  setPressed($('btnOverlay'), s.overlay);
  const ids = s.view === 'live' && s.show_markers ? s.markers : [];
  const idsHtml = ids.length
    ? ids.map((i) => `<span class="id${i === 11 ? ' wall' : ''}">${i}</span>`).join('')
    : `<span class="muted small">${s.show_markers ? (s.view === 'live' ? 'none in view'
      : 'switch to Live to see them') : 'off'}</span>`;
  if ($('markerIds').innerHTML !== idsHtml) $('markerIds').innerHTML = idsHtml;
  for (const [arm, pre] of [['left', 'mLeft'], ['right', 'mRight']]) {
    const m = s.models[arm];
    setText($(pre + 'Model'), m.has ? (m.model || '?') : 'none');
    setText($(pre + 'Fit'), m.rms_px ? `${fmt(m.rms_px, 2)} px` : '–');
    setText($(pre + 'Anch'), m.anchors + (m.anchor_mm ? ` · ${fmt(m.anchor_mm, 1)} mm` : ''));
  }
  setText($('modelWhen'), s.model_when ? String(s.model_when).replace('T', ' ') : '');
  setText($('btnDepthCap'), `Depth: capture${s.depth_n ? ` (${s.depth_n})` : ''}`);
  setText($('autoCalStatus'), [s.depth_cal ? `Depth agrees with colour to ${s.depth_cal}.` : 'No depth check yet.',
    s.box_cal ? `Cal tip: ${Math.round(s.box_cal.box_holdout_mm * 10) / 10} mm mean on unseen postures (90 % under ${Math.round(s.box_cal.box_holdout_p90_mm * 10) / 10} mm), ${s.box_cal.box_tips} tips.` : 'No Cal tip yet.'].join(' '));
  $('btnApply').disabled = !s.calibration_ready || s.busy;

  // system
  setText($('sysState'), s.arm_state);
  setText($('sysPhase'), s.phase || 'idle');
  setText($('sysHeld'), heldArms.length ? heldArms.map((a) => `${a} ${s.held[a].join('x')} mm`).join(', ') : 'none');
  setText($('sysResult'), s.result || '–');
  setText($('sysTarget'), s.target ? `${s.target.tool} · ${s.target.arm} · ${s.target.approach} · ${xyz(s.target.xyz)} · ${s.target.confidence}` : '–');
  setText($('sysCamera'), s.camera ? 'streaming' : 'no frames');

  // busy: only what is safe stays live
  document.querySelectorAll('[data-action]').forEach((b) => {
    const name = b.dataset.action;
    if (name === 'view' || name === 'apply') return;
    const off = (s.busy || s.teaching || s.calibrating) && !ALWAYS.has(name)
      && !(name === 'topdown' && !s.busy);
    if (b.disabled !== off) b.disabled = off;
  });
  if (!s.busy && !s.teaching && !s.calibrating) move.disabled = !p || p.ok === false;
  $('btnRelease').disabled = !heldArms.length || s.busy;

  // activity
  const list = $('log');
  for (const entry of s.log) {
    if (entry.t <= lastLog) continue;
    lastLog = entry.t;
    const li = document.createElement('li');
    if (entry.error) li.className = 'error';
    const time = new Date(entry.t * 1000).toTimeString().slice(0, 8);
    li.innerHTML = `<time>${time}</time><span></span>`;
    li.lastChild.textContent = entry.text;
    list.prepend(li);
  }
  while (list.children.length > 120) list.lastChild.remove();
}

// the dummy size: sent when a field is changed (Enter or leaving it)
['dimH', 'dimW', 'dimD'].forEach((id) => $(id).addEventListener('change', () => {
  act('dummy_dims', { h: $('dimH').value, w: $('dimW').value, d: $('dimD').value })
    .then((j) => { if (j && j.ok) toast(j.message); });
}));

connect();
