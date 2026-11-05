// OpenArm control page: state in over server-sent events, buttons and
// clicks out over POST. Everything the robot does is decided on beast.
'use strict';

const $ = (id) => document.getElementById(id);
const ALWAYS = new Set(['stop', 'markers', 'view', 'dummy_size', 'tip_check', 'abort', 'save', 'skip', 'up', 'down', 'left',
  'right', 'in', 'out', 'confirm', 'abort',
  // the box picker's views and settings: nothing moves
  'depth_view', 'picker_view', 'snapshot_save', 'picker_setting', 'forget_skipped', 'top_faces',
  // the vacuum by hand: also mid-pick (OFF lets go of the box)
  'vacuum',
  // a setting: may be switched off while DETECT BOX looks (then it waits for MOVE)
  'bypass']);
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
  es.onmessage = (e) => { S = JSON.parse(e.data); render(S); renderScene(S); };
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
  if (name === 'manual_target') {
    const val = (id) => $(id).value.trim();
    const xyzT = { x: val('mX'), y: val('mY'), z: val('mZ') };
    try { localStorage.setItem('manualTarget', JSON.stringify({ ...xyzT, approach: manualApproach, arm: manualArm })); } catch (_) {}
    const arm = manualArm || (S && S.pick_tool === 'finger' ? 'right' : 'left');
    return act('manual_target', { ...xyzT, approach: manualApproach, arm });
  }
  if (name === 'pick_tool') {
    if (S && S.pick_tool === b.dataset.value) return;
    return act('pick_tool', { value: b.dataset.value });
  }
  if (name === 'tool') {
    return act('tool', { arm: b.dataset.arm, step: b.dataset.step ? Number(b.dataset.step) : 0 });
  }
  if (name === 'order') return act('order', { value: b.dataset.value });
  if (name === 'top_faces') return act('top_faces', { value: b.dataset.value });
  if (name === 'vacuum') {
    if (S && S.vacuum === (b.dataset.value === 'on')) return;
    return act('vacuum', { value: b.dataset.value });
  }
  if (name === 'picker_setting') {
    return act('picker_setting', { key: b.dataset.key, step: Number(b.dataset.step) });
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
  if (S && !S.click_only && !S.anchor_mode) return;   // the model picks the point
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
  selectTab(['pick', 'calibrate', 'advanced', 'system'].includes(asked) ? asked : saved || 'pick');
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
  const map = { m: 'move', ' ': S && S.click_only ? null : 'detect', r: 'recapture',
    t: S && S.click_only ? 'teach' : null, b: 'return',
    n: 'navigation', p: 'pre_pick', f: 'cal_arms', c: 'calibrate', o: 'overlay',
    k: 'markers', a: 'anchor', e: 'refit', i: 'tip_check' };
  const key = e.key.toLowerCase();
  if (key === 'd') {                    // FRONT / TOP with --click; else the depth picture
    e.preventDefault();
    return act(S && S.click_only ? 'topdown' : 'depth_view');
  }
  // the box picker window's keys (not with --click: T is TEACH there)
  const pickerKeys = { v: 'picker_view', t: 'next_target', s: 'snapshot_save', h: 'check_now' };
  if (S && !S.click_only && pickerKeys[key]) { e.preventDefault(); return act(pickerKeys[key]); }
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
  const pk = s.picker || {};
  const viewing = pk.picker_on && pk.has_view && s.view === 'captured' ? " · PICKER'S VIEW"
    : pk.depth_on ? ' · DEPTH' : '';
  setText($('viewChip'), (s.calib_view ? 'CALIBRATION' : s.view === 'live' ? 'LIVE' : 'CAPTURED') + viewing);
  setText($('fpsChip'), pk.fps == null ? '– fps' : `${Math.round(pk.fps)} fps`);
  $('fpsChip').classList.toggle('warn', pk.fps != null && pk.fps < 5);
  setText($('approachChip'), s.top_down ? 'TOP DOWN' : 'FRONT');
  $('haltChip').hidden = !s.halted;
  $('clickChip').hidden = !s.click_only;
  document.querySelectorAll('.btn.model').forEach((b) => { b.hidden = s.click_only; });
  document.querySelectorAll('.click-only').forEach((g) => { g.hidden = !s.click_only; });
  $('grpAuto').hidden = s.click_only;                 // (bypass and pick all: the box model's)
  $('stage').classList.toggle('pickable', !!(s.click_only || s.anchor_mode));
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
  if (manualArm === null) {             // the typed target's arm follows the gripper
    $('mLeft').setAttribute('aria-pressed', s.pick_tool !== 'finger');
    $('mRight').setAttribute('aria-pressed', s.pick_tool === 'finger');
  }
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
    const ba = r.base_adjust;
    const weight = r.held === true ? ` · on the cup (${Math.round(r.held_g)} g)` : r.held === false ? ' · nothing on the cup' : '';
    setText($('resultWhy'), `${r.arm} arm · ${r.contact ? 'contact felt' : 'no contact'}${weight} · ${r.message || ''} · ${r.time}`
      + (ba ? ` · BASE ADJUST sent (${ba.reason}): ` + (ba.direction === 'none'
        ? 'none - already lined up sideways, left or right cannot help'
        : `move ${ba.direction} ${Number(ba.distance_m).toFixed(3)} m - detect again after the move`) : ''));
  }
  renderPicker(s);
  const badge = $('targetBadge');
  if (!p) {
    setText(badge, 'none'); badge.dataset.tone = 'idle';
    setText($('targetXyz'), s.click_only ? 'Click the picture to pick a point'
      : 'Detect a box: the model picks it, FRONT or TOP');
    setText($('targetWhy'), '');
  } else {
    const tone = p.ok === null ? 'busy' : !p.ok ? 'bad' : p.edge ? 'edge' : 'ok';
    setText(badge, { busy: 'checking', bad: 'not reachable', edge: 'edge of reach', ok: 'reachable' }[tone]);
    badge.dataset.tone = tone;
    setText($('targetXyz'), `${p.arm} arm · ${xyz(p.xyz)}`);
    setText($('targetWhy'), p.ok === false ? (p.why || '') : (p.label ? `plan: ${p.label}` : ''));
  }
  const move = $('btnMove');
  // after a pick that did not take its box: MOVE = TRY AGAIN (looks again, picks that box)
  const retry = !!s.retry && !p;
  setText($('moveLabel'), retry ? 'Try again' : 'Move');
  move.title = retry ? `The last pick did not take the box (${s.retry}): look again and pick it` : '';
  move.dataset.tone = retry ? 'retry' : !p ? '' : p.ok === false ? 'bad' : p.ok ? 'ok' : '';
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

  // pick all, bypass
  setPressed($('tgBypass'), s.bypass);
  setText($('pickAllLabel'), s.pick_all
    ? `Picking all · ${s.pick_all.picked} picked${s.pick_all.skipped ? `, ${s.pick_all.skipped} skipped` : ''}` +
      `${s.pick_all.dropped ? `, ${s.pick_all.dropped} dropped` : ''}`
    : 'Pick all');

  // advanced: the vacuum
  setPressed($('vacOn'), s.vacuum);
  setPressed($('vacOff'), !s.vacuum);
  const bridge = s.vacuum_listeners > 0;
  $('vacStatus').classList.toggle('warn', !bridge);
  setText($('vacStatus'), bridge
    ? `${s.vacuum ? 'On' : 'Off'} · the ESP32 bridge is listening. A pick switches it by itself: on at the press, off at the drop.`
    : 'No bridge is listening on /beast_arm/vacuum (terminal 1 starts it): these buttons switch nothing.');

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
  if (!s.busy && !s.teaching && !s.calibrating) move.disabled = !retry && (!p || p.ok === false);
  $('btnMove2').disabled = move.disabled;             // (the Advanced tab's MOVE: the same)
  $('btnMove2').dataset.tone = move.dataset.tone || '';
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

// -- the box picker's own window, as cards -------------------------------------------
// (box_picker/src/pick_loop.py: tally, the target close up with its camera
// coordinates and tries, timings, the strip of the last picks, the check's
// numbers, FLAG; on the Advanced tab its views, order and settings)

const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
const cm = (v) => (v == null ? '–' : (v * 100).toFixed(1));
let closeUpKey = null;
let historyKey = '';

function renderPicker(s) {
  const pk = s.picker || {};
  document.querySelectorAll('.model-group').forEach((g) => { g.hidden = s.click_only; });
  const tally = pk.tally || {};
  const stats = [['picked', tally.picked ?? 0, 'ok'], ['failed tries', tally.failed ?? 0, 'bad'],
    ['skipped', tally.skipped ?? 0, 'edge']].concat(tally.passed ? [['passed over', tally.passed, '']] : [])
    .map(([what, n, tone]) => `<span data-tone="${n ? tone : ''}"><b>${n}</b>${what}</span>`).join('');
  if ($('pickerTally').innerHTML !== stats) $('pickerTally').innerHTML = stats;
  // the tilt: gravity's (the accelerometer, or its saved reading) or the TF's
  const tilt = pk.tilt || {};
  const tb = $('tiltBadge');
  const deg = tilt.deg == null ? '' : `${Math.abs(tilt.deg).toFixed(1)}° ${tilt.deg >= 0 ? 'down' : 'up'}`;
  setText(tb, !tilt.source ? 'tilt –' : tilt.source === 'tf' ? `tilt ${deg} · TF` : `tilt ${deg} · accel`);
  tb.dataset.tone = !tilt.source ? 'idle' : tilt.source === 'tf' ? 'edge' : 'ok';
  tb.title = !tilt.source ? 'The tilt the picker uses: not known yet'
    : tilt.source === 'tf' ? "The picker's tilt comes from the TF, not gravity: no accelerometer reading yet (install the RealSense udev rules, then the camera launch's accel:=auto turns it on)"
    : `The picker's tilt from the camera's accelerometer (${tilt.source})`;

  // the target, close up
  const t = pk.target;
  const showCrop = !!(t && pk.close_up && s.pick);
  $('closeUpRow').hidden = !showCrop;
  if (showCrop && closeUpKey !== pk.close_up) {
    closeUpKey = pk.close_up;
    $('closeUp').src = `/picker/close_up.jpg?${pk.close_up}`;
  }
  if (t && t.cam) {
    [['camX', 0], ['camY', 1], ['camZ', 2]].forEach(([id, i]) => {
      const h = `${cm(t.cam[i])}<i>cm</i>`;
      if ($(id).innerHTML !== h) $(id).innerHTML = h;
    });
  }
  const dots = Array.from({ length: pk.tries || 1 }, (_, i) =>
    `<span class="dot${i < (pk.attempt || 1) ? (pk.attempt > 1 ? ' on retry' : ' on') : ''}"></span>`).join('')
    + `<span class="muted small">try ${pk.attempt || 1} of ${pk.tries || 1}</span>`;
  if ($('tryDots').innerHTML !== dots) $('tryDots').innerHTML = dots;
  setText($('pickerAgree'), !t || !s.pick ? '' : `${t.agreeing ?? '?'} of ${t.frames ?? '?'} frames agree · ${t.boxes ?? '?'} boxes`
    + (t.size_mm ? ` · face ${t.size_mm[0]} × ${t.size_mm[1]} mm` : ''));
  const took = t && t.took;
  setText($('pickerTook'), !took || !s.pick ? '' : `found in ${fmt(took.pictures + took.picker, 1)} s · pictures ${fmt(took.pictures, 1)} s · picker ${fmt(took.picker, 1)} s`);

  // the last picks: close-ups with their result's colour
  const hist = pk.history || [];
  const hk = hist.map((h) => `${h.n}${h.tone}`).join(',');
  if (hk !== historyKey) {
    historyKey = hk;
    $('history').innerHTML = hist.slice().reverse().map((h) => (h.img
      ? `<img src="/picker/history/${h.n}.jpg" data-tone="${h.tone}" alt="${h.outcome}" title="${h.time} · ${h.outcome}">`
      : `<span class="nothumb" data-tone="${h.tone}" title="${h.time} · ${h.outcome}"></span>`)).join('');
  }
  const last = pk.last;
  setText($('pickerLast'), last && last.time && Date.now() / 1000 - last.time < 30 ? `box picker: ${last.message}` : '');

  // the result card: the check's numbers, and a FLAG
  const c = pk.check;
  const r = s.pick_result;
  setText($('resultNumbers'), !c || !r || !r.verdict ? '' : `gone ${pct(c.gone)} · same ${pct(c.same)}`
    + (c.other_box ? ` · another box ${pct(c.other_box)}` : '') + ` · ${c.by || '?'} · ${c.frames ?? '?'} frames`
    + (c.disturbed ? ` · ${c.disturbed} other box(es) disturbed` : '')
    + (c.boxes_around ? ` · boxes around it ${c.boxes_around[0]} → ${c.boxes_around[1]}` : ''));
  const flag = last && last.text === 'flag' && last.ok && last.photos && Date.now() / 1000 - last.time < 600;
  $('resultFlag').hidden = !flag;
  if (flag) {
    const dir = Object.values(last.photos)[0] || '';
    setText($('resultFlag'), `FLAG RAISED: not picked after ${r && r.base_adjust ? r.base_adjust.failures : 'repeated'} tries. Photos for a person in ${dir.replace(/\/[^/]*$/, '/')}`);
  }

  // the Advanced tab
  const st = pk.settings || {};
  setPressed($('tgDepth'), pk.depth_on);
  setPressed($('tgPickerView'), pk.picker_on);
  setPressed($('ordV1'), st.order === 'v1');
  setPressed($('ordV2'), st.order === 'v2');
  setPressed($('facesFront'), st.top_faces === false);
  setPressed($('facesLarger'), st.top_faces === true);
  setText($('setFrames'), st.frames == null ? '–' : String(st.frames));
  setText($('setScore'), st.score == null ? '–' : Number(st.score).toFixed(2));
  setText($('setCheck'), st.check_frames == null ? '–' : String(st.check_frames));
  setText($('btnForget'), `Forget skipped${tally.skipped || tally.passed ? ` (${(tally.skipped || 0) + (tally.passed || 0)})` : ''}`);
}

// the dummy size: sent when a field is changed (Enter or leaving it)
['dimH', 'dimW', 'dimD'].forEach((id) => $(id).addEventListener('change', () => {
  act('dummy_dims', { h: $('dimH').value, w: $('dimW').value, d: $('dimD').value })
    .then((j) => { if (j && j.ok) toast(j.message); });
}));

connect();

// -- the box picker's scene, drawn over the picture -----------------------------
// The picture is clean; what the picker found (every box, the face it chose,
// the pick point) and the pick check's verdict are vector graphics on top, in
// the picture's own pixels (viewBox), so they line up at any size.

const SVG_NS = 'http://www.w3.org/2000/svg';
const VERDICT = {
  success: ['ok', '✓', 'Picked', 'The box is gone from its place'],
  missed: ['bad', '✕', 'Not picked', 'The box is still there'],
  moved: ['bad', '!', 'Box moved', 'Pushed, tipped or pulled part way'],
  dropped: ['bad', '✕', 'Dropped', 'It left its place but was not held'],
  wrong_box: ['bad', '!', 'Wrong box', 'A neighbour was taken instead'],
  blocked: ['edge', '?', "Can't tell", 'Something was in view'],
  camera_moved: ['edge', '?', "Can't tell", 'The camera moved'],
  unknown: ['edge', '?', "Can't tell", 'Too little depth on the box'],
};
let sceneKey = '';

function el(name, attrs, parent) {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const pathOf = (pts) => pts && pts.length > 2 ? 'M' + pts.map((p) => p.join(' ')).join('L') + 'Z' : '';

function renderScene(s) {
  const svg = $('scene');
  const sc = s.scene;
  const scale = svg.getBoundingClientRect().width / ((sc && sc.size && sc.size[0]) || 640);
  const p = s.pick;
  const key = JSON.stringify([sc && sc.time, p && [p.uv, p.ok, p.edge], s.target && s.target.box_mm,
    Math.round(scale * 100)]);
  if (key === sceneKey) return;
  sceneKey = key;
  svg.replaceChildren();
  const verdict = $('verdict');
  if (!sc) {
    svg.classList.remove('on');
    verdict.classList.remove('on');
    return;
  }
  const [w, h] = sc.size || [640, 480];
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');   // as the picture: object-fit contain
  const px = (n) => n / Math.max(scale, 0.01);                 // screen pixels -> picture pixels

  if (sc.kind === 'check') {
    const v = VERDICT[sc.outcome] || ['edge', '?', sc.title || 'Checked', sc.why || ''];
    const cls = sc.picked === true ? 'gone' : sc.picked === false ? 'still' : 'unsure';
    if (sc.outline) {
      el('path', { d: pathOf(sc.outline), class: 'halo' }, svg);
      el('path', { d: pathOf(sc.outline), class: cls }, svg);
    }
    for (const d of sc.disturbed || []) el('circle', { cx: d.u, cy: d.v, r: px(9), class: 'knocked' }, svg);
    verdict.dataset.tone = v[0];
    setText($('verdictIcon'), v[1]);
    setText($('verdictTitle'), v[2]);
    setText($('verdictWhy'), (sc.disturbed || []).length ? `${v[3]} · a neighbour changed too` : v[3]);
    verdict.classList.add('on');
    svg.classList.add('on');
    return;
  }

  verdict.classList.remove('on');
  const tone = !p ? '' : p.ok === false ? 'bad' : p.ok ? 'ok' : '';
  let target = null;
  // every box exactly as the model outlined it (its mask), in yellow
  for (const b of sc.boxes || []) {
    if (b.target) target = b;
    if (!b.poly) continue;
    el('path', { d: pathOf(b.poly), class: 'halo' }, svg);              // shows on white boxes too
    el('path', { d: pathOf(b.poly), class: 'box' + (b.target ? ' chosen' : '') }, svg);
  }
  // the chosen box: the face to grip
  if (target && target.face) {
    el('path', { d: pathOf(target.face), class: 'face', 'data-tone': tone }, svg);
  }
  const uv = p && p.uv ? p.uv : sc.target ? [sc.target.u, sc.target.v] : null;
  if (!uv) { svg.classList.add('on'); return; }
  const [u, v] = uv;
  el('circle', { cx: u, cy: v, r: px(9), class: 'pulse' }, svg);
  el('circle', { cx: u, cy: v, r: px(9), class: 'ring' }, svg);
  el('circle', { cx: u, cy: v, r: px(2.5), class: 'dot' }, svg);

  // the tag: what and how, next to the point (flipped to stay inside the picture)
  const t = sc.target || {};
  const box = s.target && s.target.box_mm && s.target.box_mm[0] ? s.target.box_mm : null;
  const face = t.size_mm ? `${t.size_mm[0]} × ${t.size_mm[1]} mm` : '';
  const line1 = `${(t.approach || 'front').toUpperCase()}${face ? '  ·  ' + face : ''}`;
  const line2 = !p ? `${t.agreeing ?? '?'}/${t.frames ?? '?'} frames agree`
    : p.ok === null ? 'Checking the reach…' : p.ok === false ? 'Not reachable'
    : (box ? `${box[0]} × ${box[1]} × ${box[2]} box · ` : '') + 'Press Move';
  const g = el('g', { class: 'tag' }, svg);
  const rect = el('rect', { rx: px(9) }, g);
  const a = el('text', { class: 't1', 'font-size': px(12.5) }, g);
  a.textContent = line1;
  const b2 = el('text', { class: 't2', 'font-size': px(11.5), 'data-tone': p ? (p.ok === null ? 'busy' : tone) : '' }, g);
  b2.textContent = line2;
  const pad = px(10);
  const tw = Math.max(a.getComputedTextLength(), b2.getComputedTextLength()) + 2 * pad;
  const th = px(44);
  let x = u + px(18), y = v - th / 2;
  if (x + tw > w - px(4)) x = u - px(18) - tw;
  y = Math.min(Math.max(y, px(4)), h - th - px(4));
  rect.setAttribute('x', x); rect.setAttribute('y', y);
  rect.setAttribute('width', tw); rect.setAttribute('height', th);
  a.setAttribute('x', x + pad); a.setAttribute('y', y + px(18));
  b2.setAttribute('x', x + pad); b2.setAttribute('y', y + px(35));
  svg.classList.add('on');
}
window.addEventListener('resize', () => { if (S) renderScene(S); });

// -- ADVANCED: a target typed in camera coordinates ------------------------------
let manualApproach = 'front';
let manualArm = null;                    // null: the arm of the gripper chosen above
function setManualArm(a) {
  manualArm = a === 'right' ? 'right' : 'left';
  $('mLeft').setAttribute('aria-pressed', manualArm === 'left');
  $('mRight').setAttribute('aria-pressed', manualArm === 'right');
}
document.querySelectorAll('[data-marm]').forEach((b) => {
  b.addEventListener('click', () => setManualArm(b.dataset.marm));
});
function setManualApproach(a) {
  manualApproach = a === 'top' ? 'top' : 'front';
  $('mFront').setAttribute('aria-pressed', manualApproach === 'front');
  $('mTop').setAttribute('aria-pressed', manualApproach === 'top');
}
document.querySelectorAll('[data-approach]').forEach((b) => {
  b.addEventListener('click', () => setManualApproach(b.dataset.approach));
});
try {                                    // the last typed target, in this browser
  const last = JSON.parse(localStorage.getItem('manualTarget') || 'null');
  if (last) {
    $('mX').value = last.x ?? ''; $('mY').value = last.y ?? ''; $('mZ').value = last.z ?? '';
    setManualApproach(last.approach);
    if (last.arm) setManualArm(last.arm);
  }
} catch (_) {}
