// OpenArm control page: state in over server-sent events, buttons and
// clicks out over POST. Everything the robot does is decided on beast.
'use strict';

const $ = (id) => document.getElementById(id);
const ALWAYS = new Set(['stop', 'markers', 'view', 'dummy_size', 'tip_check', 'abort', 'save', 'skip', 'up', 'down', 'left',
  'right', 'in', 'out', 'confirm', 'abort',
  // the box picker's views and settings: nothing moves
  'depth_view', 'picker_view', 'snapshot_save', 'picker_setting', 'forget_skipped', 'top_faces',
  // Pick all's base movement setting and its alert: while it runs
  'no_base_move', 'base_try', 'base_skip',
  // the vacuum by hand: also mid-pick (OFF lets go of the box)
  'vacuum',
  // a setting: may be switched off while DETECT BOX looks (then it waits for MOVE)
  'bypass',
  // AUTO CALIBRATE waiting for the box: pressed while it runs (renderAutocal enables it)
  'autocal_box_on']);
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
  if (name === 'auto_cal') return act('auto_cal', { weigh: $('acWeigh').checked });
  if (name === 'order') return act('order', { value: b.dataset.value });
  if (name === 'top_faces') return act('top_faces', { value: b.dataset.value });
  if (name === 'pick_all') {
    try {
      if (window.Notification && Notification.permission === 'default') Notification.requestPermission();
    } catch (_) { /* in-page only */ }
    // COUNT (2026-11-17): Pick N sends the stepper's number; All sends "all"
    if (b.dataset.count === 'all') return act('pick_all', { count: 'all' });
    const n = pickCount();
    setPickCount(n);                    // (the field as sent, remembered)
    return act('pick_all', { count: n });
  }
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
  const names = tabs.map((t) => t.dataset.tab);
  selectTab(names.includes(asked) ? asked : names.includes(saved) ? saved : 'pick');
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
  // the L2 LiDAR: working -> the walls are off; not -> the walls are used
  const ld = s.lidar;
  const lc = $('lidarChip');
  // (looking up: it stands upright and sees nothing below itself -- tilt it down)
  // (mount?: its points don't sit where the camera sees things -- not mounted as set)
  const up = !!ld && !ld.ok && /looks up/.test(ld.why || '');
  const mount = !!ld && !ld.ok && /match the camera|what the camera sees/.test(ld.why || '');
  // (dirty: the L2 reports its window dirty -- its close points are false, none are shown or used)
  const dirty = !!ld && !ld.ok && /window is dirty/.test(ld.why || '');
  setText(lc, !ld ? 'LIDAR –' : ld.ok ? `LIDAR ✓${ld.walls_off ? ' · walls off' : ''}`
    : up ? 'LIDAR ✕ · looks up · walls on' : dirty ? 'LIDAR ✕ · dirty window · walls on' : mount ? 'LIDAR ✕ · mount? · walls on' : 'LIDAR ✕ · walls on');
  lc.classList.toggle('warn', !!ld && !ld.ok);
  lc.title = !ld ? 'No word from the LiDAR bridge (lidar:=off, or terminal 1 not started with it)'
    : ld.ok ? `The 4D LiDAR works: ${ld.rate_hz} clouds/s, ${ld.reach_points ?? ld.work_points} points where the arms work`
      + (ld.lean_deg != null ? ` (tilted ${Math.round(ld.lean_deg)}°). ` : '. ')
      + (ld.walls_off ? 'The green walls are off.' : 'The walls go off at the next capture.')
    : `The 4D LiDAR is not working (${ld.why || 'no word for ' + ld.age_s + ' s'}): the green walls are used.`;
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
  // (AUTO CALIBRATE: the sheet all through its run -- also while it weighs or waits for the box)
  const ac = s.autocal || {};
  const acRun = !!ac.running;
  const cal = s.calibrating || s.confirming || acRun;
  $('stage').classList.toggle('sheet-up', s.teaching || cal);
  $('calSheet').classList.toggle('open', cal && !s.teaching);
  $('calSheet').setAttribute('aria-hidden', !cal);
  $('calConfirm').hidden = !s.confirming;
  setText($('calTitle'), s.confirming ? 'Is the outline on marker 11?'
    : acRun ? `Auto calibration · ${AC_STEP[ac.step] || ac.step}` : 'Calibrating');
  setText($('calSub'), s.confirming ? 'Confirm and the arms start; nothing has moved yet'
    : acRun && ac.step === 'wait_box' ? 'Put the ArUco box on the cup (cup on the cross of face B), then press Box is on the cup'
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
  renderPickAll(s);
  renderBaseCorrection(s);
  renderBaseAlert(s);
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
  renderSizes(s);
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
  // (2026-11-17) the count: progress on the Pick N button while it runs (the
  // card below has the skipped and dropped), else the number to pick
  const pa = s.pick_all;
  setText($('pickAllLabel'), pa
    ? (pa.want ? `Picking ${pa.picked}/${pa.want}` : `Picking all · ${pa.picked}`)
    : `Pick ${pickCount()}`);
  const countOff = !!(s.busy || s.teaching || s.calibrating);
  if ($('pickCount').disabled !== countOff) $('pickCount').disabled = countOff;
  document.querySelectorAll('[data-count-step]').forEach((b) => {
    if (b.disabled !== countOff) b.disabled = countOff;
  });
  $('pickAllCount').hidden = s.click_only;

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
    if (name === 'view' || name === 'apply' || name === 'autocal_box_on') return;   // (set elsewhere)
    const off = (s.busy || s.teaching || s.calibrating) && !ALWAYS.has(name)
      && !(name === 'topdown' && !s.busy);
    if (b.disabled !== off) b.disabled = off;
  });
  if (!s.busy && !s.teaching && !s.calibrating) move.disabled = !retry && (!p || p.ok === false);
  $('btnMove2').disabled = move.disabled;             // (the Advanced tab's MOVE: the same)
  $('btnMove2').dataset.tone = move.dataset.tone || '';
  $('btnRelease').disabled = !heldArms.length || s.busy;
  renderAutocal(s);

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

// -- PICK ALL: each box's time, the total, and "done" -----------------------------------

const BASE_TITLE = document.title;
let reportSeen;                 // the last report id shown (undefined: page just loaded)
let audio = null;               // made on a click: browsers only play sound after one
document.addEventListener('click', () => {
  try {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === 'suspended') audio.resume();
  } catch (_) { /* no sound */ }
}, { capture: true });

function chime(ok) {
  if (!audio) return;
  const notes = ok ? [660, 880, 1320] : [520, 390];
  notes.forEach((f, i) => {
    const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime + i * 0.18;
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    o.connect(g).connect(audio.destination);
    o.start(t); o.stop(t + 0.17);
  });
}

function dur(sec) {
  if (sec == null) return '–';
  return sec < 60 ? `${Number(sec).toFixed(1)} s` : `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
}

// the count (2026-11-17): how many boxes Pick N picks, 1..200 (beast refuses
// anything else), remembered in this browser; nothing is sent until Pick N / All
const COUNT_MAX = 200;
function clampCount(v) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(COUNT_MAX, Math.max(1, n)) : 1;
}
function pickCount() { return clampCount($('pickCount').value); }
function setPickCount(v, save = true) {
  const n = clampCount(v);
  if ($('pickCount').value !== String(n)) $('pickCount').value = String(n);
  if (save) { try { localStorage.setItem('pickCount', String(n)); } catch (_) { /* private window */ } }
  if (!(S && S.pick_all)) setText($('pickAllLabel'), `Pick ${n}`);
}
document.querySelectorAll('[data-count-step]').forEach((b) => b.addEventListener('click', () => {
  if (!b.disabled) setPickCount(pickCount() + Number(b.dataset.countStep));
}));
// typing: the label follows; the field itself is tidied once done (Enter, leaving it)
$('pickCount').addEventListener('input', () => {
  if (!(S && S.pick_all)) setText($('pickAllLabel'), `Pick ${pickCount()}`);
});
$('pickCount').addEventListener('change', () => setPickCount($('pickCount').value));
try {
  const saved = localStorage.getItem('pickCount');
  if (saved) setPickCount(saved, false);
} catch (_) { /* private window */ }

let timesKey = '';
function renderTimes(picks) {
  const key = picks.map((p) => p.n + p.result).join(',');
  if (key === timesKey) return;
  timesKey = key;
  const body = $('pickAllTimes').querySelector('tbody');
  body.innerHTML = '';
  for (const p of picks) {
    const tr = document.createElement('tr');
    // ('picked (by weight)': the camera could not tell, the cup weighed it -- picked)
    const tone = String(p.result).startsWith('picked') ? 'picked'
      : ['skipped', 'dropped', 'tried'].includes(p.result) ? p.result : 'bad';
    const cells = [p.n, p.result, dur(p.seconds), dur(p.cycle), p.at];
    cells.forEach((v, i) => {
      const td = document.createElement('td');
      td.textContent = v;
      if (i === 1) { td.className = tone; td.title = `${p.outcome || ''}${p.approach ? ' · ' + p.approach : ''}${p.face_mm && p.face_mm.length ? ' · face ' + p.face_mm.join(' x ') + ' mm' : ''}`; }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

// (2026-11-17) a run or its report: "Pick 3" / "Pick all", "2 of 3 picked" / "2 picked"
const runName = (r) => (r.want ? `Pick ${r.want}` : 'Pick all');
const pickedText = (r) => (r.want ? `${r.picked} of ${r.want} picked` : `${r.picked} picked`);

function renderPickAll(s) {
  const run = s.pick_all, rep = s.pick_all_report;
  const card = $('pickAllCard'), badge = $('pickAllBadge');
  const state = $('pickAllState');
  setText(state, s.click_only ? 'no box model' : s.base_wait ? 'waiting for the base'
    : run ? 'running' : s.busy ? 'busy' : 'ready');
  state.dataset.tone = s.base_wait ? 'edge' : run ? 'busy' : s.click_only || s.busy ? 'idle' : 'ok';
  if (run) {
    card.hidden = false;
    setText($('pickAllTitle'), `${runName(run)} · running`);
    setText(badge, dur(run.elapsed_s)); badge.dataset.tone = 'busy';
    setText($('pickAllSummary'), `${pickedText(run)}, ${run.skipped} skipped`
      + (run.dropped ? `, ${run.dropped} dropped` : '') + (run.pass > 1 ? ` · round ${run.pass}` : ''));
    renderTimes(run.picks || []);
    setText($('pickAllFile'), '');
  } else if (rep) {
    card.hidden = false;
    setText($('pickAllTitle'), `${runName(rep)} · done ${String(rep.finished).slice(-8)}`);
    setText(badge, dur(rep.total_s)); badge.dataset.tone = rep.ok ? 'ok' : 'edge';
    setText($('pickAllSummary'), `${pickedText(rep)}, ${rep.skipped} skipped`
      + (rep.dropped ? `, ${rep.dropped} dropped` : '') + (rep.rounds > 1 ? ` in ${rep.rounds} rounds` : '')
      + ` · total ${dur(rep.total_s)}` + (rep.per_box_s ? ` · ${dur(rep.per_box_s)} per box picked` : '')
      + ` · ${rep.why}`);
    renderTimes(rep.picks || []);
    setText($('pickAllFile'), rep.file ? `Saved: ${rep.file}` : '');
  } else {
    card.hidden = true;
  }
  // "done": once per new report -- not for one that was there when the page opened
  const id = rep ? rep.id : null;
  if (reportSeen === undefined) { reportSeen = id; return; }
  if (id != null && id !== reportSeen) {
    reportSeen = id;
    const text = `${pickedText(rep)}, ${rep.skipped} skipped`
      + (rep.dropped ? `, ${rep.dropped} dropped` : '') + ` in ${dur(rep.total_s)}`
      + (rep.per_box_s ? ` (${dur(rep.per_box_s)} per box)` : '') + ` – ${rep.why}`;
    const title = `${runName(rep)} ${rep.ok ? 'done' : 'stopped'}`;
    $('doneBanner').classList.toggle('bad', !rep.ok);
    setText($('doneTitle'), title);
    setText($('doneText'), text);
    $('doneBanner').hidden = false;
    document.title = `${rep.ok ? '✓' : '!'} ${title} · ` + BASE_TITLE;
    chime(rep.ok);
    try {                                   // (only where the browser allows it: https / localhost)
      if (window.Notification && Notification.permission === 'granted') {
        new Notification(title, { body: text });
      }
    } catch (_) { /* in-page only */ }
  }
}

// -- the base correction: where the robot should stand for a box it cannot pick ------

const ARROW = { left: '←', right: '→', forward: '↑', back: '↓' };
const cmText = (m) => `${Math.round(m * 100)} cm`;

function correctionMap(c) {
  // the robot from above, facing up: its front at the bottom edge; the spot the box
  // should be in (green), and where the box is (orange). x ahead 0..0.7 m, y +-0.35 m
  const W = 84, H = 84, k = H / 0.7, cx = W / 2;
  const px = (y) => cx - y * k, py = (x) => H - x * k;
  const sp = c.spot;
  const bx = Math.min(Math.max(px(c.side_m), 6), W - 6), by = Math.min(Math.max(py(c.ahead_m), 6), H - 6);
  return `<svg class="cmap" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"
      aria-label="Top view: the robot at the bottom, the box and where it should be">
    <rect x="${px(sp.left)}" y="${py(sp.far)}" width="${Math.max((sp.left - sp.right) * k, 5)}"
      height="${Math.max((sp.far - sp.near) * k, 5)}" rx="2" class="cmap-spot"/>
    <rect x="${cx - 18}" y="${H - 7}" width="36" height="7" rx="2" class="cmap-robot"/>
    <line x1="${cx}" y1="${H - 7}" x2="${cx}" y2="3" class="cmap-axis"/>
    <circle cx="${bx}" cy="${by}" r="4.5" class="cmap-box"/>
  </svg>`;
}

function correctionHtml(c, where, noMove) {
  const top = c.approach === 'top';
  const moves = c.moves.length
    ? c.moves.map((m) => `<div class="move"><span class="arrow">${ARROW[m.dir] || '•'}</span>
        <span class="dir">Move ${m.dir}</span><span class="dist">${cmText(m.m)}</span></div>`).join('')
    : `<div class="move ok"><span class="arrow">✓</span><span class="dir">In its zone</span></div>`;
  const side = Math.abs(c.side_m) < 0.005 ? 'centred' : `${cmText(Math.abs(c.side_m))} ${c.side_m > 0 ? 'left' : 'right'}`;
  const across = (v) => (Math.abs(v) < 0.005 ? 'centre' : `${cmText(Math.abs(v))} ${v > 0 ? 'left' : 'right'}`);
  const want = `${Math.round(c.spot.near * 100)}–${Math.round(c.spot.far * 100)} cm ahead, `
    + `${across(c.spot.right)} to ${across(c.spot.left)}`;
  return `<div class="card-row"><span class="label">Base correction</span>
      <span class="badge" data-tone="edge">${top ? 'Top' : 'Front'} pick</span></div>
    <div class="corr">
      <div class="moves">${moves}</div>
      ${correctionMap(c)}
    </div>
    <div class="why">The box is ${cmText(c.ahead_m)} ahead, ${side}. ${top ? 'Top' : 'Front'} picks work
      ${want} (green)${c.moves.length ? (c.centre ? '; these moves bring it to the middle of that zone.'
        : '; these are the shortest moves into it.') : '. It is in that zone already: it cannot be picked for another reason (below).'}</div>
    <div class="why small">${c.why || ''} · ${c.time}${c.more ? ` · ${c.more} more box${c.more > 1 ? 'es' : ''} out of reach` : ''}
      · ${noMove ? 'shown only: Pick all does not wait for the base (No base movement)'
                 : 'Pick all waits for the base to be moved, then picks it'}</div>`;
}

// Pick all waiting for the base: the alert, until the box is in place (or Skip / STOP)
let alertShown = false;
function renderBaseAlert(s) {
  const w = s.base_wait;
  setPressed($('tgNoBaseMove'), s.no_base_move);
  setText($('noBaseMoveHint'), s.no_base_move
    ? 'A box it cannot reach from here is skipped.'
    : 'A box it cannot reach: it stops, shows how to move the base, and picks it by itself once the box is in place.');
  const el = $('baseAlert');
  if (!w) {
    if (alertShown) {
      el.hidden = true; alertShown = false;
      if (document.title.startsWith('! Move the base')) document.title = BASE_TITLE;
    }
    return;
  }
  const c = w.correction || {};
  const moves = c.moves || [];
  $('baseAlertMoves').innerHTML = moves.length
    ? moves.map((m) => `<span class="ba-move"><span class="arrow">${ARROW[m.dir] || '•'}</span>${m.dir[0].toUpperCase() + m.dir.slice(1)} <b>${cmText(m.m)}</b></span>`).join('')
    : '<span class="ba-move ok"><span class="arrow">✓</span>in place</span>';
  setText($('baseAlertWhy'), `${c.approach === 'top' ? 'Top' : 'Front'} pick · ${c.why || ''}`);
  setText($('baseAlertNote'), (w.note || '') + (w.checking ? '' : ' · Pick all goes on by itself once the box is in place'));
  el.classList.toggle('checking', !!w.checking);
  el.querySelector('[data-action="base_try"]').disabled = !!w.checking;
  if (!alertShown) {
    el.hidden = false; alertShown = true;
    document.title = '! Move the base · ' + BASE_TITLE;
    chime(false);
  }
}

let corrKey = '';
function renderBaseCorrection(s) {
  const c = s.base_correction;
  const key = c ? JSON.stringify(c) + s.no_base_move : '';
  if (key === corrKey) return;
  corrKey = key;
  for (const [id, where] of [['baseCorrPick', 'pick'], ['baseCorrAll', 'all']]) {
    const el = $(id);
    el.hidden = !c;
    el.innerHTML = c ? correctionHtml(c, where, s.no_base_move) : '';
  }
}

$('doneClose').addEventListener('click', () => {
  $('doneBanner').hidden = true;
  document.title = BASE_TITLE;
});
window.addEventListener('focus', () => {
  if ($('doneBanner').hidden) document.title = BASE_TITLE;
});

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

// -- AUTO CALIBRATE: the one button, its progress, result and error map (2026-11-16) --

const AC_STEP = {
  weigh: 'weighing the tool', wait_box: 'waiting for the box', choose: 'choosing postures',
  measure: 'measuring', fit: 'fitting the model', done: 'done', failed: 'failed', aborted: 'aborted',
};
const AC_RUNNING = ['weigh', 'wait_box', 'choose', 'measure', 'fit'];
const AC_MODEL = {
  urdf: 'URDF (no correction)', rigid: 'Rigid (base only)', zeros: 'Joint zeros',
  'zeros+sag': 'Joint zeros + sag',
};
const AC_ORDER = ['urdf', 'rigid', 'zeros', 'zeros+sag'];      // simplest first, as the run chooses
// |error| in 4 mm bins, one hue: larger is lighter on dark, darker on light (both
// checked with the dataviz validator: monotone, visible steps, the first clears the map)
const AC_BINS = [4, 8, 12, 16];
const AC_RAMP = {
  dark: ['#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
  light: ['#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
};
const lightScheme = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
function acColour(e) {
  const ramp = AC_RAMP[lightScheme && lightScheme.matches ? 'light' : 'dark'];
  let i = 0;
  while (i < AC_BINS.length && e >= AC_BINS[i]) i++;
  return ramp[i];
}
const mm1 = (v) => (v == null || !isFinite(v) ? '–' : `${Number(v).toFixed(1)} mm`);
const signed = (v) => (v == null || !isFinite(v) ? '–' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)}`);
const eta = (sec) => (sec < 90 ? `${Math.round(sec)} s` : `${Math.round(sec / 60)} min`);
// [mean, p90, max]: the run sends lists; {mean_mm, p90_mm, max_mm} is read too
const triple = (v) => (Array.isArray(v) ? v : v && typeof v === 'object'
  ? [v.mean_mm ?? v.mean, v.p90_mm ?? v.p90, v.max_mm ?? v.max] : [v, null, null]);
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let acProgKey = '', acResKey = '', acMapKey = '', acEndSeen, acWaitShown = false;
if (lightScheme && lightScheme.addEventListener) {
  lightScheme.addEventListener('change', () => { acMapKey = ''; if (S) renderAutocal(S); });
}

function renderAutocal(s) {
  const ac = s.autocal || {};
  const step = ac.step || null;
  const running = !!ac.running;                 // (beast: busy, and a step of the run)
  const waiting = running && step === 'wait_box';
  const r = ac.result || null;
  // Box is on the cup: only while it waits -- busy and calibrating then, so not
  // by the busy rule in render() (it skips this button)
  for (const id of ['btnBoxOn', 'calBoxOn']) {
    const b = $(id);
    if (b.hidden !== !waiting) b.hidden = !waiting;
    if (b.disabled !== !waiting) b.disabled = !waiting;
  }
  $('btnAutoCal').hidden = waiting;
  setText($('btnAutoCal'), running ? 'Auto calibrating…' : 'Auto calibrate');
  $('acWeigh').disabled = !!s.busy;
  const stale = AC_RUNNING.includes(step) && !running;   // its job ended without a last word
  const [label, tone] = waiting ? ['put the box on', 'edge']
    : running ? [AC_STEP[step] || step, 'busy']
    : step === 'done' ? (r && r.saved ? ['saved', 'ok'] : ['not saved', 'edge'])
    : step === 'failed' || stale ? ['failed', 'bad']
    : step === 'aborted' ? ['aborted', 'edge'] : ['ready', 'idle'];
  setText($('acBadge'), label);
  $('acBadge').dataset.tone = tone;

  // it waits for you: say so once, also from another tab
  if (waiting && !acWaitShown) {
    acWaitShown = true;
    chime(true);
    document.title = '! Put the box on the cup · ' + BASE_TITLE;
  } else if (!waiting && acWaitShown) {
    acWaitShown = false;
    if (document.title.startsWith('! Put the box')) document.title = BASE_TITLE;
  }

  // progress
  const segs = Array.isArray(ac.segments) ? ac.segments.length : Number(ac.segments) || 1;
  const progKey = JSON.stringify([step, running, ac.text, ac.measured, ac.total, segs, ac.eta_s]);
  if (progKey !== acProgKey) {
    acProgKey = progKey;
    $('acProgress').hidden = !(running || step === 'failed' || step === 'aborted' || stale);
    setText($('acStep'), ac.text || AC_STEP[step] || '');
    $('acStep').classList.toggle('bad', step === 'failed' || stale);
    const total = Number(ac.total) || 0;
    const n = Math.max(0, Math.min(Number(ac.measured) || 0, total));
    $('acBarWrap').hidden = !total;
    $('acBarWrap').setAttribute('aria-valuemax', String(total));
    $('acBarWrap').setAttribute('aria-valuenow', String(n));
    $('acBar').style.transform = `scaleX(${total ? n / total : 0})`;
    const parts = [];
    if (total) parts.push(`${n} of ${total} postures measured`);
    if (running && ac.eta_s > 0) parts.push(`about ${eta(ac.eta_s)} left`);
    if (segs > 1) parts.push(`the box moved on the cup ${segs - 1}× (accounted for)`);
    setText($('acCounts'), parts.join(' · '));
  }

  // the result
  const resKey = JSON.stringify(r);
  if (resKey !== acResKey) {
    acResKey = resKey;
    $('acResult').hidden = !r;
    if (r) renderAutocalResult(r);
  }

  // the error map
  const pts = ac.points || [];
  const mapKey = JSON.stringify(pts);
  if (mapKey !== acMapKey) {
    acMapKey = mapKey;
    $('acMaps').hidden = !pts.length;
    $('acMaps').innerHTML = pts.length ? autocalMaps(pts) : '';
  }

  // the end: a chime and a toast once per run -- not for one that ended before the page opened
  const end = ['done', 'failed', 'aborted'].includes(step) ? `${ac.report_id}|${step}|${ac.text}` : null;
  if (acEndSeen === undefined) { acEndSeen = end; return; }
  if (end && end !== acEndSeen) {
    acEndSeen = end;
    const ok = step === 'done';
    chime(ok);
    toast(ok ? `Auto calibration done: ${r && r.saved ? 'saved' : 'not saved, the old one stays'}`
      : `Auto calibration ${step}`, !ok);
  }
}

function renderAutocalResult(r) {
  // every model's cross-validated error, the chosen one marked
  const cv = r.cv || {};
  const names = AC_ORDER.filter((k) => k in cv).concat(Object.keys(cv).filter((k) => !AC_ORDER.includes(k)));
  const body = $('acCvBody');
  body.innerHTML = '';
  for (const k of names) {
    const [mean, p90] = triple(cv[k]);
    const tr = document.createElement('tr');
    const chosen = k === r.variant;
    if (chosen) {
      tr.className = 'chosen';
      tr.title = 'Chosen: the simplest model within 10 % (+0.3 mm) of the best';
    }
    [`${chosen ? '✓ ' : ''}${AC_MODEL[k] || k}`, mm1(mean), mm1(p90)].forEach((v) => {
      const td = document.createElement('td');
      td.textContent = v;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
  const [vm, vp, vx] = triple(r.val);
  setText($('acVal'), r.val == null ? '' : `Chosen: ${AC_MODEL[r.variant] || r.variant || '?'} – ${mm1(vm)} `
    + `mean, 90 % under ${mm1(vp)}, worst ${mm1(vx)}, on postures it was not fitted to.`);
  const b = r.baseline || {};
  const bu = triple(b.urdf)[0], bc = triple(b.current)[0];
  setText($('acBaseline'), bu == null && bc == null ? '' : `Before, on the same postures: plain URDF ${mm1(bu)}`
    + (bc != null ? `, the old model ${mm1(bc)}.` : ' (there was no old model).'));
  setText($('acSaved'), r.saved ? `Saved${r.why ? ': ' + r.why : ''}. Used from the next pick.`
    : `Not saved${r.why ? ': ' + r.why : ''}. The old calibration stays.`);
  $('acSaved').dataset.tone = r.saved ? 'ok' : 'edge';
  $('acNotes').innerHTML = [['Box on the cup', r.drift], ['Depth', r.depth], ['Tool weight', r.weigh]]
    .filter(([, t]) => t).map(([k, t]) => `<div><b>${k}:</b> ${esc(t)}</div>`).join('');
  $('acRestart').hidden = !r.restart_needed;
}

function autocalMaps(pts) {
  // world metres. Top view: the robot at the bottom facing up, x ahead 0..0.75,
  // y +-0.4 (left is left). Side view: x ahead to the right, height z 0.2..1.0 up.
  // Each tip measured, coloured by |error| (4 mm bins); its line is the error
  // (model minus camera) 4x magnified. Hover a point for its numbers.
  const K = 180, L = 4 * K / 1000, Z0 = 0.2, Z1 = 1.0;     // K: map units a metre, L: a mm
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo + 4), hi - 4);
  const top = { w: 0.8 * K, h: 0.75 * K, at: (p) => [0.4 * K - p.y * K, 0.75 * K - p.x * K],
    err: (p) => [-(p.dy || 0) * L, -(p.dx || 0) * L] };
  const side = { w: 0.75 * K, h: (Z1 - Z0) * K, at: (p) => [p.x * K, (Z1 - p.z) * K],
    err: (p) => [(p.dx || 0) * L, -(p.dz || 0) * L] };
  const tip = (p) => `${p.kind || '?'} · x ${fmt(p.x)} y ${fmt(p.y)} z ${fmt(p.z)} m · error `
    + (p.e == null ? 'not fitted yet' : `${mm1(p.e)} (dx ${signed(p.dx)}, dy ${signed(p.dy)}, dz ${signed(p.dz)})`);
  const view = (v, label, extra) => {
    const lines = [], marks = [];
    for (const p of pts) {
      if (p.x == null) continue;
      const [u0, w0] = v.at(p);
      const u = clamp(u0, 0, v.w).toFixed(1), w = clamp(w0, 0, v.h).toFixed(1);   // (across, down)
      const has = p.e != null;
      if (has) {
        const [du, dw] = v.err(p);
        lines.push(`<line x1="${u}" y1="${w}" x2="${(+u + du).toFixed(1)}" y2="${(+w + dw).toFixed(1)}" class="acmap-err"/>`);
      }
      const fill = has ? `fill="${acColour(Math.abs(p.e))}"` : 'class="acmap-open"';
      const mark = p.kind === 'top'
        ? `<rect x="${(u - 3.5).toFixed(1)}" y="${(w - 3.5).toFixed(1)}" width="7" height="7" rx="1" ${fill}/>`
        : `<circle cx="${u}" cy="${w}" r="4" ${fill}/>`;
      marks.push(`<g class="acmap-pt"><title>${esc(tip(p))}</title><circle cx="${u}" cy="${w}" r="8" class="acmap-hit"/>${mark}</g>`);
    }
    return `<figure><svg viewBox="0 0 ${v.w} ${v.h}" class="acmap" role="img" aria-label="${label}">
        <rect x="0.5" y="0.5" width="${v.w - 1}" height="${v.h - 1}" rx="6" class="acmap-frame"/>${extra}
        ${lines.join('')}${marks.join('')}</svg><figcaption>${label}</figcaption></figure>`;
  };
  const cx = 0.4 * K;
  const topExtra = `<line x1="${cx}" y1="4" x2="${cx}" y2="${top.h - 8}" class="acmap-axis"/>
    <rect x="${cx - 16}" y="${top.h - 7}" width="32" height="7" rx="2" class="acmap-robot"/>`;
  // the numbers under the maps: the mean error vector (mostly up or down: a height
  // offset), and front against top
  const fitted = pts.filter((p) => p.e != null);
  const mean = (k, list) => list.reduce((a, p) => a + (Number(p[k]) || 0), 0) / Math.max(list.length, 1);
  const byKind = ['front', 'top'].map((k) => [k, fitted.filter((p) => p.kind === k)]).filter(([, l]) => l.length)
    .map(([k, l]) => `${k} ${mm1(mean('e', l))} (${l.length})`).join(', ');
  const swatches = AC_BINS.concat([null]).map((hi, i) => {
    const lo = i ? AC_BINS[i - 1] : 0;
    return `<span><i style="background:${AC_RAMP[lightScheme && lightScheme.matches ? 'light' : 'dark'][i]}"></i>${hi == null ? `≥${lo}` : `${lo}–${hi}`}</span>`;
  }).join('');
  return `<div class="ac-mapgrid">${view(top, 'Top view · x ahead 0–0.75 m, y ±0.4 m', topExtra)}
      ${view(side, 'Side view · x ahead 0–0.75 m, height 0.2–1.0 m', '')}</div>
    <div class="ac-legend">${swatches}<span class="muted">mm</span>
      <span class="ac-shape">● front ■ top</span></div>
    <div class="why small">${fitted.length ? `Mean error (model − camera): dx ${signed(mean('dx', fitted))}, `
      + `dy ${signed(mean('dy', fitted))}, dz ${signed(mean('dz', fitted))} mm · by approach: ${byKind}. ` : ''}`
      + `Each point's line is its error, 4× magnified; hover a point for its numbers.</div>`;
}

// the dummy size: sent when a field is changed (Enter or leaving it)
['dimH', 'dimW', 'dimD'].forEach((id) => $(id).addEventListener('change', () => {
  act('dummy_dims', { h: $('dimH').value, w: $('dimW').value, d: $('dimD').value })
    .then((j) => { if (j && j.ok) toast(j.message); });
}));

// named held-object sizes (2026-11-18): pick one to use it, Save the fields under a name
let sizeKeys = null;
function renderSizes(s) {
  const presets = s.size_presets || {};
  const keys = JSON.stringify(Object.entries(presets).sort());
  const pick = $('sizePick');
  if (keys !== sizeKeys && document.activeElement !== pick) {
    sizeKeys = keys;
    pick.textContent = '';
    const first = document.createElement('option');
    first.value = '';
    first.textContent = Object.keys(presets).length ? 'Saved…' : 'None saved';
    pick.appendChild(first);
    Object.keys(presets).sort((a, b) => a.localeCompare(b)).forEach((name) => {
      const o = document.createElement('option');
      const [h, w, d] = presets[name];
      o.value = name;
      o.textContent = `${name} · ${Math.round(h)} × ${Math.round(w)} × ${Math.round(d)}`;
      pick.appendChild(o);
    });
  }
  if (document.activeElement !== pick) pick.value = s.size_name && presets[s.size_name] ? s.size_name : '';
  $('btnSizeDel').disabled = !pick.value;
}
$('sizePick').addEventListener('change', () => {
  const name = $('sizePick').value;
  $('btnSizeDel').disabled = !name;
  if (name) act('size_use', { label: name }).then((j) => { if (j) toast(j.message); });
});
$('btnSizeSave').addEventListener('click', () => {
  const name = $('sizeName').value.trim() || $('sizePick').value;
  if (!name) { toast('Type a name for this size first'); $('sizeName').focus(); return; }
  act('size_save', { label: name }).then((j) => { if (j) { toast(j.message); if (j.ok) $('sizeName').value = ''; } });
});
$('sizeName').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btnSizeSave').click(); });
$('btnSizeDel').addEventListener('click', () => {
  const name = $('sizePick').value;
  if (name) act('size_delete', { label: name }).then((j) => { if (j) toast(j.message); });
});

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
