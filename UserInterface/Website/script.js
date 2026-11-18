// ═══════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════
// const SERVO_DEFS = [
//   { id: 'FrontLeft',  name: 'Front Left',  min: -45, max: 45, val: 0 },
//   { id: 'FrontRight', name: 'Front Right', min: -45, max: 45, val: 0 },
//   { id: 'RearLeft',   name: 'Rear Left',   min: -45, max: 45, val: 0 },
//   { id: 'RearRight',  name: 'Rear Right',  min: -45, max: 45, val: 0 },
//   { id: 'Head',       name: 'Head',        min: -60, max: 60, val: 0 },
// ]

// Slider ranges = the servo driver's command limits (Canine/ServoDriver.py Max_Leg_Command,
// Max_Head_Command), which clamp every command anyway. +-180 let a leg reach its end stop and
// stall the servo rail (11-17). tests/check_ui.py fails if these drift from the driver.
const SERVO_DEFS = [
  { id: 'FrontLeft',  name: 'Front Left',  min: -45, max: 45, val: 0 },
  { id: 'FrontRight', name: 'Front Right', min: -45, max: 45, val: 0 },
  { id: 'RearLeft',   name: 'Rear Left',   min: -45, max: 45, val: 0 },
  { id: 'RearRight',  name: 'Rear Right',  min: -45, max: 45, val: 0 },
  { id: 'Head',       name: 'Head',        min: -60, max: 60, val: 0 },
]

const AUDIO_DEFS = [
  { id: 'BarkShort',      label: '🔊 BarkShort'      },
  { id: 'BarkBurst',      label: '🔊 BarkBurst'      },
  { id: 'Growl',          label: '🔊 Growl'          },
  { id: 'AggressiveBark', label: '🔊 AggressiveBark' },
  { id: 'BarkShort_2x2',  label: '🔊 BarkShort_2x2' },
  { id: 'BarkShort_3x',   label: '🔊 BarkShort_3x'  },
  { id: 'Stop',           label: 'Stop Audio'         },
]

// ═══════════════════════════════════════════════
// STATE — all let declarations hoisted here so
// the init block at the bottom can reference them
// safely (let is block-scoped, not hoisted like var)
// ═══════════════════════════════════════════════

// WebSocket
let ws      = null
let wsReady = false

// Mode
let isManual = false
let isStopped = false

// Map state
let mapTurkeys       = []
let mapHumans        = []
let mapObstacles     = []
let mapWaypoints     = []
let selectedWaypoint = null
let mapRobot         = { x: 0, y: 0, theta: 0 }
let mapReferenceTraj = []
let mapMPCHorizon    = []
let _dashOffset      = 0

// Map viewport
let MAP_W      = 100.0
let MAP_H      = 100.0
let mapZoom    = 1.0
let mapPanX    = 0.0
let mapPanY    = 0.0
let _dragActive = false
let _dragLast   = { x: 0, y: 0 }
let _lastS      = 1

// Camera — base64/WebSocket fields removed; MJPEG uses an <img> element
const CAM_TIMEOUT_MS = 3000
// Last detector update (vision Meta, ~28 Hz). Chrome fires an <img> load only once for an
// MJPEG stream, so the robot's own frame updates are the reliable sign the camera is live.
let _lastVisionMs = 0

// Pose / macrostate
let _activePose       = null
let _activeMacrostate = null
let _activeSubCmd     = null

// Heartbeat
let _heartbeatTimer = null
let _lastBeatMs     = 0

// ═══════════════════════════════════════════════
// BUILD DYNAMIC UI
// ═══════════════════════════════════════════════
function buildServos() {
  const list = document.getElementById('servo-list')
  SERVO_DEFS.forEach(def => {
    const row = document.createElement('div')
    row.className = 'servo-row'
    row.innerHTML = `
      <span class="servo-name">${def.name}</span>
      <input type="range" class="srange"
             id="sr-${def.id}"
             min="${def.min}" max="${def.max}" value="${def.val}"
             oninput="onServoSlide('${def.id}', this)">
      <span class="servo-val" id="sv-${def.id}">${def.val}°</span>
    `
    list.appendChild(row)
    updateSliderBg(document.getElementById(`sr-${def.id}`))
  })
}

function buildAudio() {
  const grid = document.getElementById('audio-grid')
  AUDIO_DEFS.forEach(a => {
    const btn = document.createElement('button')
    btn.className       = 'btn pu'
    btn.textContent     = a.label
    btn.dataset.audioId = a.id
    btn.onclick         = () => playAudio(a.id, btn)
    grid.appendChild(btn)
  })
}

// ═══════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════
function connect() {
  // Same host that served this page: the Pi over Ethernet (192.168.10.2), over the phone
  // hotspot, or localhost on a laptop running --profile sim. No address to edit per network.
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(`${proto}://${location.host}/ws`)

  ws.onopen    = ()  => { wsReady = true;  _startHeartbeat() }
  ws.onclose   = ()  => { wsReady = false; _stopHeartbeat(); setTimeout(connect, 1000) }
  ws.onmessage = (e) => {
    try   { handle(JSON.parse(e.data)) }
    catch (err) { console.error('handle() error:', err) }
  }
}

function send(obj) {
  if (wsReady && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify(obj))
}

// ═══════════════════════════════════════════════
// INCOMING MESSAGES
// ═══════════════════════════════════════════════
function handle(m) {

  // ── State (10 Hz) ──────────────────────────────────────────
  if (m.type === 'state') {

    // Answer the robot's 10 Hz state with a heartbeat. Browsers slow timers in a background
    // tab to about 1 per second, which tripped the 1 s watchdog; incoming WebSocket messages
    // aren't slowed, so this keeps the beat going while the tab is hidden.
    _beat()

    _applyPlannerState(m)

    // Raw sensors
    set('uwb-x', fmt(m.uwb_x, 2))
    set('uwb-y', fmt(m.uwb_y, 2))
    set('imu-t', deg(m.imu_theta))

    // UKF fused pose
    set('ukf-x', fmt(m.x, 2))
    set('ukf-y', fmt(m.y, 2))
    set('ukf-t', deg(m.theta))
    if (m.x != null) mapRobot = { x: m.x, y: m.y, theta: m.theta ?? 0, targetX: m.target?.x, targetY: m.target?.y }

    // GPS / IMU / drivetrain dots: real data arriving, simulated, or not running
    if (m.sources) applySources(m.sources)

    // Nav command type
    {
      const nc  = m.nav_cmd_type ?? null
      const el  = document.getElementById('nav-cmd-type')
      const det = document.getElementById('nav-cmd-detail')
      if (el) {
        if      (nc === 'Move')   { el.textContent = 'MOVE';   el.className = 'v ok' }
        else if (nc === 'Orient') { el.textContent = 'ORIENT'; el.className = 'v w'  }
        else                      { el.textContent = 'IDLE';   el.className = 'v'    }
      }
      if (det && m.target != null) {
        if      (nc === 'Move')   det.textContent = `→ (${fmt(m.target.x,2)}, ${fmt(m.target.y,2)}) m`
        else if (nc === 'Orient') det.textContent = `→ (${fmt(m.target.x,2)}, ${fmt(m.target.y,2)}) m  θ=${deg(m.target.theta)}°`
        else                      det.textContent = '—'
      }
    }

    if (m.is_estopped !== undefined) {
      _applyEstopUI(m.is_estopped)
    }

    // Current destination
    if (m.target != null) {
      set('dest-x', fmt(m.target.x, 2))
      set('dest-y', fmt(m.target.y, 2))
      set('dest-t', deg(m.target.theta))
    }

    // Drivetrain readout
    set('cmd-l', fmt(m.cmd_left,  2))
    set('cmd-r', fmt(m.cmd_right, 2))

    // Servo sliders — always sync
    if (Array.isArray(m.servos)) syncServos(m.servos)

    // MacroState button highlights
    if (m.macrostate !== undefined) {
      _activeMacrostate = m.macrostate
      _syncMacrostateButtons(_activeMacrostate)
    }

    if (m.substate !== undefined) {
      _activeSubCmd = m.substate
      _syncSubCmdButtons(_activeSubCmd)
    }

    // Speed slider
    if (m.speed_ms != null) {
      const sl = document.getElementById('spd-range')
      if (sl) { sl.value = m.speed_ms; updateSliderBg(sl) }
      set('spd-disp', m.speed_ms.toFixed(1))
    }

    // Pose button highlight
    if (m.active_pose !== undefined) {
      _activePose = m.active_pose
      _syncPoseButtons(_activePose)
    }

    // Audio button highlight
    if (m.active_audio !== undefined) {
      document.querySelectorAll('#audio-grid .btn').forEach(b => {
        b.classList.toggle('lit', b.dataset.audioId === m.active_audio)
      })
    }

    // ── Protocol 6.1 — Reference trajectory ──────────────────
    if (m.reference_trajectory != null) {
      mapReferenceTraj = m.reference_trajectory
    }

    // ── Protocol 6.2 — MPC horizon ────────────────────────────
    if (m.mpc_horizon != null) {
      mapMPCHorizon = m.mpc_horizon
    }

    // Mode sync
    if (m.is_manual !== undefined && m.is_manual !== isManual) {   // only on change: rewriting
      isManual = m.is_manual                                       // the mode button 10x/s ate clicks
      _applyModeUI()
    }
  }

  // NOTE: 'frame' message type is intentionally removed.
  // Camera is now served via MJPEG on GET /camera (see _initCamera below).

  // ── Servo state ─────────────────────────────────────────────
  if (m.type === 'servo_state') syncServos(m.servos)

  // ── Map obstacles ───────────────────────────────────────────
  if (m.type === 'map_obstacles') {
    mapObstacles = m.obstacles ?? []
    mapWaypoints = m.waypoints ?? []
    if (m.map_w) MAP_W = m.map_w
    if (m.map_h) MAP_H = m.map_h
    const sub = document.getElementById('map-sub')
    if (sub) sub.textContent = `${MAP_W} × ${MAP_H} M · 10 HZ`
  }

  // ── Vision detections ────────────────────────────────────────
  if (m.type === 'vision') {
    if (m.meta) _lastVisionMs = Date.now()
    mapTurkeys = (m.turkeys ?? []).map(t => [t[0], t[1]])
    mapHumans  = (m.humans  ?? []).map(h => [h[0], h[1]])
    applyDetectorMeta(m.meta)
  }

  // ── Geofence (contract C4) ───────────────────────────────────
  if (m.type === 'geofence') applyGeofence(m)
}

// ── Detector readout on the camera panel ──────────────────────
// Meta comes with the onboard detector (contract C2): backend, fps, one confidence per turkey.
function applyDetectorMeta(meta) {
  const el = document.getElementById('cam-sub')
  if (!el || !meta) return
  const parts = [`DETECTOR ${String(meta.backend ?? '?').toUpperCase()}`]
  if (meta.fps != null) parts.push(`${Math.round(meta.fps)} FPS`)
  const conf = meta.confidence ?? []
  parts.push(conf.length ? `TURKEY ${Math.max(...conf).toFixed(2)}` : 'NO TURKEY')
  el.textContent = parts.join(' · ')
}

// ── Geofence status dot ───────────────────────────────────────
//   inside → green · near_edge → amber · outside / unknown → red
function applyGeofence(m) {
  const dot = document.getElementById('gf-dot')
  if (!dot) return
  dot.className = 'dot'
  if      (m.state === 'near_edge') dot.classList.add('warn')
  else if (m.state !== 'inside')    dot.classList.add('err')
  const d = m.dist == null ? '' : ` · ${m.dist.toFixed(1)} m to edge`
  dot.parentElement.title = `Geofence ${m.state}${d} · fix ${m.fix ?? '?'}`
}

// ── GPS / IMU / drivetrain status dots ────────────────────────
// From the robot's 'sources' (UserInterface/Robot.py): what this run profile starts for each
// source, and how long ago its data last arrived.
//   grey    not running, or simulated with no data    hollow amber  simulated, data arriving
//   green   real hardware, data arriving              red           real hardware, no data for > 2 s
const SOURCE_DOTS    = { gps: ['gps-dot', 'GPS'], imu: ['imu-dot', 'IMU'], drivetrain: ['dt-dot', 'Drivetrain'] }
const SOURCE_STALE_S = 2

function applySources(sources) {
  for (const [name, [id, label]] of Object.entries(SOURCE_DOTS)) {
    const dot = document.getElementById(id)
    const s   = sources[name]
    if (!dot || !s) continue
    let cls, tip
    const fresh = s.age != null && s.age <= SOURCE_STALE_S
    if (s.mode === 'off') {
      cls = 'dot off'; tip = `${label}: not running in this profile`
    } else if (s.mode === 'sim' && !fresh) {
      cls = 'dot off'; tip = `${label}: simulated, sends no data`
    } else if (!fresh) {
      cls = 'dot err'; tip = `${label}: no data ${s.age == null ? 'yet' : `for ${s.age} s`}`
    } else if (s.mode === 'sim') {
      cls = 'dot sim'; tip = `${label}: simulated, not real hardware`
    } else {
      cls = 'dot'; tip = `${label}: live`
    }
    if (dot.className !== cls) dot.className = cls
    dot.parentElement.title = tip
  }
}

// ── Drivetrain status dot ─────────────────────────────────────
function applyDrivetrainStatus(status) {
  const dot = document.getElementById('dt-dot')
  if (!dot) return
  dot.className = 'dot'
  if      (status === 1) dot.classList.add('warn')
  else if (status === 2) dot.classList.add('err')
}

// ── Pose state machine ────────────────────────────────────────
function _syncPoseButtons(activePoseId) {
  document.querySelectorAll('.pose-g .btn').forEach(b => {
    b.classList.toggle('lit', b.dataset.poseId === activePoseId)
  })
}

// ── Planner state readout ─────────────────────────────────────
// Manual and E-stop come first: the planner's last state goes stale once the UI takes over.
const PLANNER_STATES = {
  Deter:     ['DETER · TURKEY CONFIRMED', 'v e'],
  DeterIdle: ['DETER · WATCHING',         'v ok'],
  Scanning:  ['SCAN',                     'v ok'],
  Chase:     ['CHASE',                    'v w'],
  Alert:     ['ALERT',                    'v w'],
  Stop:      ['STOPPED',                  'v'],
}

function _applyPlannerState(m) {
  const el = document.getElementById('planner-state')
  if (!el) return
  let text, cls
  if      (m.is_estopped)       { text = 'E-STOP';  cls = 'v e' }
  else if (m.is_manual)         { text = 'MANUAL';  cls = 'v'   }
  else if (PLANNER_STATES[m.macrostate]) [text, cls] = PLANNER_STATES[m.macrostate]
  else                          { text = m.macrostate ?? '—'; cls = 'v' }
  if (el.textContent !== text) el.textContent = text
  if (el.className   !== cls)  el.className   = cls
}

// ── Macrostate state machine ───────────────────────────────────
function _syncMacrostateButtons(activeState) {
  document.querySelectorAll('.macrostate-g .btn[data-state]').forEach(b => {
    b.classList.toggle('lit', b.dataset.state === activeState)
  })
}

// ── Sub-command state machine ─────────────────────────────────
function _syncSubCmdButtons(activeCmdId) {
  document.querySelectorAll('.btn.sub-cmd').forEach(b => {
    b.classList.toggle('lit', b.dataset.subCmd === activeCmdId)
  })
}

// ── Shared servo sync helper ──────────────────────────────────
function syncServos(servos) {
  servos?.forEach(sv => {
    const sl = document.getElementById(`sr-${sv.id}`)
    const vl = document.getElementById(`sv-${sv.id}`)
    if (sl) { sl.value = sv.angle; updateSliderBg(sl) }
    if (vl) vl.textContent = `${Number(sv.angle).toFixed(1)}°`
  })
}

// ═══════════════════════════════════════════════
// SERVO CONTROLS
// ═══════════════════════════════════════════════
function onServoSlide(servoId, el) {
  updateSliderBg(el)
  const angle = parseFloat(el.value)
  document.getElementById(`sv-${servoId}`).textContent = `${angle}°`
  send({ type: 'servo_angle', servo: servoId, angle })
}

function updateSliderBg(el) {
  const pct = ((parseFloat(el.value) - parseFloat(el.min)) /
               (parseFloat(el.max)  - parseFloat(el.min))) * 100
  el.style.setProperty('--pct', `${pct}%`)
}

// ═══════════════════════════════════════════════
// AUDIO CONTROLS
// ═══════════════════════════════════════════════
function playAudio(soundId, btn) {
  send({ type: 'audio_play', sound: soundId })
}

// ═══════════════════════════════════════════════
// SPEED CONTROL
// ═══════════════════════════════════════════════
function onSpeedRange(el) {
  const v = parseFloat(el.value)
  updateSliderBg(el)
  set('spd-disp', v.toFixed(1))
  send({ type: 'speed_limit', speed: v })
}

function setSpeed(v) {
  const el = document.getElementById('spd-range')
  el.value = v
  updateSliderBg(el)
  set('spd-disp', v.toFixed(1))
  send({ type: 'speed_limit', speed: v })
}

// ═══════════════════════════════════════════════
// MACROSTATE COMMANDS
// ═══════════════════════════════════════════════
function sendMacrostate(state) {
  send({ type: 'macrostate_command', state })
}

function sendMacrostateSub(cmdId) {
  send({ type: 'macrostate_sub_command', state: cmdId })
}

function stopMacrostate() {
  send({ type: 'macrostate_command', state: 'Stop' })
}

// ═══════════════════════════════════════════════
// DRIVE CONTROLS
// ═══════════════════════════════════════════════
const WASD_DRIVE_BTNS = {
  w: 'drive-fwd',
  s: 'drive-rev',
  a: 'drive-left',
  d: 'drive-right',
}

function startDrive(key, btnId) {
  if (!isManual) return
  document.getElementById(btnId)?.classList.add('lit')
  send({ type: 'key_command', key, state: 'press' })
}

function stopDrive(key, btnId) {
  document.getElementById(btnId)?.classList.remove('lit')
  send({ type: 'key_command', key, state: 'release' })
}

function sendPose(p, el) {
  send({ type: 'pose_command', pose: p })
}

// ── WASD ──────────────────────────────────────────────────────
function wasdPress(k) {
  if (!isManual) return
  const keyEl = document.getElementById('key-' + k)
  if (keyEl) keyEl.classList.add('pressed')
  const btnId = WASD_DRIVE_BTNS[k]
  if (btnId) document.getElementById(btnId)?.classList.add('lit')
  send({ type: 'key_command', key: k, state: 'press' })
}

function wasdRelease(k) {
  const keyEl = document.getElementById('key-' + k)
  if (keyEl) keyEl.classList.remove('pressed')
  const btnId = WASD_DRIVE_BTNS[k]
  if (btnId) document.getElementById(btnId)?.classList.remove('lit')
  send({ type: 'key_command', key: k, state: 'release' })
}

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase()
  if (k in WASD_DRIVE_BTNS && !e.repeat) wasdPress(k)
})
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase()
  if (k in WASD_DRIVE_BTNS) wasdRelease(k)
})

// ═══════════════════════════════════════════════
// E-STOP
// ═══════════════════════════════════════════════
// E-stop fires on PRESS (pointerdown), not on release: a stop shouldn't wait for the click
// to complete, and the button's text used to be rewritten 10x/s, which made Chrome drop clicks.
let _estopPressedAt = 0
function emergencyStop() {
  _estopPressedAt = Date.now()
  send({ type: 'estop' })
}

function resetEStop() {
  // Ignore the release of the same press that triggered the stop.
  if (Date.now() - _estopPressedAt < 800) return
  send({ type: 'estop_reset' })
}

// ═══════════════════════════════════════════════
// MODE TOGGLE
// ═══════════════════════════════════════════════
function toggleMode() {
  send({ type: 'toggle_mode' })
}

function _applyModeUI() {
  document.getElementById('mode-badge').textContent = isManual ? 'MANUAL' : 'AUTO'
  document.getElementById('mode-badge').className   = isManual ? 'mode-badge manual' : 'mode-badge'
  document.getElementById('mode-btn').textContent   = isManual ? '⇄ SWITCH TO AUTO' : '⇄ SWITCH TO MANUAL'
  document.getElementById('mode-btn').className     = isManual ? 'modetog m' : 'modetog'
  document.getElementById('ctrl-tag').textContent   = isManual ? 'MODE: MANUAL — INPUTS ACTIVE' : 'MODE: AUTO — INPUTS LOCKED'
  applyModeToControls()
}

function applyModeToControls() {
  _updateGoToBtn()
  const locked = !isManual

  document.querySelector('.ctrl-top')    ?.classList.toggle('locked', locked)
  document.querySelector('.macrostate-g')?.classList.toggle('locked', locked)
  document.querySelector('.speed-wrap')  ?.classList.toggle('locked', locked)

  const selectors = [
    '#pctrl button:not(#estop-btn):not(#mode-btn)',
    '#pctrl input[type=range]',
    '.macrostate-g button',
  ]
  document.querySelectorAll(selectors.join(',')).forEach(el => {
    el.disabled = locked
  })
}

// ═══════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════
function set(id, v) { const e = document.getElementById(id); if (e) e.textContent = v }
function fmt(v, d)  { return v != null ? Number(v).toFixed(d) : '—' }
function deg(r)     { return r != null ? (r * 180 / Math.PI).toFixed(1) : '—' }
function flash(el, duration=1200) {
  if (!el) return
  el.classList.add('lit')
  setTimeout(() => el.classList.remove('lit'), duration)
}

// ═══════════════════════════════════════════════
// HEARTBEAT
// ═══════════════════════════════════════════════
// Two sources, one rate limit: the 500 ms timer (works while the tab is visible) and a reply to
// every incoming 'state' message (works while it's hidden, see handle()). The robot's watchdog
// still E-stops if the page closes, freezes, or loses its connection: both sources stop then.
function _beat() {
  const now = performance.now()
  if (now - _lastBeatMs < 200) return
  _lastBeatMs = now
  send({ type: 'heartbeat' })
}

function _startHeartbeat() {
  _stopHeartbeat()
  _heartbeatTimer = setInterval(_beat, 500)
}

function _stopHeartbeat() {
  if (_heartbeatTimer) { clearInterval(_heartbeatTimer); _heartbeatTimer = null }
}

// Touch the button only when the E-stop state changes (rewriting it every 100 ms ate clicks).
let _estopShown = null
function _applyEstopUI(estopped) {
  const b = document.getElementById('estop-btn')
  if (!b || estopped === _estopShown) return
  _estopShown = estopped
  if (estopped) {
    b.textContent      = '⬡ STOPPED — CLICK TO RESET'
    b.style.background = 'rgba(255,34,51,.35)'
    b.onpointerdown    = null
    b.onclick          = resetEStop
  } else {
    b.textContent      = '⬡ EMERGENCY STOP'
    b.style.background = ''
    b.onpointerdown    = emergencyStop
    b.onclick          = null
  }
}

// ═══════════════════════════════════════════════
// CAMERA — MJPEG via <img> (dedicated HTTP connection,
// completely separate from the WebSocket)
// ═══════════════════════════════════════════════
function _initCamera() {
  const container = document.getElementById('cam-ph')?.parentElement
  if (!container) return

  // ── <img> element ──────────────────────────────────────────
  const img = document.createElement('img')
  img.style.cssText = 'width:100%;height:100%;object-fit:contain;display:block;'
  img.alt = ''
  container.appendChild(img)
  document.getElementById('cam-ph').style.display = 'none'

  // ── SIGNAL LOST overlay ────────────────────────────────────
  const overlay = document.createElement('div')
  overlay.style.cssText = `
    position:absolute;inset:0;display:none;flex-direction:column;
    align-items:center;justify-content:center;
    background:rgba(5,8,12,0.82);color:#00d4ff;
    font-family:'Courier New',monospace;font-size:12px;
    letter-spacing:0.08em;gap:8px;z-index:10;
  `
  overlay.innerHTML = `
    <div style="font-size:22px;opacity:0.6">⬡</div>
    <div>SIGNAL LOST</div>
    <div id="cam-stale-timer" style="opacity:0.5;font-size:10px">— s</div>
  `
  // container.appendChild(overlay)

  // ── Header status dot ──────────────────────────────────────
  const camDot = document.getElementById('cam-dot')

  // ── Watchdog state ─────────────────────────────────────────
  let lastLoad = Date.now()
  let stalled  = false
  let reloading = false   // prevent overlapping reload attempts

  // ── Callbacks ──────────────────────────────────────────────
  img.onload = () => {
    lastLoad  = Date.now()
    stalled   = false
    reloading = false
    overlay.style.display = 'none'
    if (camDot) camDot.className = 'dot'        // green — live
  }

  img.onerror = () => {
    if (camDot) camDot.className = 'dot warn'   // amber — connection error
    reloading = false
    _reloadCamImg(img)
  }

  // ── Stall watchdog — checks every second ───────────────────
  // If no onload has fired in CAM_TIMEOUT_MS, show overlay and
  // force a fresh HTTP request by replacing the src with a cache-buster.
  setInterval(() => {
    const elapsed = Date.now() - Math.max(lastLoad, _lastVisionMs)
    if (elapsed > CAM_TIMEOUT_MS) {
      if (!stalled) {
        stalled = true
        overlay.style.display = 'flex'
        if (camDot) camDot.className = 'dot warn'
      }
      const el = document.getElementById('cam-stale-timer')
      if (el) el.textContent = `${Math.floor(elapsed / 1000)}s ago`

      // Prod the src — opens a brand-new TCP connection to /camera
      if (!reloading) _reloadCamImg(img)
    } else {
      stalled = false
      if (camDot) camDot.className = 'dot'      // green — frames flowing
    }
  }, 1000)

  // ── Initial load ───────────────────────────────────────────
  _reloadCamImg(img)
}

function _reloadCamImg(img) {
  // Cache-buster prevents the browser reusing a cached error response.
  // Same host as the page and the WebSocket in connect() above.
  const base = `${location.origin}/camera`
  img.src = base + '?t=' + Date.now()
}

// ═══════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════
buildServos()
buildAudio()
updateSliderBg(document.getElementById('spd-range'))
_applyModeUI()
_initCamera()    // replaces the old _initCamWatchdog()
connect()
requestAnimationFrame(initMap)

// ═══════════════════════════════════════════════
// MAP — 2D Top-Down Canvas
// ═══════════════════════════════════════════════
function initMap() {
  const canvas = document.getElementById('map-canvas')
  if (!canvas) return

  function resize() {
    const dpr  = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width  = Math.round(rect.width  * dpr)
    canvas.height = Math.round(rect.height * dpr)
    canvas._cssW  = rect.width
    canvas._cssH  = rect.height
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0)
  }
  resize()
  new ResizeObserver(resize).observe(canvas)

  canvas.addEventListener('wheel', e => {
    e.preventDefault()
    mapZoom = Math.min(6.0, Math.max(1.0, mapZoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)))
    _clampPan()
  }, { passive: false })

  let _mouseDownPos = { x: 0, y: 0 }

  canvas.addEventListener('mousedown', e => {
    _dragActive   = true
    _dragLast     = { x: e.clientX, y: e.clientY }
    _mouseDownPos = { x: e.clientX, y: e.clientY }
    canvas.style.cursor = 'grabbing'
  })
  window.addEventListener('mousemove', e => {
    if (!_dragActive) return
    mapPanX -= (e.clientX - _dragLast.x) / _lastS
    mapPanY -= (e.clientY - _dragLast.y) / _lastS
    _dragLast = { x: e.clientX, y: e.clientY }
    _clampPan()
  })
  window.addEventListener('mouseup', e => {
    _dragActive = false
    canvas.style.cursor = 'crosshair'

    const dx = e.clientX - _mouseDownPos.x
    const dy = e.clientY - _mouseDownPos.y
    if (Math.sqrt(dx*dx + dy*dy) > 5) return

    const rect = canvas.getBoundingClientRect()
    const cx   = e.clientX - rect.left
    const cy   = e.clientY - rect.top
    const wx   = (cx - (rect.width  / 2 - MAP_W * _lastS / 2) + mapPanX * _lastS) / _lastS
    const wy   = (cy - (rect.height / 2 - MAP_H * _lastS / 2) + mapPanY * _lastS) / _lastS

    const HIT_R = 2.5
    let best = null, bestDist = HIT_R
    mapWaypoints.forEach(wp => {
      const d = Math.sqrt((wp.x - wx)**2 + (wp.y - wy)**2)
      if (d < bestDist) { bestDist = d; best = wp }
    })

    if (best && isManual) {
      selectedWaypoint = (selectedWaypoint?.label === best.label) ? null : best
      _updateGoToBtn()
    }
  })

  ;(function loop() { drawMap(canvas); requestAnimationFrame(loop) })()
}

function _updateGoToBtn() {
  const btn = document.getElementById('goto-btn')
  if (!btn) return
  if (selectedWaypoint && isManual) {
    btn.textContent = `⊕ Go To ${selectedWaypoint.label}`
    btn.disabled    = false
    btn.className   = 'btn ac'
  } else if (selectedWaypoint) {
    btn.textContent = `${selectedWaypoint.label} selected`
    btn.disabled    = true
    btn.className   = 'btn'
  } else {
    btn.textContent = 'Click waypoint'
    btn.disabled    = true
    btn.className   = 'btn'
  }
}

function sendWaypointMove() {
  if (!selectedWaypoint || !isManual) return
  send({ type: 'waypoint_move', x: selectedWaypoint.x, y: selectedWaypoint.y, label: selectedWaypoint.label })
  selectedWaypoint = null
  _updateGoToBtn()
}

function _clampPan() {
  const maxPanX = MAP_W - MAP_W / mapZoom
  const maxPanY = MAP_H - MAP_H / mapZoom
  mapPanX = Math.max(-MAP_W*2 / mapZoom, Math.min(maxPanX + MAP_W / mapZoom, mapPanX))
  mapPanY = Math.max(-MAP_H*2 / mapZoom, Math.min(maxPanY + MAP_H / mapZoom, mapPanY))
}

function drawMap(canvas) {
  const ctx = canvas.getContext('2d')
  const cw  = canvas._cssW || canvas.offsetWidth
  const ch  = canvas._cssH || canvas.offsetHeight
  if (!cw || !ch) return

  const t      = Date.now() / 1000
  const margin = 14
  const S      = Math.min((cw - margin * 2) / MAP_W, (ch - margin * 2) / MAP_H) * mapZoom
  _lastS       = S
  const OX     = (cw - MAP_W * S) / 2 - mapPanX * S
  const OY     = (ch - MAP_H * S) / 2 - mapPanY * S

  function proj(wx, wy) { return [OX + wx * S, OY + wy * S] }

  ctx.save()
  ctx.shadowBlur = 0
  ctx.setLineDash([])
  ctx.clearRect(0, 0, cw, ch)
  ctx.fillStyle = '#05080c'
  ctx.fillRect(0, 0, cw, ch)

  const [fx0, fy0] = proj(0, 0)
  const fw = MAP_W * S, fh = MAP_H * S
  ctx.fillStyle = '#080e17'
  ctx.fillRect(fx0, fy0, fw, fh)

  // Grid
  ctx.lineWidth   = 0.5
  ctx.strokeStyle = 'rgba(0,180,220,0.13)'
  for (let gx = 0; gx <= MAP_W; gx += 10) {
    const x = OX + gx * S
    ctx.beginPath(); ctx.moveTo(x, OY); ctx.lineTo(x, OY + fh); ctx.stroke()
  }
  for (let gy = 0; gy <= MAP_H; gy += 10) {
    const y = OY + gy * S
    ctx.beginPath(); ctx.moveTo(OX, y); ctx.lineTo(OX + fw, y); ctx.stroke()
  }

  // Walls and border are drawn steady (no sweep line, no pulsing glow).
  const wallPulse = 1

  // Obstacles
  mapObstacles.filter(o => o.type === 'line').forEach(ob => {
    const [ax, ay] = proj(ob.x1, ob.y1)
    const [bx, by] = proj(ob.x2, ob.y2)
    ctx.strokeStyle = `rgba(0,200,240,${wallPulse})`
    ctx.lineWidth   = 2
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 6
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
    ctx.shadowBlur  = 0
  })

  mapObstacles.filter(o => o.type === 'wall').forEach(ob => {
    const [tx, ty] = proj(ob.x, ob.y)
    ctx.fillStyle   = `rgba(0,70,110,${0.45 * wallPulse})`
    ctx.fillRect(tx, ty, S, S)
    ctx.strokeStyle = `rgba(0,200,240,${wallPulse})`
    ctx.lineWidth   = 1
    ctx.strokeRect(tx, ty, S, S)
  })

  mapObstacles.filter(o => o.type === 'point').forEach(ob => {
    const [px, py] = proj(ob.x, ob.y)
    ctx.fillStyle   = '#0d1520'
    ctx.strokeStyle = '#2a6080'; ctx.lineWidth = 1
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 5
    ctx.beginPath(); ctx.rect(px - 5, py - 5, 10, 10)
    ctx.fill(); ctx.stroke()
    ctx.shadowBlur  = 0
    ctx.fillStyle   = '#00d4ff'
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.fillText(String(ob.label ?? 'OBS'), px, py - 8)
  })

  // Border
  ctx.strokeStyle = `rgba(0,212,255,${wallPulse})`
  ctx.lineWidth   = 1.5
  ctx.strokeRect(fx0, fy0, fw, fh)

  // Detection boxes
  function drawDetectionBox(wx, wy, color, label, boxW, boxH, idx) {
    const [cx, cy] = proj(wx, wy)
    const bx = cx - (boxW * S) / 2
    const by = cy - (boxH * S) / 2
    const bw    = boxW * S, bh = boxH * S
    const pulse = 0.65 + 0.35 * Math.sin(t * 3.5 + idx * 1.7)
    const col      = color.replace('ALPHA', String(pulse))
    const colFill  = color.replace('ALPHA', String(0.09 * pulse))
    ctx.fillStyle   = colFill;                    ctx.fillRect(bx, by, bw, bh)
    ctx.strokeStyle = col; ctx.lineWidth = 1.2;   ctx.strokeRect(bx, by, bw, bh)
    ctx.fillStyle   = col
    ctx.font        = 'bold 9px Courier New'
    ctx.textAlign   = 'center'
    ctx.fillText(label, bx + bw / 2, by - 4)
  }

  mapTurkeys.forEach((tk, i) =>
    drawDetectionBox(tk[0], tk[1], 'rgba(255,120,0,ALPHA)', `TURKEY-${i+1}`, 2.0, 2.0, i))

  mapHumans.forEach((hm, i) =>
    drawDetectionBox(hm[0], hm[1], 'rgba(255,30,60,ALPHA)', `HUMAN-${i+1}`, 1.5, 1.5, i + 100))

  // Waypoints
  mapWaypoints.forEach(wp => {
    const [px, py]   = proj(wp.x, wp.y)
    const isActive   = mapRobot && Math.abs(wp.x - (mapRobot.targetX ?? -999)) < 0.5
                                 && Math.abs(wp.y - (mapRobot.targetY ?? -999)) < 0.5
    const isSelected = selectedWaypoint?.label === wp.label
    const col  = isActive   ? 'rgba(255,200,0,0.95)'
               : isSelected ? 'rgba(0,212,255,0.95)'
               :               'rgba(255,200,0,0.55)'
    const glow = (isActive || isSelected) ? 10 : 4
    const r    = (isActive || isSelected) ? 7  : 5

    if (isSelected) {
      ctx.strokeStyle = 'rgba(0,212,255,0.6)'
      ctx.lineWidth   = 1
      ctx.shadowColor = 'rgba(0,212,255,1)'; ctx.shadowBlur = 12
      ctx.beginPath(); ctx.arc(px, py, r + 6, 0, Math.PI * 2); ctx.stroke()
      ctx.shadowBlur  = 0
    }

    ctx.save()
    ctx.translate(px, py)
    ctx.rotate(Math.PI / 4)
    ctx.fillStyle   = isActive ? 'rgba(255,200,0,0.25)' : 'rgba(255,200,0,0.1)'
    ctx.strokeStyle = col
    ctx.lineWidth   = isActive ? 2 : 1.5
    ctx.shadowColor = 'rgba(255,200,0,1)'; ctx.shadowBlur = glow
    ctx.beginPath(); ctx.rect(-r, -r, r * 2, r * 2)
    ctx.fill(); ctx.stroke()
    ctx.shadowBlur = 0
    ctx.restore()

    ctx.fillStyle   = col
    ctx.font        = `${isActive ? 'bold ' : ''}8px Courier New`
    ctx.textAlign   = 'center'
    ctx.shadowColor = 'rgba(255,200,0,1)'; ctx.shadowBlur = glow
    ctx.fillText(wp.label, px, py - r - 5)
    ctx.shadowBlur  = 0
  })

  // ─── REFERENCE TRAJECTORY (Protocol 6.1) ─────────────────────
  if (mapReferenceTraj.length > 1) {
    ctx.save()
    ctx.beginPath()
    const [rsx, rsy] = proj(mapReferenceTraj[0][0], mapReferenceTraj[0][1])
    ctx.moveTo(rsx, rsy)
    mapReferenceTraj.slice(1).forEach(pt => {
      const [rx, ry] = proj(pt[0], pt[1])
      ctx.lineTo(rx, ry)
    })
    ctx.strokeStyle = 'rgba(0,212,255,0.3)'
    ctx.lineWidth   = 3  // 0.8
    ctx.setLineDash([4, 6])
    ctx.stroke()
    ctx.restore()
    ctx.setLineDash([])

    mapReferenceTraj.forEach((pt, i) => {
      const [px, py] = proj(pt[0], pt[1])
      ctx.beginPath()
      ctx.arc(px, py, 1.5, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0,212,255,0.3)'
      ctx.fill()
    })
  }

  // ─── MPC HORIZON (Protocol 6.2) ──────────────────────────────
  if (mapMPCHorizon.length > 1) {
    const [robX, robY] = proj(mapRobot.x, mapRobot.y)

    ctx.save()
    ctx.shadowColor = 'rgba(255,160,0,0.45)'
    ctx.shadowBlur  = 8
    ctx.beginPath()
    ctx.moveTo(robX, robY)
    mapMPCHorizon.forEach(pt => {
      const [mx, my] = proj(pt[0], pt[1])
      ctx.lineTo(mx, my)
    })
    ctx.strokeStyle = 'rgba(255,140,0,0.22)'
    ctx.lineWidth   = 5
    ctx.stroke()
    ctx.restore()

    for (let i = 0; i < mapMPCHorizon.length; i++) {
      const [ax, ay] = i === 0
        ? [robX, robY]
        : proj(mapMPCHorizon[i - 1][0], mapMPCHorizon[i - 1][1])
      const [bx, by] = proj(mapMPCHorizon[i][0], mapMPCHorizon[i][1])
      const alpha     = 1.0 - (i / mapMPCHorizon.length) * 0.55
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.strokeStyle = `rgba(255,160,20,${alpha * 0.85})`
      ctx.lineWidth   = Math.max(1, 2 - i * 0.12)
      ctx.stroke()
    }

    mapMPCHorizon.forEach((pt, i) => {
      const [mx, my] = proj(pt[0], pt[1])
      const alpha     = 1.0 - (i / mapMPCHorizon.length) * 0.7
      ctx.beginPath()
      ctx.arc(mx, my, Math.max(1.5, 2.5 - i * 0.15), 0, Math.PI * 2)
      ctx.fillStyle = `rgba(255,180,40,${alpha})`
      ctx.fill()
    })

    const tip      = mapMPCHorizon[mapMPCHorizon.length - 1]
    const [tx, ty] = proj(tip[0], tip[1])
    const pulse    = 0.5 + 0.5 * Math.sin(t * 6)
    ctx.save()
    ctx.shadowColor = 'rgba(255,200,0,1)'
    ctx.shadowBlur  = 12 * pulse
    ctx.beginPath()
    ctx.arc(tx, ty, 4 + pulse * 2, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(255,220,60,${0.8 + 0.2 * pulse})`
    ctx.fill()
    ctx.restore()

    ctx.fillStyle = 'rgba(255,180,40,0.7)'
    ctx.font      = '7px Courier New'
    ctx.textAlign = 'left'
    ctx.fillText('MPC', tx + 6, ty + 3)
  }

  // ─── ROBOT ───────────────────────────────────────────────────
  const [rpx, rpy] = proj(mapRobot.x, mapRobot.y)
  const rth        = mapRobot.theta ?? 0
  const rthDraw    = -rth
  const beamTipX   = rpx + Math.cos(rthDraw) * Math.max(30, S * 5)
  const beamTipY   = rpy + Math.sin(rthDraw) * Math.max(30, S * 5)
  const r          = Math.max(7, Math.min(13, S * 0.7))

  const haloGrd = ctx.createRadialGradient(rpx, rpy, 0, rpx, rpy, Math.max(18, S * 1.2))
  haloGrd.addColorStop(0, 'rgba(0,212,255,0.22)')
  haloGrd.addColorStop(1, 'rgba(0,212,255,0)')
  ctx.fillStyle = haloGrd
  ctx.beginPath(); ctx.arc(rpx, rpy, Math.max(18, S * 1.2), 0, Math.PI * 2); ctx.fill()

  const beamGrd = ctx.createLinearGradient(rpx, rpy, beamTipX, beamTipY)
  beamGrd.addColorStop(0, 'rgba(0,212,255,0.65)')
  beamGrd.addColorStop(1, 'rgba(0,212,255,0)')
  ctx.strokeStyle = beamGrd; ctx.lineWidth = 2
  ctx.beginPath(); ctx.moveTo(rpx, rpy); ctx.lineTo(beamTipX, beamTipY); ctx.stroke()

  ctx.save()
  ctx.translate(rpx, rpy); ctx.rotate(rthDraw)
  ctx.strokeStyle = '#00d4ff'; ctx.fillStyle = 'rgba(0,212,255,0.22)'
  ctx.lineWidth   = 2; ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 10
  ctx.beginPath()
  ctx.moveTo(r, 0); ctx.lineTo(-r * 0.6,  r * 0.75)
  ctx.lineTo(-r * 0.25, 0); ctx.lineTo(-r * 0.6, -r * 0.75)
  ctx.closePath(); ctx.fill(); ctx.stroke()
  ctx.shadowBlur = 0; ctx.restore()

  ctx.fillStyle = '#00d4ff'; ctx.font = 'bold 8px Courier New'; ctx.textAlign = 'center'
  ctx.fillText('ROBOT', rpx, rpy - r - 6)

  // Axis labels
  ctx.font = '7px Courier New'; ctx.fillStyle = 'rgba(0,180,220,0.3)'
  ctx.textAlign = 'center'
  for (let gx = 0; gx <= MAP_W; gx += 20)
    ctx.fillText(`${gx}`, OX + gx * S, OY + fh + 10)
  ctx.textAlign = 'right'
  for (let gy = 0; gy <= MAP_H; gy += 20)
    ctx.fillText(`${gy}`, OX - 4, OY + gy * S + 3)

  // Legend
  const legend = [
    ['rgba(255,120,0,0.9)',  '▣', 'Turkey'],
    ['rgba(255,30,60,0.9)',  '▣', 'Human'],
    ['rgba(0,212,255,0.9)',  '▶', 'Robot'],
    ['rgba(255,200,0,0.8)',  '◆', 'Waypoint'],
    ['rgba(0,212,255,0.65)', '╌', 'Ref Path'],
    ['rgba(255,160,20,0.85)','—', 'MPC Horizon'],
    ['rgba(0,212,255,0.45)', '—', 'Wall'],
  ]
  legend.forEach(([col, sym, lbl], i) => {
    const ly = 12 + i * 13
    ctx.font = '9px Courier New'; ctx.textAlign = 'left'
    ctx.fillStyle = col;                     ctx.fillText(sym,        8, ly)
    ctx.fillStyle = 'rgba(100,140,160,0.7)'; ctx.fillText(` ${lbl}`, 19, ly)
  })

  ctx.font = '8px Courier New'; ctx.textAlign = 'right'
  ctx.fillStyle = 'rgba(0,180,220,0.4)'
  ctx.fillText(mapZoom > 1.01
    ? `${mapZoom.toFixed(1)}×  scroll to zoom · drag to pan`
    : `scroll to zoom`, cw - 8, ch - 8)

  ctx.restore()
}