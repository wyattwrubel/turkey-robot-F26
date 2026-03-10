// ═══════════════════════════════════════════════
// CONFIG — swap in real values when ready
// ═══════════════════════════════════════════════
const SERVO_DEFS = [
  { id: 'FrontLeft', name: 'Front Left', min: 0,   max: 180, val: 90  },
  { id: 'FrontRight', name: 'Front Right', min: 0,   max: 180, val: 90  },
  { id: 'RearLeft', name: 'Rear Left', min: 0,   max: 270, val: 135 },
  { id: 'RearRight', name: 'Rear Right', min: -90, max: 90,  val: 0   },
  { id: 'Head', name: 'Head', min: 0,   max: 180, val: 90  },
]

const AUDIO_DEFS = [
  { id: 'BARK',    label: '🔊 Bark'    },
  { id: 'ALARM',   label: '🔊 Alarm'   },
  { id: 'WHISTLE', label: '🔊 Whistle' },
  { id: 'HOWL',    label: '🔊 Howl'    },
]

const SPEED_MAX = 3.0   // m/s — update when you have real value

// ── Static map CSV ──────────────────────────────────────────────────────────
// Set to '/map.csv' to auto-load your occupancy grid on page open.
// Drop your CSV file in the Website/ folder and add the /map.csv route in
// app.py (already done — see the static routes section).
// Set to '' to disable auto-load and use the manual ⊕ CSV button instead.
const MAP_CSV = '/map.csv'

// WASD direction vectors (normalized, scaled by speed at send time)
const WASD_DIRS = {
  w: [ 1,  1],
  s: [-1, -1],
  a: [-1,  1],
  d: [ 1, -1],
}

// ═══════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════
let ws        = null
let wsReady   = false
let isManual  = false
let isStopped = false
let speed     = 1.5    // m/s

// ── Map / Vision state ──────────────────────────────────────
let mapTurkeys   = []   // [[x, y], ...] live from VisionObject — position only
let mapHumans    = []   // [[x, y], ...] live from VisionObject — position only
let mapObstacles = []   // [[x, y, label], ...] loaded from CSV
let mapRobot     = { x: 0, y: 0, theta: 0 }  // from state broadcast


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
    btn.dataset.audioId = a.id        // used by active_audio highlight
    btn.onclick         = () => playAudio(a.id, btn)
    grid.appendChild(btn)
  })
}

// ═══════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════
function connect() {
  // ws = new WebSocket('ws://raspberrypi.local:8000/ws')
  ws = new WebSocket('ws://localhost:8000/ws')
  // ws = new WebSocket('ws://192.168.4.2:8000/ws')
  ws.onopen    = ()  => { wsReady = true }
  ws.onclose   = ()  => { wsReady = false; setTimeout(connect, 1500) }
  ws.onmessage = (e) => {
    try {
      handle(JSON.parse(e.data))
    } catch (err) {
      console.error('handle() error:', err)   // visible in browser F12 console
    }
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

    // sensors
    set('uwb-x',   fmt(m.uwb_x, 2))
    set('uwb-y',   fmt(m.uwb_y, 2))
    set('uwb-age', `last fix: ${fmt(m.uwb_age, 1)}s ago`)
    set('imu-t',   deg(m.imu_theta))
    set('imu-cal', `cal: sys=${m.cal_sys ?? '—'} gyro=${m.cal_gyro ?? '—'} mag=${m.cal_mag ?? '—'}`)

    // UKF
    set('ukf-x', fmt(m.x, 2))
    set('ukf-y', fmt(m.y, 2))
    set('ukf-t', deg(m.theta))
    if (m.x != null) mapRobot = { x: m.x, y: m.y, theta: m.theta ?? 0 }

    // Confidence indicator — collapse sigma_x, sigma_y into a single position uncertainty
    if (m.sigma_x != null && m.sigma_y != null) {
      const sigmaPos = Math.max(m.sigma_x, m.sigma_y)
      set('conf-val', sigmaPos.toFixed(3))
      const dot   = document.getElementById('conf-dot')
      const label = document.getElementById('conf-label')
      if (sigmaPos < 0.08) {
        dot.className = 'conf-dot good';  label.textContent = 'HIGH'
      } else if (sigmaPos < 0.18) {
        dot.className = 'conf-dot warn';  label.textContent = 'MED'
      } else {
        dot.className = 'conf-dot poor';  label.textContent = 'LOW'
      }
    }

    // waypoint
    set('wp-id', m.wp_id ?? 'WP — / —')
    set('wp-i',  m.wp_dist != null ? `dist: ${fmt(m.wp_dist, 1)} m` : 'dist: — m')

    // drivetrain
    set('cmd-l', fmt(m.cmd_left, 2))
    set('cmd-r', fmt(m.cmd_right, 2))
    {
      const src     = m.drive_source ?? null
      const srcChip = document.getElementById('drv-src-chip')
      if (src === 'UI') {
        srcChip.textContent = 'UI';     srcChip.className = 'chip ac'
      } else if (src === 'System') {
        srcChip.textContent = 'SYS';   srcChip.className = 'chip ok'
      } else {
        srcChip.textContent = '—';     srcChip.className = 'chip'
      }
      set('drv-src', `source: ${src ?? '—'}`)
    }

    // system state
    {
      const ss    = m.sys_state ?? null
      const ssEl  = document.getElementById('sys-state')
      if (ss === 'Chase') {
        ssEl.textContent = 'CHASE';    ssEl.className = 'v d'
      } else if (ss === 'Alert') {
        ssEl.textContent = 'ALERT';    ssEl.className = 'v d'
      } else if (ss === 'Scanning') {
        ssEl.textContent = 'SCANNING'; ssEl.className = 'v w'
      } else {
        ssEl.textContent = '—';        ssEl.className = 'v'
      }
      set('sys-dest', m.destination != null
        ? `destination: ${fmt(m.destination, 2)} m`
        : 'destination: —')
    }

    // navigation command
    {
      const nc = m.nav_cmd ?? null
      if (nc && nc.type === 'Move') {
        set('nav-cmd-type',   'MOVE')
        set('nav-cmd-detail', `target: (${fmt(nc.x,2)}, ${fmt(nc.y,2)}) m`)
      } else if (nc && nc.type === 'Orient') {
        set('nav-cmd-type',   'ORIENT')
        set('nav-cmd-detail', `target: (${fmt(nc.x,2)}, ${fmt(nc.y,2)}) m  θ=${deg(nc.theta)}°`)
      } else {
        set('nav-cmd-type',   '—')
        set('nav-cmd-detail', '—')
      }
    }

    // speed slider sync
    if (m.speed_ms != null) {
      speed = m.speed_ms
      const sl = document.getElementById('spd-range')
      sl.value = speed
      updateSliderBg(sl)
      set('spd-disp', speed.toFixed(1))
    }

    // drivetrain button highlight (not WASD)
    if (m.cmd_left != null && m.cmd_right != null) {
      reflectDriveOnButtons(m.cmd_left, m.cmd_right)
    }

    // servo sliders — variable named 'sv' to avoid collision with detStatus 's'
    if (Array.isArray(m.servos)) {
      m.servos.forEach(sv => {
        const sl = document.getElementById(`sr-${sv.id}`)
        const vl = document.getElementById(`sv-${sv.id}`)
        if (sl) { sl.value = sv.angle; updateSliderBg(sl) }
        if (vl) vl.textContent = `${Number(sv.angle).toFixed(1)}°`
      })
    }

    // pose button highlight — works for both UI ('Pose1') and system ('Pose1') sources
    if (m.active_pose !== undefined) {
      document.querySelectorAll('.pose-g .btn').forEach(b => b.classList.remove('lit'))
      if (m.active_pose) {
        document.querySelectorAll('.pose-g .btn').forEach(b => {
          if (b.dataset.poseId === m.active_pose)
            b.classList.add('lit')
        })
      }
    }

    // audio button highlight
    if (m.active_audio !== undefined) {
      document.querySelectorAll('#audio-grid .btn').forEach(b => {
        b.classList.toggle('lit', b.dataset.audioId === m.active_audio)
      })
    }
  }

  // ── Camera frame (1 Hz) ────────────────────────────────────
  if (m.type === 'frame') {
    const img = document.getElementById('live-img')
    // Build data URI and load it; on success reveal the img and hide placeholder text
    const uri = 'data:image/jpeg;base64,' + m.image
    img.onload = () => {
      img.style.display = 'flex'   // flex child of feed-ph — fills available space
      document.querySelectorAll('#cam-ph .fi, #cam-ph .fl, #cam-ph .fs, #cam-ph .cr')
        .forEach(el => el.style.display = 'none')
    }
    img.onerror = () => {
      // Keep placeholder visible if image fails to decode
      img.style.display = 'none'
    }
    img.src = uri
  }

  // ── Servo state (on connect) ───────────────────────────────
  if (m.type === 'servo_state') {
    m.servos?.forEach(sv => {
      const sl = document.getElementById(`sr-${sv.id}`)
      const vl = document.getElementById(`sv-${sv.id}`)
      if (sl) { sl.value = sv.angle; updateSliderBg(sl) }
      if (vl) vl.textContent = `${Number(sv.angle).toFixed(1)}°`
    })
  }

  // ── Vision detections (VisionObject_Receiver → browser) ────
  if (m.type === 'vision') {
    mapTurkeys = m.turkeys ?? []
    mapHumans  = m.humans  ?? []
  }
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
  const min = parseFloat(el.min)
  const max = parseFloat(el.max)
  const val = parseFloat(el.value)
  const pct = ((val - min) / (max - min)) * 100
  el.style.setProperty('--pct', `${pct}%`)
}

// ═══════════════════════════════════════════════
// AUDIO CONTROLS
// ═══════════════════════════════════════════════
function playAudio(soundId, btn) {
  send({ type: 'audio_play', sound: soundId })
  btn.classList.add('audio-playing')
  btn.addEventListener('animationend', () => btn.classList.remove('audio-playing'), { once: true })
}

// ═══════════════════════════════════════════════
// SPEED CONTROL
// ═══════════════════════════════════════════════
function onSpeedRange(el) {
  speed = parseFloat(el.value)
  updateSliderBg(el)
  set('spd-disp', speed.toFixed(1))
  send({ type: 'speed_limit', speed })
}

function setSpeed(v) {
  speed = Math.min(v, SPEED_MAX)
  const el = document.getElementById('spd-range')
  el.value = speed
  updateSliderBg(el)
  set('spd-disp', speed.toFixed(1))
  send({ type: 'speed_limit', speed })
}

// ═══════════════════════════════════════════════
// DRIVE CONTROLS
// ═══════════════════════════════════════════════
function startDrive(l, r) {
  if (!isManual || isStopped) return
  const sc = speed / SPEED_MAX
  send({ type: 'motor_command', Id: 'UI', Cmd: [l * sc, r * sc], speed_ms: speed })
}

function stopDrive() {
  send({ type: 'motor_command', Id: 'UI', Cmd: [0, 0], speed_ms: 0 })
}

function sendPose(p, el) {
  if (!isManual) return
  send({ type: 'pose_command', pose: p })
  flash(el)
}

// ── WASD ───────────────────────────────────────────────────
function wasdPress(k) {
  if (!isManual || isStopped) return
  const el = document.getElementById(`key-${k}`)
  if (el) el.classList.add('pressed')
  const d = WASD_DIRS[k]
  if (!d) return
  const sc = speed / SPEED_MAX
  send({ type: 'motor_command', Id: 'UI', Cmd: [d[0] * sc, d[1] * sc], speed_ms: speed })
}

function wasdRelease(k) {
  const el = document.getElementById(`key-${k}`)
  if (el) el.classList.remove('pressed')
  send({ type: 'motor_command', Id: 'UI', Cmd: [0, 0], speed_ms: 0 })
}

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase()
  if (WASD_DIRS[k] && !e.repeat) wasdPress(k)
})
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase()
  if (WASD_DIRS[k]) wasdRelease(k)
})

// ── Drivetrain button highlight ────────────────────────────
// Reflects actual motor state from server onto the 4 drive buttons.
// WASD keys are NOT touched here — they only light up from user input.
function reflectDriveOnButtons(left, right) {
  const T = 0.05

  // W lights if both sides forward, S if both back
  // A/D light if sides are opposite (spin in place)
  // W+D = curving right, W+A = curving left, etc.
  const fwd   = left > T && right > T
  const rev   = left < -T && right < -T
  const turnL = left < -T && right > T
  const turnR = left > T && right < -T

  // curve right: left fast, right slow → W + D both lit
  const curveR = left > T && right > T && left > right + T
  const curveL = right > T && left > T && right > left + T

  document.getElementById('drive-fwd')  ?.classList.toggle('lit', fwd || curveR || curveL)
  document.getElementById('drive-rev')  ?.classList.toggle('lit', rev)
  document.getElementById('drive-left') ?.classList.toggle('lit', turnL || curveL)
  document.getElementById('drive-right')?.classList.toggle('lit', turnR || curveR)
}

// ═══════════════════════════════════════════════
// E-STOP
// ═══════════════════════════════════════════════
function emergencyStop() {
  isStopped = true
  send({ type: 'estop' })   // internal — handlers.py calls apply_estop_engage()
  const b = document.getElementById('estop-btn')
  b.textContent      = '⬡ STOPPED — CLICK TO RESET'
  b.style.background = 'rgba(255,34,51,.35)'
  b.onclick          = resetEStop
}

function resetEStop() {
  isStopped = false
  send({ type: 'estop_reset' })   // internal — handlers.py calls apply_estop_release()
  const b = document.getElementById('estop-btn')
  b.textContent      = '⬡ EMERGENCY STOP'
  b.style.background = ''
  b.onclick          = emergencyStop
}

// ═══════════════════════════════════════════════
// MODE TOGGLE
// ═══════════════════════════════════════════════
function toggleMode() {
  isManual = !isManual
  // Mode is local UI state only — no message sent to server
  document.getElementById('mode-badge').textContent = isManual ? 'MANUAL' : 'AUTO'
  document.getElementById('mode-badge').className   = isManual ? 'mode-badge manual' : 'mode-badge'
  document.getElementById('mode-btn').textContent   = isManual ? '⇄ SWITCH TO AUTO' : '⇄ SWITCH TO MANUAL'
  document.getElementById('mode-btn').className     = isManual ? 'modetog m' : 'modetog'
  document.getElementById('ctrl-tag').textContent   = isManual ? 'MODE: MANUAL — INPUTS ACTIVE' : 'MODE: AUTO — INPUTS LOCKED'
  if (!isManual) stopDrive()
}

// ═══════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════
function set(id, v) { const e = document.getElementById(id); if (e) e.textContent = v }
function fmt(v, d)  { return v != null ? Number(v).toFixed(d) : '—' }
function deg(r)     { return r != null ? (r * 180 / Math.PI).toFixed(1) : '—' }
function flash(el)  {
  if (!el) return
  el.classList.add('lit')
  setTimeout(() => el.classList.remove('lit'), 280)
}

// ═══════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════
buildServos()
buildAudio()
updateSliderBg(document.getElementById('spd-range'))
connect()
// Defer by one rAF so the flex layout completes its first paint before
// getBoundingClientRect() is called — otherwise it returns 0×0 and the
// canvas never draws.
requestAnimationFrame(initMap)
// ═══════════════════════════════════════════════
// MAP CANVAS  — 2D Top-Down Tron Arena
// ═══════════════════════════════════════════════
//
// Orthographic top-down projection — world x → screen right, world y → screen down.
// Field is centred in the panel with equal margins. Zoom and pan supported.
//
// All draw calls are plain Canvas2D — no libraries, Pi-safe at 60 fps.

let MAP_W = 100.0
let MAP_H = 100.0

// ── Zoom / pan state (modified by wheel + drag on the canvas) ────────────
let mapZoom    = 1.0          // 1.0 = fit-to-panel; max capped at 6×
let mapPanX    = 0.0          // pan offset in world metres (x)
let mapPanY    = 0.0          // pan offset in world metres (y)
let _dragActive = false
let _dragLast   = { x: 0, y: 0 }

// ── 2D top-down orthographic projection ─────────────────────────────────────
// No oblique tilt — world x→screen right, world y→screen down (y=0 at top)

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
    // Reset the DPR scale so every draw starts clean
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  resize()
  new ResizeObserver(resize).observe(canvas)

  // Auto-load the static map CSV if configured
  if (MAP_CSV) {
    fetch(MAP_CSV)
      .then(r => {
        if (!r.ok) throw new Error(`map.csv not found (${r.status})`)
        return r.text()
      })
      .then(text => parseCSVText(text))
      .catch(err => console.warn('MAP_CSV auto-load skipped:', err.message))
  }

  // ── Zoom: scroll wheel ──────────────────────────────────────────────────
  canvas.addEventListener('wheel', e => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    mapZoom = Math.min(6.0, Math.max(1.0, mapZoom * factor))
    // Clamp pan so the view never goes empty
    _clampPan()
  }, { passive: false })

  // ── Pan: mouse drag ──────────────────────────────────────────────────────
  canvas.addEventListener('mousedown', e => {
    _dragActive = true
    _dragLast   = { x: e.clientX, y: e.clientY }
    canvas.style.cursor = 'grabbing'
  })
  window.addEventListener('mousemove', e => {
    if (!_dragActive) return
    const dx = e.clientX - _dragLast.x
    const dy = e.clientY - _dragLast.y
    _dragLast = { x: e.clientX, y: e.clientY }
    // Convert screen px delta → world metres (inverse of projection x-axis)
    // proj x: OX + wx*S  →  dx px = dwx * S  →  dwx = dx / S
    const S = _lastS || 1
    mapPanX -= dx / S
    mapPanY -= dy / S
    _clampPan()
  })
  window.addEventListener('mouseup', () => {
    _dragActive = false
    canvas.style.cursor = 'crosshair'
  })

  // ── Touch pan (single finger) ────────────────────────────────────────────
  canvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      _dragActive = true
      _dragLast   = { x: e.touches[0].clientX, y: e.touches[0].clientY }
    }
  }, { passive: true })
  canvas.addEventListener('touchmove', e => {
    if (!_dragActive || e.touches.length !== 1) return
    const dx = e.touches[0].clientX - _dragLast.x
    const dy = e.touches[0].clientY - _dragLast.y
    _dragLast = { x: e.touches[0].clientX, y: e.touches[0].clientY }
    const S = _lastS || 1
    mapPanX -= dx / S
    mapPanY -= dy / S
    _clampPan()
  }, { passive: true })
  canvas.addEventListener('touchend', () => { _dragActive = false })

  ;(function loop() { drawMap(canvas); requestAnimationFrame(loop) })()
}

// How much of the world is visible at current zoom
function _visibleWorld() {
  return { hw: MAP_W / mapZoom, hh: MAP_H / mapZoom }
}
function _clampPan() {
  const { hw, hh } = _visibleWorld()
  mapPanX = Math.max(0, Math.min(MAP_W - hw, mapPanX))
  mapPanY = Math.max(0, Math.min(MAP_H - hh, mapPanY))
}
// Expose for drawMap to record the last computed S
let _lastS = 1

function drawMap(canvas) {
  const ctx = canvas.getContext('2d')
  const cw  = canvas._cssW || canvas.offsetWidth
  const ch  = canvas._cssH || canvas.offsetHeight
  if (!cw || !ch) return

  const t = Date.now() / 1000

  // ── Orthographic 2D projection ────────────────────────────────────────────
  // World: x → right, y → down (origin top-left of field)
  // Fit the entire field with equal margins on all sides
  const margin = 14
  const baseS  = Math.min((cw - margin * 2) / MAP_W, (ch - margin * 2) / MAP_H)
  const S      = baseS * mapZoom
  _lastS       = S

  // Centre the field in the panel, then offset by pan
  const OX = (cw - MAP_W * S) / 2 - mapPanX * S
  const OY = (ch - MAP_H * S) / 2 - mapPanY * S

  function proj(wx, wy) { return [OX + wx * S, OY + wy * S] }

  // ── Background ────────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, cw, ch)
  ctx.fillStyle = '#05080c'
  ctx.fillRect(0, 0, cw, ch)

  // Dark field floor
  const [fx0, fy0] = proj(0, 0)
  const fw = MAP_W * S, fh = MAP_H * S
  ctx.fillStyle = '#080e17'
  ctx.fillRect(fx0, fy0, fw, fh)

  // ── Grid ──────────────────────────────────────────────────────────────────
  const GRID_STEP = 10
  ctx.lineWidth = 0.5
  for (let gx = 0; gx <= MAP_W; gx += GRID_STEP) {
    const x = OX + gx * S
    ctx.strokeStyle = 'rgba(0,180,220,0.13)'
    ctx.beginPath(); ctx.moveTo(x, OY); ctx.lineTo(x, OY + fh); ctx.stroke()
  }
  for (let gy = 0; gy <= MAP_H; gy += GRID_STEP) {
    const y = OY + gy * S
    ctx.strokeStyle = 'rgba(0,180,220,0.13)'
    ctx.beginPath(); ctx.moveTo(OX, y); ctx.lineTo(OX + fw, y); ctx.stroke()
  }

  // ── Scanline sweep (horizontal, travels top→bottom) ───────────────────────
  const sweepWY   = ((t * 0.18) % 1.0) * MAP_H
  const sweepFade = Math.sin(t * 0.57 * Math.PI) * 0.5 + 0.5
  const sweepY    = OY + sweepWY * S
  const sweepGrd  = ctx.createLinearGradient(OX, sweepY, OX + fw, sweepY)
  sweepGrd.addColorStop(0,   'rgba(0,212,255,0)')
  sweepGrd.addColorStop(0.3, `rgba(0,212,255,${0.45 * sweepFade})`)
  sweepGrd.addColorStop(0.7, `rgba(0,212,255,${0.45 * sweepFade})`)
  sweepGrd.addColorStop(1,   'rgba(0,212,255,0)')
  ctx.strokeStyle = sweepGrd
  ctx.lineWidth   = 1.5
  ctx.beginPath(); ctx.moveTo(OX, sweepY); ctx.lineTo(OX + fw, sweepY); ctx.stroke()

  // ── Wall pulse (shared by tiles + perimeter) ──────────────────────────────
  const wallPulse  = 0.75 + 0.25 * Math.sin(t * 1.4)
  const WALL_COLOR = `rgba(0,212,255,${wallPulse})`
  const GLOW_COLOR = `rgba(0,212,255,${0.18 * wallPulse})`
  const TILE_COLOR = `rgba(0,200,240,${wallPulse})`
  const TILE_FILL  = `rgba(0,70,110,${0.45 * wallPulse})`

  // ── Occupancy wall tiles — neon filled squares ────────────────────────────
  mapObstacles.filter(o => o.type === 'wall').forEach(ob => {
    const tx = OX + ob.x * S
    const ty = OY + ob.y * S
    const ts = S   // 1 world unit = S px

    // Filled cell
    ctx.fillStyle = TILE_FILL
    ctx.fillRect(tx, ty, ts, ts)

    // Glow pass
    ctx.strokeStyle = `rgba(0,200,240,${0.25 * wallPulse})`
    ctx.lineWidth   = 4
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 8
    ctx.strokeRect(tx, ty, ts, ts)
    ctx.shadowBlur  = 0

    // Sharp neon outline
    ctx.strokeStyle = TILE_COLOR
    ctx.lineWidth   = 1
    ctx.strokeRect(tx, ty, ts, ts)
  })

  // ── Named point obstacles ─────────────────────────────────────────────────
  mapObstacles.filter(o => o.type === 'point').forEach(ob => {
    const [px, py] = proj(ob.x, ob.y)
    const r = 5
    ctx.fillStyle   = '#0d1520'
    ctx.strokeStyle = '#2a6080'; ctx.lineWidth = 1
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 5
    ctx.beginPath(); ctx.rect(px - r, py - r, r * 2, r * 2)
    ctx.fill(); ctx.stroke()
    ctx.shadowBlur  = 0
    ctx.fillStyle   = '#00d4ff'
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 3
    ctx.fillText(String(ob.label ?? 'OBS'), px, py - r - 3)
    ctx.shadowBlur  = 0
  })

  // ── Neon arena perimeter ──────────────────────────────────────────────────
  // Outer glow
  ctx.strokeStyle = GLOW_COLOR
  ctx.lineWidth   = 8
  ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 18
  ctx.strokeRect(fx0, fy0, fw, fh)
  ctx.shadowBlur  = 0
  // Sharp neon line
  ctx.strokeStyle = WALL_COLOR
  ctx.lineWidth   = 1.5
  ctx.strokeRect(fx0, fy0, fw, fh)

  // Corner accent brackets
  const BKT = Math.min(16, fw * 0.06)
  const corners2d = [[fx0,fy0],[fx0+fw,fy0],[fx0+fw,fy0+fh],[fx0,fy0+fh]]
  const dirs      = [[1,0,0,1],[-1,0,0,1],[-1,0,0,-1],[1,0,0,-1]]
  ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2
  ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 6
  corners2d.forEach(([cx,cy], i) => {
    const [dx1,,, dy2] = dirs[i]
    ctx.beginPath()
    ctx.moveTo(cx + dx1*BKT, cy)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx, cy + dy2*BKT)
    ctx.stroke()
  })
  ctx.shadowBlur = 0

  // ── Detection box helper ──────────────────────────────────────────────────
  function drawDetectionBox(wx0, wy0, color, label, boxW, boxH, idx) {
    const hw    = boxW / 2, hh = boxH / 2
    const bx    = OX + (wx0 - hw) * S
    const by    = OY + (wy0 - hh) * S
    const bw    = boxW * S, bh = boxH * S

    const pulse    = 0.65 + 0.35 * Math.sin(t * 3.5 + idx * 1.7)
    const colSolid = color.replace('ALPHA', String(pulse))
    const colFill  = color.replace('ALPHA', String(0.09 * pulse))
    const colGlow  = color.replace('ALPHA', String(0.18 * pulse))
    const colShadow= color.replace('ALPHA', '1')

    // Fill
    ctx.fillStyle = colFill
    ctx.fillRect(bx, by, bw, bh)

    // Glow pass
    ctx.strokeStyle = colGlow
    ctx.lineWidth   = 6
    ctx.shadowColor = colShadow; ctx.shadowBlur = 12
    ctx.strokeRect(bx, by, bw, bh)
    ctx.shadowBlur  = 0

    // Sharp outline
    ctx.strokeStyle = colSolid
    ctx.lineWidth   = 1.2
    ctx.strokeRect(bx, by, bw, bh)

    // YOLO corner brackets
    const BK = Math.min(8, bw * 0.2)
    const corners = [[bx,by],[bx+bw,by],[bx+bw,by+bh],[bx,by+bh]]
    const cdirs   = [[1,0,0,1],[-1,0,0,1],[-1,0,0,-1],[1,0,0,-1]]
    ctx.strokeStyle = colSolid; ctx.lineWidth = 2
    ctx.shadowColor = colShadow; ctx.shadowBlur = 4
    corners.forEach(([cx,cy],i) => {
      const [dx1,,,dy2] = cdirs[i]
      ctx.beginPath()
      ctx.moveTo(cx + dx1*BK, cy)
      ctx.lineTo(cx, cy)
      ctx.lineTo(cx, cy + dy2*BK)
      ctx.stroke()
    })
    ctx.shadowBlur = 0

    // Label above box
    ctx.fillStyle   = colSolid
    ctx.font        = `bold ${Math.max(7, Math.min(10, bw * 0.18))}px Courier New`
    ctx.textAlign   = 'center'
    ctx.shadowColor = colShadow; ctx.shadowBlur = 4
    ctx.fillText(label, bx + bw / 2, by - 4)
    ctx.shadowBlur  = 0
  }

  // ── Turkeys ───────────────────────────────────────────────────────────────
  mapTurkeys.forEach((tk, i) => {
    if (!Array.isArray(tk) || tk.length < 2) return
    drawDetectionBox(tk[0], tk[1], 'rgba(255,120,0,ALPHA)', `TURKEY-${i+1}`, 2.0, 2.0, i)
  })

  // ── Humans ────────────────────────────────────────────────────────────────
  mapHumans.forEach((hm, i) => {
    if (!Array.isArray(hm) || hm.length < 2) return
    drawDetectionBox(hm[0], hm[1], 'rgba(255,30,60,ALPHA)', `HUMAN-${i+1}`, 1.5, 1.5, i + 100)
  })

  // ── Robot ─────────────────────────────────────────────────────────────────
  if (mapRobot) {
    const rx  = mapRobot.x, ry = mapRobot.y
    const rth = mapRobot.theta ?? 0   // radians, 0 = right (+x), π/2 = down (+y)
    const [rpx, rpy] = proj(rx, ry)

    // Radial halo
    const haloR = Math.max(18, S * 1.2)
    const grd   = ctx.createRadialGradient(rpx, rpy, 0, rpx, rpy, haloR)
    grd.addColorStop(0, 'rgba(0,212,255,0.22)')
    grd.addColorStop(1, 'rgba(0,212,255,0)')
    ctx.fillStyle = grd
    ctx.beginPath(); ctx.arc(rpx, rpy, haloR, 0, Math.PI * 2); ctx.fill()

    // Direction beam
    const beamLen = Math.max(30, S * 5)
    const beamGrd = ctx.createLinearGradient(
      rpx, rpy,
      rpx + Math.cos(rth) * beamLen,
      rpy + Math.sin(rth) * beamLen
    )
    beamGrd.addColorStop(0, 'rgba(0,212,255,0.65)')
    beamGrd.addColorStop(1, 'rgba(0,212,255,0)')
    ctx.strokeStyle = beamGrd; ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(rpx, rpy)
    ctx.lineTo(rpx + Math.cos(rth) * beamLen, rpy + Math.sin(rth) * beamLen)
    ctx.stroke()

    // Chevron body
    const r = Math.max(7, Math.min(13, S * 0.7))
    ctx.save()
    ctx.translate(rpx, rpy)
    ctx.rotate(rth)
    ctx.strokeStyle = '#00d4ff'
    ctx.fillStyle   = 'rgba(0,212,255,0.22)'
    ctx.lineWidth   = 2
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 10
    ctx.beginPath()
    ctx.moveTo( r,        0)
    ctx.lineTo(-r * 0.6,  r * 0.75)
    ctx.lineTo(-r * 0.25, 0)
    ctx.lineTo(-r * 0.6, -r * 0.75)
    ctx.closePath(); ctx.fill(); ctx.stroke()
    ctx.shadowBlur = 0
    ctx.restore()

    // Label
    ctx.fillStyle   = '#00d4ff'
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 4
    ctx.fillText('ROBOT', rpx, rpy - r - 6)
    ctx.shadowBlur  = 0
  }

  // ── Axis labels ───────────────────────────────────────────────────────────
  ctx.font      = '7px Courier New'
  ctx.fillStyle = 'rgba(0,180,220,0.3)'
  ctx.textAlign = 'center'
  for (let gx = 0; gx <= MAP_W; gx += 20) {
    ctx.fillText(`${gx}`, OX + gx * S, OY + fh + 10)
  }
  ctx.textAlign = 'right'
  for (let gy = 0; gy <= MAP_H; gy += 20) {
    ctx.fillText(`${gy}`, OX - 4, OY + gy * S + 3)
  }

  // ── Legend ────────────────────────────────────────────────────────────────
  const legend = [
    ['rgba(255,120,0,0.9)',  '▣', 'Turkey'],
    ['rgba(255,30,60,0.9)',  '▣', 'Human'],
    ['rgba(0,212,255,0.9)',  '▶', 'Robot'],
    ['rgba(0,212,255,0.45)', '—', 'Arena / wall'],
  ]
  const lx0 = 8, ly0 = 12
  legend.forEach(([col, sym, lbl], i) => {
    const ly = ly0 + i * 13
    ctx.font = '9px Courier New'; ctx.textAlign = 'left'
    ctx.fillStyle = col;                    ctx.fillText(sym,       lx0,      ly)
    ctx.fillStyle = 'rgba(100,140,160,0.7)';ctx.fillText(` ${lbl}`, lx0 + 11, ly)
  })

  // ── Zoom hint ─────────────────────────────────────────────────────────────
  const zoomLabel = mapZoom > 1.01
    ? `${mapZoom.toFixed(1)}×  scroll to zoom · drag to pan`
    : `scroll to zoom`
  ctx.font = '8px Courier New'; ctx.textAlign = 'right'
  ctx.fillStyle = 'rgba(0,180,220,0.4)'
  ctx.fillText(zoomLabel, cw - 8, ch - 8)
}

// ── CSV parsing ────────────────────────────────────────────────────────────
//
// AUTO-DETECTS one of two formats:
//
// FORMAT A — Occupancy grid:
//   Row 0: ,0,1,2,...,N            ← column headers (first cell blank/text)
//   Row 1: 0,1,,,,1,,,             ← row index, then 1=wall blank=free
//   Cell (col, row) → world (x=col, y=flipped_row) so row 0 = top of field.
//
// FORMAT B — Named obstacle list:
//   x, y, label                    ← one obstacle per row, header optional
//
// Detection: header row col-1 and col-2 parse as numbers → occupancy grid.

function parseCSVText(text) {
  const lines      = text.trim().split(/\r?\n/).map(l => l.split(','))
  if (lines.length < 2) return

  const headerCols  = lines[0].map(c => c.trim())
  const isOccupancy = headerCols.length > 2 &&
                      !isNaN(parseFloat(headerCols[1])) &&
                      !isNaN(parseFloat(headerCols[2]))

  const obs = []
  let csvW = 0, csvH = 0

  if (isOccupancy) {
    const colX    = headerCols.slice(1).map(c => parseFloat(c.trim()))
    const maxColX = Math.max(...colX.filter(v => !isNaN(v)))
    const maxRowY = lines.length - 2

    csvW = maxColX + 1
    csvH = maxRowY + 1

    for (let r = 1; r < lines.length; r++) {
      const cells = lines[r]
      const rowY  = parseFloat(cells[0])
      if (isNaN(rowY)) continue
      for (let c = 1; c < cells.length; c++) {
        if (cells[c].trim() === '1') {
          const cellX = colX[c - 1]
          if (isNaN(cellX)) continue
          obs.push({ type: 'wall', x: cellX, y: csvH - 1 - rowY })
        }
      }
    }
  } else {
    lines.forEach(cols => {
      const x = parseFloat(cols[0])
      const y = parseFloat(cols[1])
      if (!isNaN(x) && !isNaN(y))
        obs.push({ type: 'point', x, y, label: cols[2]?.trim() ?? 'OBS' })
    })
  }

  mapObstacles = obs

  if (isOccupancy && csvW > 0 && csvH > 0) {
    MAP_W = csvW; MAP_H = csvH
    const sub = document.getElementById('map-sub')
    if (sub) sub.textContent = `${csvW} × ${csvH} M · GRID · ${obs.length} WALLS`
  }

  const btn = document.getElementById('csv-btn')
  if (btn) {
    btn.textContent       = `⊕ ${isOccupancy ? obs.length + ' WALLS' : obs.length + ' OBS'}`
    btn.style.color       = '#00d4ff'
    btn.style.borderColor = '#00d4ff'
  }
}

// File-picker entry point — called by the ⊕ CSV button
function loadCSV(input) {
  const file = input.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = e => parseCSVText(e.target.result)
  reader.readAsText(file)
}