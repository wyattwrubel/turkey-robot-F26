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
// MAP CANVAS  — Pseudo-3D Tron arena
// ═══════════════════════════════════════════════
//
// Projection model: oblique axonometric (Cabinet projection, 45°, ratio 0.5)
//   World (x, y) in metres, y=0 at bottom, y=MAP_H at top.
//   Screen origin is the bottom-centre of the arena.
//
//   sx(x,y) = originX + (x - MAP_W/2) * S  +  y * cos45 * S * TILT
//   sy(x,y) = originY - y * sin45 * S * TILT  +  (x - MAP_W/2) * 0
//
//   TILT controls how steep the floor looks (0 = flat top-down, 1 = full iso).
//   We use TILT=0.45 — enough depth without distorting entity positions badly.
//
// All draw calls are plain Canvas2D — no libraries, Pi-safe at 60 fps.

let MAP_W = 100.0
let MAP_H = 100.0

// ── Projection constants (recomputed in drawMap each frame for resize safety) ──
const TILT   = 0.45   // vertical compression factor
const CAB    = 0.5    // cabinet projection oblique ratio
const WALL_H = 18     // px height of the extruded arena wall

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

  ;(function loop() { drawMap(canvas); requestAnimationFrame(loop) })()
}

function drawMap(canvas) {
  const ctx = canvas.getContext('2d')
  const cw  = canvas._cssW || canvas.offsetWidth
  const ch  = canvas._cssH || canvas.offsetHeight
  if (!cw || !ch) return

  const t = Date.now() / 1000   // animation clock (seconds)

  // ── Projection setup ─────────────────────────────────────────────────────
  // Scale: fit the projected floor inside the panel with margin
  const margin = 28
  // In oblique projection the floor's rendered width = MAP_W * S
  // and rendered height ≈ MAP_H * TILT * sin45 * S + MAP_W * 0 (no x shear on y)
  // We use a simpler fit: scale to fit MAP_W horizontally and MAP_H*TILT vertically
  const scaleX = (cw - margin * 2) / MAP_W
  const scaleY = (ch - margin * 2 - WALL_H * 2) / (MAP_H * TILT + MAP_W * CAB * 0.5)
  const S      = Math.min(scaleX, scaleY)

  // Arena origin: bottom-centre of the panel (floor origin = world 0,0)
  const OX = cw / 2 - (MAP_W / 2) * S          // left edge of floor at y=0
  const OY = ch - margin - WALL_H               // bottom edge of floor

  // Project world (x,y) → canvas (px,py)
  // x goes right, y goes up-and-back (oblique)
  function proj(wx, wy) {
    const px = OX + wx * S + wy * CAB * S * 0.5
    const py = OY - wy * TILT * S  + wx * 0
    return [px, py]
  }

  // ── Background ───────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, cw, ch)
  ctx.fillStyle = '#05080c'
  ctx.fillRect(0, 0, cw, ch)

  // ── Floor grid ───────────────────────────────────────────────────────────
  // Draw grid lines on the oblique floor plane
  const GRID_STEP = 10
  const gridAlpha = 0.18

  // Lines parallel to x-axis (constant y)
  for (let gy = 0; gy <= MAP_H; gy += GRID_STEP) {
    const [x0, y0] = proj(0,     gy)
    const [x1, y1] = proj(MAP_W, gy)
    const fade = 0.4 + 0.6 * (gy / MAP_H)  // lines farther back are slightly brighter
    ctx.strokeStyle = `rgba(0,180,220,${gridAlpha * fade})`
    ctx.lineWidth   = 0.5
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke()
  }

  // Lines parallel to y-axis (constant x)
  for (let gx = 0; gx <= MAP_W; gx += GRID_STEP) {
    const [x0, y0] = proj(gx, 0)
    const [x1, y1] = proj(gx, MAP_H)
    ctx.strokeStyle = `rgba(0,180,220,${gridAlpha})`
    ctx.lineWidth   = 0.5
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke()
  }

  // ── Scanline sweep ───────────────────────────────────────────────────────
  // A line that travels from y=0 to y=MAP_H and back, like a radar sweep
  const sweepY    = ((t * 0.18) % 1.0) * MAP_H   // world y of sweep line
  const sweepFade = Math.sin(t * 0.57 * Math.PI) * 0.5 + 0.5
  const [sx0, sy0] = proj(0,     sweepY)
  const [sx1, sy1] = proj(MAP_W, sweepY)
  const sweepGrd   = ctx.createLinearGradient(sx0, sy0, sx1, sy1)
  sweepGrd.addColorStop(0,   'rgba(0,212,255,0)')
  sweepGrd.addColorStop(0.5, `rgba(0,212,255,${0.55 * sweepFade})`)
  sweepGrd.addColorStop(1,   'rgba(0,212,255,0)')
  ctx.strokeStyle = sweepGrd
  ctx.lineWidth   = 1.5
  ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(sx1, sy1); ctx.stroke()

  // ── Occupancy wall tiles ─────────────────────────────────────────────────
  mapObstacles.filter(o => o.type === 'wall').forEach(ob => {
    // Draw a flat filled quad for each 1m×1m cell
    const [ax, ay] = proj(ob.x,     ob.y)
    const [bx, by] = proj(ob.x + 1, ob.y)
    const [cx2,cy2]= proj(ob.x + 1, ob.y + 1)
    const [dx, dy] = proj(ob.x,     ob.y + 1)
    ctx.fillStyle   = '#111c28'
    ctx.strokeStyle = '#1a3045'
    ctx.lineWidth   = 0.4
    ctx.beginPath()
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx2, cy2); ctx.lineTo(dx, dy)
    ctx.closePath(); ctx.fill(); ctx.stroke()
  })

  // Named point obstacles
  mapObstacles.filter(o => o.type === 'point').forEach(ob => {
    const [px, py] = proj(ob.x, ob.y)
    const r = 5
    ctx.strokeStyle = '#2a4060'; ctx.lineWidth = 1
    ctx.fillStyle   = '#0d1520'
    ctx.beginPath(); ctx.rect(px - r, py - r, r * 2, r * 2)
    ctx.fill(); ctx.stroke()
    ctx.fillStyle   = '#00d4ff'
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 3
    ctx.fillText(String(ob.label ?? 'OBS'), px, py - r - 3)
    ctx.shadowBlur  = 0
  })

  // ── Neon arena wall (perimeter) ──────────────────────────────────────────
  // Four corners of the arena
  const corners = [
    proj(0,     0),
    proj(MAP_W, 0),
    proj(MAP_W, MAP_H),
    proj(0,     MAP_H),
  ]
  const wallPulse  = 0.75 + 0.25 * Math.sin(t * 1.4)
  const WALL_COLOR = `rgba(0,212,255,${wallPulse})`
  const GLOW_COLOR = `rgba(0,212,255,${0.18 * wallPulse})`

  // Draw "extruded" wall face for each edge — a thin quad rising from the floor line
  const edges = [[0,1],[1,2],[2,3],[3,0]]
  edges.forEach(([a, b]) => {
    const [x0, y0] = corners[a]
    const [x1, y1] = corners[b]
    // Wall face: floor edge + top of wall (raised straight up in screen space)
    ctx.fillStyle = `rgba(0,130,180,${0.07 * wallPulse})`
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.lineTo(x1, y1 - WALL_H)
    ctx.lineTo(x0, y0 - WALL_H)
    ctx.closePath(); ctx.fill()
    // Top neon edge of the wall
    ctx.strokeStyle = WALL_COLOR
    ctx.lineWidth   = 1.5
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 8
    ctx.beginPath()
    ctx.moveTo(x0, y0 - WALL_H); ctx.lineTo(x1, y1 - WALL_H)
    ctx.stroke()
    ctx.shadowBlur = 0
  })

  // Floor perimeter line — glow pass then sharp pass
  ctx.strokeStyle = GLOW_COLOR
  ctx.lineWidth   = 6
  ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 14
  ctx.beginPath()
  ctx.moveTo(...corners[0])
  corners.slice(1).forEach(c => ctx.lineTo(...c))
  ctx.closePath(); ctx.stroke()
  ctx.shadowBlur  = 0

  ctx.strokeStyle = WALL_COLOR
  ctx.lineWidth   = 1.5
  ctx.beginPath()
  ctx.moveTo(...corners[0])
  corners.slice(1).forEach(c => ctx.lineTo(...c))
  ctx.closePath(); ctx.stroke()

  // ── Helper: draw a detection box in world space ──────────────────────────
  // box: { wx, wy, angle, color, label, boxW, boxH }
  // boxW/boxH are in metres — half-extents. Corners are projected onto the floor.
  // Draws an axis-aligned detection box centred at world (wx0, wy0).
  // No orientation — entities are x/y position only.
  function drawDetectionBox(wx0, wy0, color, label, boxW, boxH, idx) {
    const hw = boxW / 2, hh = boxH / 2
    // Four axis-aligned corners in world space
    const sCorners = [
      proj(wx0 - hw, wy0 - hh),
      proj(wx0 + hw, wy0 - hh),
      proj(wx0 + hw, wy0 + hh),
      proj(wx0 - hw, wy0 + hh),
    ]

    const pulse    = 0.65 + 0.35 * Math.sin(t * 3.5 + idx * 1.7)
    const colSolid = color.replace('ALPHA', String(pulse))
    const colFill  = color.replace('ALPHA', String(0.08 * pulse))
    const colGlow  = color.replace('ALPHA', String(0.12 * pulse))

    // Floor fill
    ctx.fillStyle = colFill
    ctx.beginPath()
    ctx.moveTo(...sCorners[0])
    sCorners.slice(1).forEach(c => ctx.lineTo(...c))
    ctx.closePath(); ctx.fill()

    // Glow pass
    ctx.strokeStyle = colGlow
    ctx.lineWidth   = 5
    ctx.shadowColor = color.replace('ALPHA', '1')
    ctx.shadowBlur  = 10
    ctx.beginPath()
    ctx.moveTo(...sCorners[0])
    sCorners.slice(1).forEach(c => ctx.lineTo(...c))
    ctx.closePath(); ctx.stroke()
    ctx.shadowBlur = 0

    // Sharp box outline
    ctx.strokeStyle = colSolid
    ctx.lineWidth   = 1.2
    ctx.beginPath()
    ctx.moveTo(...sCorners[0])
    sCorners.slice(1).forEach(c => ctx.lineTo(...c))
    ctx.closePath(); ctx.stroke()

    // Corner brackets (YOLO-style tick marks)
    const BKT = 5
    sCorners.forEach(([cx, cy], i) => {
      const prev = sCorners[(i + 3) % 4]
      const next = sCorners[(i + 1) % 4]
      const lenP = Math.hypot(prev[0] - cx, prev[1] - cy) || 1
      const lenN = Math.hypot(next[0] - cx, next[1] - cy) || 1
      ctx.strokeStyle = colSolid
      ctx.lineWidth   = 1.8
      ctx.beginPath()
      ctx.moveTo(cx + (prev[0] - cx) / lenP * BKT, cy + (prev[1] - cy) / lenP * BKT)
      ctx.lineTo(cx, cy)
      ctx.lineTo(cx + (next[0] - cx) / lenN * BKT, cy + (next[1] - cy) / lenN * BKT)
      ctx.stroke()
    })

    // Label tag — centred above the top edge midpoint
    const topMidX = (sCorners[2][0] + sCorners[3][0]) / 2
    const topMidY = (sCorners[2][1] + sCorners[3][1]) / 2 - 10
    ctx.fillStyle   = colSolid
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.shadowColor = color.replace('ALPHA', '1')
    ctx.shadowBlur  = 4
    ctx.fillText(label, topMidX, topMidY)
    ctx.shadowBlur  = 0
  }

  // ── Turkeys ──────────────────────────────────────────────────────────────
  mapTurkeys.forEach((tk, i) => {
    if (!Array.isArray(tk) || tk.length < 2) return
    const [tx, ty] = tk   // angle ignored — position only
    drawDetectionBox(tx, ty, 'rgba(255,120,0,ALPHA)', `TURKEY-${i+1}`, 2.0, 2.0, i)
  })

  // ── Humans ───────────────────────────────────────────────────────────────
  mapHumans.forEach((hm, i) => {
    if (!Array.isArray(hm) || hm.length < 2) return
    const [hx, hy] = hm   // angle ignored — position only
    drawDetectionBox(hx, hy, 'rgba(255,30,60,ALPHA)', `HUMAN-${i+1}`, 1.5, 1.5, i + 100)
  })

  // ── Robot ─────────────────────────────────────────────────────────────────
  if (mapRobot) {
    const rx = mapRobot.x, ry = mapRobot.y, rth = mapRobot.theta ?? 0
    const [rpx, rpy] = proj(rx, ry)

    // Glow halo
    const grd = ctx.createRadialGradient(rpx, rpy, 0, rpx, rpy, 28)
    grd.addColorStop(0, 'rgba(0,212,255,0.28)')
    grd.addColorStop(1, 'rgba(0,212,255,0)')
    ctx.fillStyle = grd
    ctx.beginPath(); ctx.arc(rpx, rpy, 28, 0, Math.PI * 2); ctx.fill()

    // Direction beam — fan from robot heading
    const beamLen = MAP_W * S * 0.22
    const beamAng = rth   // heading in world space
    // Project beam endpoint: step along world heading direction
    const beamWX  = rx + Math.cos(beamAng) * 8
    const beamWY  = ry + Math.sin(beamAng) * 8
    const [bex, bey] = proj(beamWX, beamWY)
    const beamGrd = ctx.createLinearGradient(rpx, rpy, bex, bey)
    beamGrd.addColorStop(0,   'rgba(0,212,255,0.6)')
    beamGrd.addColorStop(1,   'rgba(0,212,255,0)')
    ctx.strokeStyle = beamGrd
    ctx.lineWidth   = 2
    ctx.beginPath(); ctx.moveTo(rpx, rpy); ctx.lineTo(bex, bey); ctx.stroke()

    // Robot body — chevron rotated to heading on the projected floor
    // We approximate heading on screen using the projection of the heading vector
    const [hx2, hy2] = proj(rx + Math.cos(rth) * 1, ry + Math.sin(rth) * 1)
    const screenAng  = Math.atan2(hy2 - rpy, hx2 - rpx)
    const r = 9
    ctx.save()
    ctx.translate(rpx, rpy)
    ctx.rotate(screenAng)
    ctx.strokeStyle = '#00d4ff'
    ctx.fillStyle   = 'rgba(0,212,255,0.25)'
    ctx.lineWidth   = 2
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 8
    ctx.beginPath()
    ctx.moveTo( r,       0)
    ctx.lineTo(-r * 0.6,  r * 0.75)
    ctx.lineTo(-r * 0.2,  0)
    ctx.lineTo(-r * 0.6, -r * 0.75)
    ctx.closePath(); ctx.fill(); ctx.stroke()
    ctx.shadowBlur = 0
    ctx.restore()

    // Label
    ctx.fillStyle   = '#00d4ff'
    ctx.font        = 'bold 8px Courier New'
    ctx.textAlign   = 'center'
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 4
    ctx.fillText('ROBOT', rpx, rpy - r - 5)
    ctx.shadowBlur  = 0
  }

  // ── Coordinate labels on floor ────────────────────────────────────────────
  ctx.font      = '7px Courier New'
  ctx.fillStyle = 'rgba(0,180,220,0.35)'
  ctx.textAlign = 'center'
  for (let gx = 0; gx <= MAP_W; gx += 20) {
    const [lx, ly] = proj(gx, 0)
    ctx.fillText(`${gx}`, lx, ly + 10)
  }
  ctx.textAlign = 'right'
  for (let gy = 0; gy <= MAP_H; gy += 20) {
    const [lx, ly] = proj(0, gy)
    ctx.fillText(`${gy}`, lx - 4, ly + 3)
  }

  // ── Legend ────────────────────────────────────────────────────────────────
  const legend = [
    ['rgba(255,120,0,0.9)',  '▣', 'Turkey'],
    ['rgba(255,30,60,0.9)',  '▣', 'Human'],
    ['rgba(0,212,255,0.9)',  '▶', 'Robot'],
    ['rgba(0,212,255,0.45)', '—', 'Arena wall'],
    ['rgba(0,180,220,0.3)',  '▪', 'Wall tile'],
  ]
  const lx0 = 8, ly0 = 12
  legend.forEach(([col, sym, lbl], i) => {
    const ly = ly0 + i * 13
    ctx.fillStyle = col;           ctx.font = '9px Courier New'; ctx.textAlign = 'left'
    ctx.fillText(sym, lx0, ly)
    ctx.fillStyle = 'rgba(100,140,160,0.7)'; ctx.fillText(` ${lbl}`, lx0 + 11, ly)
  })
}

// ── CSV loader ─────────────────────────────────────────────────────────────
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

function loadCSV(input) {
  const file = input.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = e => {
    const lines      = e.target.result.trim().split(/\r?\n/).map(l => l.split(','))
    if (lines.length < 2) return

    const headerCols = lines[0].map(c => c.trim())
    const isOccupancy = headerCols.length > 2 &&
                        !isNaN(parseFloat(headerCols[1])) &&
                        !isNaN(parseFloat(headerCols[2]))

    const obs = []
    let csvW = 0, csvH = 0

    if (isOccupancy) {
      const colX   = headerCols.slice(1).map(c => parseFloat(c.trim()))
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
  reader.readAsText(file)
}