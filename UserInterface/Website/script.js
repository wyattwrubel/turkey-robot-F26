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
let mapTurkeys   = []   // [[x, y, angle_rad], ...] live from VisionObject
let mapHumans    = []   // [[x, y, angle_rad], ...] live from VisionObject
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
initMap()
// ═══════════════════════════════════════════════
// MAP CANVAS
// ═══════════════════════════════════════════════
// Logical field dimensions in metres — updated dynamically when a CSV is loaded
let MAP_W = 100.0
let MAP_H = 100.0

function initMap() {
  const canvas = document.getElementById('map-canvas')
  if (!canvas) return

  function resize() {
    // Set pixel dimensions to match CSS layout size (devicePixelRatio sharpness)
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width  = rect.width  * dpr
    canvas.height = rect.height * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)          // scale once — all draw calls use CSS px from here
    canvas._cssW = rect.width
    canvas._cssH = rect.height
  }

  resize()
  new ResizeObserver(resize).observe(canvas)

  function loop() {
    drawMap(canvas)
    requestAnimationFrame(loop)
  }
  loop()
}

function drawMap(canvas) {
  const ctx  = canvas.getContext('2d')
  const cw   = canvas._cssW || canvas.offsetWidth
  const ch   = canvas._cssH || canvas.offsetHeight
  if (!cw || !ch) return

  const margin = 22
  // Uniform scale preserving aspect ratio, field centred in panel
  const scale  = Math.min((cw - margin * 2) / MAP_W, (ch - margin * 2) / MAP_H)
  const offX   = (cw - MAP_W * scale) / 2
  const offY   = (ch - MAP_H * scale) / 2

  // World → canvas coordinate helpers
  // Field: x right, y up.  Canvas: x right, y down — so flip y.
  function wx(x) { return offX + x * scale }
  function wy(y) { return ch - offY - y * scale }

  const t = Date.now() / 1000   // seconds, for animations

  // ── Background ─────────────────────────────────────────────
  ctx.clearRect(0, 0, cw, ch)
  ctx.fillStyle = '#080b0f'
  ctx.fillRect(0, 0, cw, ch)

  // Subtle grid lines every 10 m
  ctx.strokeStyle = '#0d1520'
  ctx.lineWidth   = 0.5
  for (let gx = 0; gx <= MAP_W; gx += 10) {
    ctx.beginPath(); ctx.moveTo(wx(gx), wy(0)); ctx.lineTo(wx(gx), wy(MAP_H)); ctx.stroke()
  }
  for (let gy = 0; gy <= MAP_H; gy += 10) {
    ctx.beginPath(); ctx.moveTo(wx(0), wy(gy)); ctx.lineTo(wx(MAP_W), wy(gy)); ctx.stroke()
  }

  // Grid labels (10 m intervals)
  ctx.font      = '7px Courier New'
  ctx.textAlign = 'center'
  ctx.fillStyle = '#1e2d3d'
  for (let gx = 0; gx <= MAP_W; gx += 10) {
    ctx.fillText(`${gx}`, wx(gx), wy(0) + 10)
  }
  ctx.textAlign = 'right'
  for (let gy = 0; gy <= MAP_H; gy += 10) {
    ctx.fillText(`${gy}`, wx(0) - 3, wy(gy) + 3)
  }

  // ── Zone boundary ───────────────────────────────────────────
  ctx.strokeStyle = '#1e2d3d'
  ctx.lineWidth   = 1
  ctx.strokeRect(wx(0), wy(MAP_H), MAP_W * scale, MAP_H * scale)

  // Corner brackets
  const cs = 8
  ;[[0, 0], [MAP_W, 0], [0, MAP_H], [MAP_W, MAP_H]].forEach(([cx, cy]) => {
    const px = wx(cx), py = wy(cy)
    const sx = cx === 0 ? 1 : -1
    const sy = cy === 0 ? -1 : 1
    ctx.strokeStyle = '#2a4060'
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(px + sx * cs, py); ctx.lineTo(px, py); ctx.lineTo(px, py + sy * cs)
    ctx.stroke()
  })

  // ── CSV obstacles or occupancy wall cells ───────────────────
  // mapObstacles entries are either:
  //   { type:'point', x, y, label }   — named obstacle dot
  //   { type:'wall',  x, y }          — 1m×1m filled wall cell from occupancy grid
  mapObstacles.forEach(ob => {
    const px = wx(ob.x), py = wy(ob.y)
    if (ob.type === 'wall') {
      // Draw a solid 1m×1m cell (clipped to 1px minimum)
      const cellW = Math.max(1, scale - 0.5)
      ctx.fillStyle = '#1a2535'
      ctx.fillRect(px, wy(ob.y + 1), cellW, cellW)
      ctx.strokeStyle = '#2a3f55'
      ctx.lineWidth   = 0.3
      ctx.strokeRect(px, wy(ob.y + 1), cellW, cellW)
    } else {
      // Named point obstacle — small diamond
      const r = Math.max(3, scale * 0.35)
      ctx.fillStyle   = '#111820'
      ctx.strokeStyle = '#2a4060'
      ctx.lineWidth   = 1
      ctx.beginPath()
      ctx.rect(px - r, py - r, r * 2, r * 2)
      ctx.fill(); ctx.stroke()
      const label = String(ob.label ?? 'OBS').substring(0, 6)
      ctx.fillStyle = '#4a6070'
      ctx.font      = '7px Courier New'
      ctx.textAlign = 'center'
      ctx.fillText(label, px, py - r - 2)
    }
  })

  // ── Turkeys (orange diamond + heading tick) ─────────────────
  mapTurkeys.forEach((tk, i) => {
    if (!Array.isArray(tk) || tk.length < 2) return
    const [tx, ty, ta = 0] = tk
    const px    = wx(tx), py = wy(ty)
    const r     = Math.max(5, scale * 0.45)
    const pulse = 0.7 + 0.3 * Math.sin(t * 3.0 + i * 1.5)

    // Radial glow
    const grd = ctx.createRadialGradient(px, py, 0, px, py, r * 3)
    grd.addColorStop(0, `rgba(255,107,0,${0.22 * pulse})`)
    grd.addColorStop(1, 'rgba(255,107,0,0)')
    ctx.fillStyle = grd
    ctx.beginPath(); ctx.arc(px, py, r * 3, 0, Math.PI * 2); ctx.fill()

    // Diamond body
    ctx.strokeStyle = `rgba(255,107,0,${pulse})`
    ctx.fillStyle   = `rgba(255,107,0,${0.3 * pulse})`
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(px,       py - r)
    ctx.lineTo(px + r,   py)
    ctx.lineTo(px,       py + r)
    ctx.lineTo(px - r,   py)
    ctx.closePath()
    ctx.fill(); ctx.stroke()

    // Heading tick (canvas y is inverted relative to field y, so negate sin)
    ctx.strokeStyle = '#ff8c00'
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(px, py)
    ctx.lineTo(px + Math.cos(ta) * r * 1.8, py - Math.sin(ta) * r * 1.8)
    ctx.stroke()

    // Label
    ctx.fillStyle = '#ff8c00'
    ctx.font      = 'bold 8px Courier New'
    ctx.textAlign = 'center'
    ctx.fillText(`T${i + 1}`, px, py - r - 3)
  })

  // ── Humans (red circle + heading tick) ─────────────────────
  mapHumans.forEach((hm, i) => {
    if (!Array.isArray(hm) || hm.length < 2) return
    const [hx, hy, ha = 0] = hm
    const px    = wx(hx), py = wy(hy)
    const r     = Math.max(5, scale * 0.45)
    const pulse = 0.8 + 0.2 * Math.sin(t * 2.0 + i * 1.8)

    // Radial glow
    const grd = ctx.createRadialGradient(px, py, 0, px, py, r * 3)
    grd.addColorStop(0, `rgba(255,34,51,${0.22 * pulse})`)
    grd.addColorStop(1, 'rgba(255,34,51,0)')
    ctx.fillStyle = grd
    ctx.beginPath(); ctx.arc(px, py, r * 3, 0, Math.PI * 2); ctx.fill()

    // Circle body
    ctx.strokeStyle = `rgba(255,34,51,${pulse})`
    ctx.fillStyle   = `rgba(255,34,51,${0.28 * pulse})`
    ctx.lineWidth   = 1.5
    ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2)
    ctx.fill(); ctx.stroke()

    // Heading tick
    ctx.strokeStyle = '#ff2233'
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(px, py)
    ctx.lineTo(px + Math.cos(ha) * r * 1.8, py - Math.sin(ha) * r * 1.8)
    ctx.stroke()

    // Label
    ctx.fillStyle = '#ff2233'
    ctx.font      = 'bold 8px Courier New'
    ctx.textAlign = 'center'
    ctx.fillText(`H${i + 1}`, px, py - r - 3)
  })

  // ── Robot pose (cyan chevron) ───────────────────────────────
  if (mapRobot) {
    const px = wx(mapRobot.x), py = wy(mapRobot.y)
    const th = mapRobot.theta ?? 0
    const r  = Math.max(7, scale * 0.65)

    // Glow
    const grd = ctx.createRadialGradient(px, py, 0, px, py, r * 3.5)
    grd.addColorStop(0, 'rgba(0,212,255,0.18)')
    grd.addColorStop(1, 'rgba(0,212,255,0)')
    ctx.fillStyle = grd
    ctx.beginPath(); ctx.arc(px, py, r * 3.5, 0, Math.PI * 2); ctx.fill()

    // Chevron — rotate in canvas space (negate th because y is flipped)
    ctx.save()
    ctx.translate(px, py)
    ctx.rotate(-th)
    ctx.strokeStyle = '#00d4ff'
    ctx.fillStyle   = 'rgba(0,212,255,0.22)'
    ctx.lineWidth   = 2
    ctx.beginPath()
    ctx.moveTo( r,       0)
    ctx.lineTo(-r * 0.6,  r * 0.7)
    ctx.lineTo(-r * 0.2,  0)
    ctx.lineTo(-r * 0.6, -r * 0.7)
    ctx.closePath()
    ctx.fill(); ctx.stroke()
    ctx.restore()

    // Label
    ctx.fillStyle = '#00d4ff'
    ctx.font      = 'bold 8px Courier New'
    ctx.textAlign = 'center'
    ctx.fillText('ROBOT', px, py - r - 4)
  }

  // ── Legend ──────────────────────────────────────────────────
  const legend = [
    ['#ff8c00', '◆', 'Turkey'],
    ['#ff2233', '●', 'Human'],
    ['#00d4ff', '▶', 'Robot'],
    ['#2a3f55', '■', 'Wall / Obstacle'],
  ]
  legend.forEach(([col, sym, lbl], i) => {
    const lx = offX + 4, ly = offY + 14 + i * 13
    ctx.fillStyle = col;       ctx.font = '9px Courier New'; ctx.textAlign = 'left'
    ctx.fillText(sym, lx, ly)
    ctx.fillStyle = '#4a6070'; ctx.fillText(` ${lbl}`, lx + 10, ly)
  })
}

// ── CSV loader ─────────────────────────────────────────────────────────────
//
// AUTO-DETECTS one of two formats:
//
// FORMAT A — Occupancy grid (your format):
//   Row 0: ,0,1,2,...,99          ← column headers (first cell empty or text)
//   Row 1: 0,1,,,,1,,,            ← row index, then 1=wall blank=free
//   Each cell represents a 1m × 1m square. Cells with "1" become wall tiles.
//   Field dimensions are set from the grid size.
//
// FORMAT B — Named obstacle list:
//   x, y, label                   ← one obstacle per row, header optional
//   First two columns must parse as numbers.
//
// Detection: if the first data row's first non-empty cell is a number AND
// the header row contains numeric column labels → occupancy grid.
// Otherwise → named obstacle list.

function loadCSV(input) {
  const file = input.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = e => {
    const text  = e.target.result
    const lines = text.trim().split(/\r?\n/).map(l => l.split(','))

    if (lines.length < 2) return

    // ── Detect format ────────────────────────────────────────────────────
    // An occupancy grid has a header row where column 1 is a number (col index 0)
    const headerCols  = lines[0].map(c => c.trim())
    const isOccupancy = headerCols.length > 2 &&
                        !isNaN(parseFloat(headerCols[1])) &&
                        !isNaN(parseFloat(headerCols[2]))

    const obs  = []
    let csvW = 0, csvH = 0

    if (isOccupancy) {
      // ── Format A: occupancy grid ────────────────────────────────────────
      // Header row gives column x-indices; first column of each data row is y-index.
      // Convention: row index = y coordinate (row 0 = y=0 at top of CSV).
      // We flip y so row 0 appears at the TOP of the field on screen (high y value).

      // Collect column x indices from header (skip first cell)
      const colX = headerCols.slice(1).map(c => parseFloat(c.trim()))
      const maxColX = Math.max(...colX.filter(v => !isNaN(v)))
      const maxRowY = lines.length - 2   // data rows (excl header)

      csvW = maxColX + 1
      csvH = maxRowY + 1

      for (let r = 1; r < lines.length; r++) {
        const cells = lines[r]
        const rowY  = parseFloat(cells[0])
        if (isNaN(rowY)) continue

        for (let c = 1; c < cells.length; c++) {
          const val = cells[c].trim()
          if (val === '1') {
            const cellX = colX[c - 1]
            if (isNaN(cellX)) continue
            // Flip Y: in CSV row 0 is "top"; on screen high y = top of field
            obs.push({ type: 'wall', x: cellX, y: csvH - 1 - rowY })
          }
        }
      }

    } else {
      // ── Format B: named obstacle list ───────────────────────────────────
      lines.forEach(cols => {
        const x = parseFloat(cols[0])
        const y = parseFloat(cols[1])
        if (!isNaN(x) && !isNaN(y)) {
          obs.push({ type: 'point', x, y, label: cols[2]?.trim() ?? 'OBS' })
        }
      })
      csvW = 0; csvH = 0  // don't override map dimensions
    }

    mapObstacles = obs

    // Resize map field to fit the grid (only for occupancy grids)
    if (isOccupancy && csvW > 0 && csvH > 0) {
      MAP_W = csvW
      MAP_H = csvH
      // Update subtitle
      const sub = document.getElementById('map-sub')
      if (sub) sub.textContent = `${csvW} × ${csvH} M · OCCUPANCY GRID · ${obs.length} WALLS`
    }

    // Update button
    const btn = document.getElementById('csv-btn')
    if (btn) {
      const fmt = isOccupancy ? `${obs.length} WALLS` : `${obs.length} OBS`
      btn.textContent      = `⊕ ${fmt}`
      btn.style.color      = '#00d4ff'
      btn.style.borderColor = '#00d4ff'
    }
  }
  reader.readAsText(file)
}