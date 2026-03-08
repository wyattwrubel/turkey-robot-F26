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
    img.src = 'data:image/jpeg;base64,' + m.image
    if (img.style.display === 'none') {
      img.style.display = 'block'
      document.querySelectorAll('#cam-ph .fi, #cam-ph .fl, #cam-ph .fs')
        .forEach(el => el.style.display = 'none')
    }
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