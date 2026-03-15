// ═══════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════
const SERVO_DEFS = [
  { id: 'FrontLeft',  name: 'Front Left',  min: -45, max: 45, val: 0 },
  { id: 'FrontRight', name: 'Front Right', min: -45, max: 45, val: 0 },
  { id: 'RearLeft',   name: 'Rear Left',   min: -45, max: 45, val: 0 },
  { id: 'RearRight',  name: 'Rear Right',  min: -45, max: 45, val: 0 },
  { id: 'Head',       name: 'Head',        min: -60, max: 60, val: 0 },
]

const AUDIO_DEFS = [
  { id: 'BarkShort',     label: '🔊 BarkShort'     },
  { id: 'BarkBurst',     label: '🔊 BarkBurst'     },
  { id: 'Growl',         label: '🔊 Growl'         },
  { id: 'AggressiveBark',label: '🔊 AggressiveBark'},
]

// ═══════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════
let ws       = null
let wsReady  = false
let isManual = false
let isStopped= false

// Map state
let mapTurkeys   = []  // [[x, y], ...]  extracted from T×3 vision array
let mapHumans    = []  // [[x, y], ...]  extracted from P×3 vision array
let mapObstacles = []  // [{x, y, type}, ...] from server on connect
let mapRobot     = { x: 0, y: 0, theta: 0 }

// Camera state — single <img> element, updated on each frame message
let camImg = null

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
    btn.className    = 'btn pu'
    btn.textContent  = a.label
    btn.dataset.audioId = a.id
    btn.onclick      = () => playAudio(a.id, btn)
    grid.appendChild(btn)
  })
}

// ═══════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════
function connect() {
  ws = new WebSocket('ws://localhost:8000/ws')
  ws.onopen    = ()  => { wsReady = true }
  ws.onclose   = ()  => { wsReady = false; setTimeout(connect, 1500) }
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

    // Raw sensors
    set('uwb-x', fmt(m.uwb_x, 2))
    set('uwb-y', fmt(m.uwb_y, 2))
    set('imu-t', deg(m.imu_theta))

    // UKF fused pose
    set('ukf-x', fmt(m.x, 2))
    set('ukf-y', fmt(m.y, 2))
    set('ukf-t', deg(m.theta))
    if (m.x != null) mapRobot = { x: m.x, y: m.y, theta: m.theta ?? 0 }

    // ── Always update — read-only sensor displays ──────────────

    // Current destination
    if (m.target != null) {
      set('dest-x', fmt(m.target.x, 2))
      set('dest-y', fmt(m.target.y, 2))
      set('dest-t', deg(m.target.theta))
    }

    // Drivetrain readout
    set('cmd-l', fmt(m.cmd_left,  2))
    set('cmd-r', fmt(m.cmd_right, 2))

    // ── Blocked in manual mode — server must not override user inputs ──
    if (!isManual) {

      // MacroState
      const ss   = m.macrostate ?? null
      const ssEl = document.getElementById('sys-state')
      if (ssEl) {
        if      (ss === 'Chase')    { ssEl.textContent = 'CHASE';    ssEl.className = 'v d' }
        else if (ss === 'Alert')    { ssEl.textContent = 'ALERT';    ssEl.className = 'v d' }
        else if (ss === 'Scanning') { ssEl.textContent = 'SCANNING'; ssEl.className = 'v w' }
        else                        { ssEl.textContent = '—';        ssEl.className = 'v'   }
      }
      document.querySelectorAll('.macrostate-g .btn').forEach(b => {
        b.classList.toggle('lit', b.dataset.state === ss)
      })

      // Speed slider
      if (m.speed_ms != null) {
        const sl = document.getElementById('spd-range')
        if (sl) { sl.value = m.speed_ms; updateSliderBg(sl) }
        set('spd-disp', m.speed_ms.toFixed(1))
      }

      // Servo sliders
      if (Array.isArray(m.servos)) syncServos(m.servos)

      // Pose button highlight
      if (m.active_pose !== undefined) {
        document.querySelectorAll('.pose-g .btn').forEach(b => {
          b.classList.toggle('lit', b.dataset.poseId === m.active_pose)
        })
      }

      // Audio button highlight
      if (m.active_audio !== undefined) {
        document.querySelectorAll('#audio-grid .btn').forEach(b => {
          b.classList.toggle('lit', b.dataset.audioId === m.active_audio)
        })
      }

    } // end !isManual
  }

  // ── Camera frame — raw passthrough ────────────────────────
  if (m.type === 'frame') {
    const ph = document.getElementById('cam-ph')
    if (!camImg) {
      // Create the img element once and insert it into the camera panel
      camImg = document.createElement('img')
      camImg.style.cssText = 'width:100%;height:100%;object-fit:contain;display:block;'
      ph.parentElement.appendChild(camImg)
      ph.style.display = 'none'
    }
    camImg.src = 'data:image/jpeg;base64,' + m.image
  }

  // ── Servo state (synced on connect) ───────────────────────
  if (m.type === 'servo_state') syncServos(m.servos)

  // ── Map obstacles (sent once on connect) ──────────────────
  if (m.type === 'map_obstacles') {
    mapObstacles = m.obstacles ?? []
    // Resize map canvas to match actual grid dimensions from server
    if (m.map_w) MAP_W = m.map_w
    if (m.map_h) MAP_H = m.map_h
    const sub = document.getElementById('map-sub')
    if (sub) sub.textContent = `${MAP_W} × ${MAP_H} M · 10 HZ`
  }

  // ── Vision detections ─────────────────────────────────────
  // Turkeys and Humans arrive as T×3 / P×3 arrays [x, y, angle].
  // Map only needs [x, y] — grab the first two columns.
  if (m.type === 'vision') {
    mapTurkeys = (m.turkeys ?? []).map(t => [t[0], t[1]])
    mapHumans  = (m.humans  ?? []).map(h => [h[0], h[1]])
  }
}

// ── Shared servo sync helper ──────────────────────────────
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
  flash(btn)
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
function sendMacrostate(state, el) {
  send({ type: 'macrostate_command', state })
  // Highlight the active button, clear the others
  document.querySelectorAll('.macrostate-g .btn').forEach(b => b.classList.remove('lit'))
  el.classList.add('lit')
}


// ═══════════════════════════════════════════════
// DRIVE CONTROLS
// Sends key name only — backend decides speed.
// ═══════════════════════════════════════════════

// Maps drive keys → drive button ids for visual feedback
const WASD_DRIVE_BTNS = {
  w: 'drive-fwd',
  s: 'drive-rev',
  a: 'drive-left',
  d: 'drive-right',
}

function startDrive(key, btnId) {
  if (!isManual || isStopped) return
  document.getElementById(btnId)?.classList.add('lit')
  send({ type: 'key_command', key, state: 'press' })
}

function stopDrive(key, btnId) {
  document.getElementById(btnId)?.classList.remove('lit')
  send({ type: 'key_command', key, state: 'release' })
}

function sendPose(p, el) {
  if (!isManual) return
  send({ type: 'pose_command', pose: p })
  flash(el)
}

// ── WASD ──────────────────────────────────────────────────
function wasdPress(k) {
  if (!isManual || isStopped) return
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
function emergencyStop() {
  isStopped = true
  send({ type: 'estop' })
  const b = document.getElementById('estop-btn')
  b.textContent      = '⬡ STOPPED — CLICK TO RESET'
  b.style.background = 'rgba(255,34,51,.35)'
  b.onclick          = resetEStop
}

function resetEStop() {
  isStopped = false
  send({ type: 'estop_reset' })
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
  document.getElementById('mode-badge').textContent = isManual ? 'MANUAL' : 'AUTO'
  document.getElementById('mode-badge').className   = isManual ? 'mode-badge manual' : 'mode-badge'
  document.getElementById('mode-btn').textContent   = isManual ? '⇄ SWITCH TO AUTO' : '⇄ SWITCH TO MANUAL'
  document.getElementById('mode-btn').className     = isManual ? 'modetog m' : 'modetog'
  document.getElementById('ctrl-tag').textContent   = isManual ? 'MODE: MANUAL — INPUTS ACTIVE' : 'MODE: AUTO — INPUTS LOCKED'
  applyModeToControls()
  if (!isManual) {
    // Release all drive keys visually
    Object.keys(WASD_DRIVE_BTNS).forEach(k => {
      document.getElementById('key-' + k)?.classList.remove('pressed')
      document.getElementById(WASD_DRIVE_BTNS[k])?.classList.remove('lit')
    })
    // Send UnlockUI to all three protocol channels
    send({ type: 'unlock_ui' })
  }
}

function applyModeToControls() {
  const locked = !isManual

  // Visual lock — dims sections and blocks pointer events via CSS class
  document.querySelector('.ctrl-top')    ?.classList.toggle('locked', locked)
  document.querySelector('.macrostate-g')?.classList.toggle('locked', locked)
  document.querySelector('.speed-wrap')  ?.classList.toggle('locked', locked)

  // Also disable all interactive elements so tab/keyboard can't reach them
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
function flash(el, duration=1200)  {
  if (!el) return
  el.classList.add('lit')
  setTimeout(() => el.classList.remove('lit'), duration)
}

// ═══════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════
buildServos()
buildAudio()
updateSliderBg(document.getElementById('spd-range'))
applyModeToControls()   // start locked in auto mode
connect()
requestAnimationFrame(initMap)

// ═══════════════════════════════════════════════
// MAP — 2D Top-Down Canvas
// ═══════════════════════════════════════════════
let MAP_W = 100.0
let MAP_H = 100.0

let mapZoom     = 1.0
let mapPanX     = 0.0
let mapPanY     = 0.0
let _dragActive = false
let _dragLast   = { x: 0, y: 0 }
let _lastS      = 1

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

  // ── Zoom ────────────────────────────────────────────────────
  canvas.addEventListener('wheel', e => {
    e.preventDefault()
    mapZoom = Math.min(6.0, Math.max(1.0, mapZoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)))
    _clampPan()
  }, { passive: false })

  // ── Pan — mouse ──────────────────────────────────────────────
  canvas.addEventListener('mousedown', e => {
    _dragActive = true
    _dragLast   = { x: e.clientX, y: e.clientY }
    canvas.style.cursor = 'grabbing'
  })
  window.addEventListener('mousemove', e => {
    if (!_dragActive) return
    mapPanX -= (e.clientX - _dragLast.x) / _lastS
    mapPanY -= (e.clientY - _dragLast.y) / _lastS
    _dragLast = { x: e.clientX, y: e.clientY }
    _clampPan()
  })
  window.addEventListener('mouseup', () => {
    _dragActive = false
    canvas.style.cursor = 'crosshair'
  })

  ;(function loop() { drawMap(canvas); requestAnimationFrame(loop) })()
}

function _clampPan() {
  mapPanX = Math.max(0, Math.min(MAP_W - MAP_W / mapZoom, mapPanX))
  mapPanY = Math.max(0, Math.min(MAP_H - MAP_H / mapZoom, mapPanY))
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

  // ── Background ────────────────────────────────────────────
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

  // ── Grid ──────────────────────────────────────────────────
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

  // ── Scan sweep ────────────────────────────────────────────
  const sweepY   = OY + ((t * 0.18) % 1.0) * fh
  const sweepGrd = ctx.createLinearGradient(OX, sweepY, OX + fw, sweepY)
  const fade     = Math.sin(t * 0.57 * Math.PI) * 0.5 + 0.5
  sweepGrd.addColorStop(0,   'rgba(0,212,255,0)')
  sweepGrd.addColorStop(0.3, `rgba(0,212,255,${0.45 * fade})`)
  sweepGrd.addColorStop(0.7, `rgba(0,212,255,${0.45 * fade})`)
  sweepGrd.addColorStop(1,   'rgba(0,212,255,0)')
  ctx.strokeStyle = sweepGrd; ctx.lineWidth = 1.5
  ctx.beginPath(); ctx.moveTo(OX, sweepY); ctx.lineTo(OX + fw, sweepY); ctx.stroke()

  // ── Obstacles ─────────────────────────────────────────────
  const wallPulse = 0.75 + 0.25 * Math.sin(t * 1.4)

  // Boundary lines
  mapObstacles.filter(o => o.type === 'line').forEach(ob => {
    const [ax, ay] = proj(ob.x1, ob.y1)
    const [bx, by] = proj(ob.x2, ob.y2)
    ctx.strokeStyle = `rgba(0,200,240,${wallPulse})`
    ctx.lineWidth   = 2
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 6
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
    ctx.shadowBlur  = 0
  })

  // Wall tiles (legacy 1×1 grid cells)
  mapObstacles.filter(o => o.type === 'wall').forEach(ob => {
    const tx = OX + ob.x * S, ty = OY + ob.y * S
    ctx.fillStyle   = `rgba(0,70,110,${0.45 * wallPulse})`
    ctx.fillRect(tx, ty, S, S)
    ctx.strokeStyle = `rgba(0,200,240,${wallPulse})`
    ctx.lineWidth   = 1
    ctx.strokeRect(tx, ty, S, S)
  })

  // Named point obstacles
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

  // ── Arena perimeter ───────────────────────────────────────
  ctx.strokeStyle = `rgba(0,212,255,${0.18 * wallPulse})`
  ctx.lineWidth   = 8
  ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 18
  ctx.strokeRect(fx0, fy0, fw, fh)
  ctx.shadowBlur  = 0
  ctx.strokeStyle = `rgba(0,212,255,${wallPulse})`
  ctx.lineWidth   = 1.5
  ctx.strokeRect(fx0, fy0, fw, fh)

  // ── Detection box helper ──────────────────────────────────
  function drawDetectionBox(wx, wy, color, label, boxW, boxH, idx) {
    const bx  = OX + (wx - boxW / 2) * S
    const by  = OY + (wy - boxH / 2) * S
    const bw  = boxW * S, bh = boxH * S
    const pulse = 0.65 + 0.35 * Math.sin(t * 3.5 + idx * 1.7)
    const col   = color.replace('ALPHA', String(pulse))
    const colFill = color.replace('ALPHA', String(0.09 * pulse))

    ctx.fillStyle   = colFill;                    ctx.fillRect(bx, by, bw, bh)
    ctx.strokeStyle = col; ctx.lineWidth = 1.2;   ctx.strokeRect(bx, by, bw, bh)
    ctx.fillStyle   = col
    ctx.font        = `bold 9px Courier New`
    ctx.textAlign   = 'center'
    ctx.fillText(label, bx + bw / 2, by - 4)
  }

  mapTurkeys.forEach((tk, i) =>
    drawDetectionBox(tk[0], tk[1], 'rgba(255,120,0,ALPHA)', `TURKEY-${i+1}`, 2.0, 2.0, i))

  mapHumans.forEach((hm, i) =>
    drawDetectionBox(hm[0], hm[1], 'rgba(255,30,60,ALPHA)', `HUMAN-${i+1}`, 1.5, 1.5, i + 100))

  // ── Robot ─────────────────────────────────────────────────
  const [rpx, rpy] = proj(mapRobot.x, mapRobot.y)
  const rth        = mapRobot.theta ?? 0
  const r          = Math.max(7, Math.min(13, S * 0.7))

  // Halo
  const haloGrd = ctx.createRadialGradient(rpx, rpy, 0, rpx, rpy, Math.max(18, S * 1.2))
  haloGrd.addColorStop(0, 'rgba(0,212,255,0.22)')
  haloGrd.addColorStop(1, 'rgba(0,212,255,0)')
  ctx.fillStyle = haloGrd
  ctx.beginPath(); ctx.arc(rpx, rpy, Math.max(18, S * 1.2), 0, Math.PI * 2); ctx.fill()

  // Direction beam
  const beamLen = Math.max(30, S * 5)
  const beamGrd = ctx.createLinearGradient(rpx, rpy,
    rpx + Math.cos(rth) * beamLen, rpy + Math.sin(rth) * beamLen)
  beamGrd.addColorStop(0, 'rgba(0,212,255,0.65)')
  beamGrd.addColorStop(1, 'rgba(0,212,255,0)')
  ctx.strokeStyle = beamGrd; ctx.lineWidth = 2
  ctx.beginPath(); ctx.moveTo(rpx, rpy)
  ctx.lineTo(rpx + Math.cos(rth) * beamLen, rpy + Math.sin(rth) * beamLen); ctx.stroke()

  // Chevron
  ctx.save()
  ctx.translate(rpx, rpy); ctx.rotate(rth)
  ctx.strokeStyle = '#00d4ff'; ctx.fillStyle = 'rgba(0,212,255,0.22)'
  ctx.lineWidth   = 2; ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 10
  ctx.beginPath()
  ctx.moveTo( r, 0); ctx.lineTo(-r * 0.6,  r * 0.75)
  ctx.lineTo(-r * 0.25, 0); ctx.lineTo(-r * 0.6, -r * 0.75)
  ctx.closePath(); ctx.fill(); ctx.stroke()
  ctx.shadowBlur = 0; ctx.restore()

  ctx.fillStyle = '#00d4ff'; ctx.font = 'bold 8px Courier New'; ctx.textAlign = 'center'
  ctx.fillText('ROBOT', rpx, rpy - r - 6)

  // ── Axis labels ───────────────────────────────────────────
  ctx.font = '7px Courier New'; ctx.fillStyle = 'rgba(0,180,220,0.3)'
  ctx.textAlign = 'center'
  for (let gx = 0; gx <= MAP_W; gx += 20)
    ctx.fillText(`${gx}`, OX + gx * S, OY + fh + 10)
  ctx.textAlign = 'right'
  for (let gy = 0; gy <= MAP_H; gy += 20)
    ctx.fillText(`${gy}`, OX - 4, OY + gy * S + 3)

  // ── Legend ────────────────────────────────────────────────
  const legend = [
    ['rgba(255,120,0,0.9)', '▣', 'Turkey'],
    ['rgba(255,30,60,0.9)', '▣', 'Human'],
    ['rgba(0,212,255,0.9)', '▶', 'Robot'],
    ['rgba(0,212,255,0.45)','—', 'Wall'],
  ]
  legend.forEach(([col, sym, lbl], i) => {
    const ly = 12 + i * 13
    ctx.font = '9px Courier New'; ctx.textAlign = 'left'
    ctx.fillStyle = col;                     ctx.fillText(sym,        8, ly)
    ctx.fillStyle = 'rgba(100,140,160,0.7)'; ctx.fillText(` ${lbl}`, 19, ly)
  })

  // ── Zoom hint ─────────────────────────────────────────────
  ctx.font = '8px Courier New'; ctx.textAlign = 'right'
  ctx.fillStyle = 'rgba(0,180,220,0.4)'
  ctx.fillText(mapZoom > 1.01
    ? `${mapZoom.toFixed(1)}×  scroll to zoom · drag to pan`
    : `scroll to zoom`, cw - 8, ch - 8)

  ctx.restore()
}