// ═══════════════════════════════════════════════
// CONFIG — swap in real values when ready
// ═══════════════════════════════════════════════
const SERVO_DEFS = [
  { id: 'FrontLeft', name: 'Front Left', min: -45,   max: 45, val: 0  },
  { id: 'FrontRight', name: 'Front Right', min: -45,   max: 45, val: 0  },
  { id: 'RearLeft', name: 'Rear Left', min: -45,   max: 45, val: 0 },
  { id: 'RearRight', name: 'Rear Right', min: -45, max: 45,  val: 0   },
  { id: 'Head', name: 'Head', min: -60,   max: 60, val: 0  },
]

const AUDIO_DEFS = [
  { id: 'BarkShort',    label: '🔊 BarkShort'    },
  { id: 'BarkBurst',   label: '🔊 BarkBurst'   },
  { id: 'Growl', label: '🔊 Growl' },
  { id: 'AggressiveBark',    label: '🔊 AggressiveBark'    },
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

// ── Camera state ─────────────────────────────────────────────
// Two canvases: #cam-bg (background, redraws only on new frame)
//               #cam-hud (HUD overlay, animates at ~15fps)
// camFrame:    currently displayed image element
// _camBgDirty: true when a new frame has arrived and bg needs one redraw
let camBboxes   = []
let camFrame    = null
let camFrameW   = 640
let camFrameH   = 480
let _camBgDirty = false   // set true when new frame decoded


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

  // ── Camera frame ──────────────────────────────────────────
  if (m.type === 'frame') {
    const w   = m.w || 640
    const h   = m.h || 480
    const img = new Image()
    img.onload = () => {
      camFrame    = img
      camFrameW   = w
      camFrameH   = h
      _camBgDirty = true   // signal bg canvas to redraw once
      // Reveal both canvases on first frame
      const bg  = document.getElementById('cam-bg')
      const hud = document.getElementById('cam-hud')
      if (bg  && bg.style.display  === 'none') bg.style.display  = 'block'
      if (hud && hud.style.display === 'none') hud.style.display = 'block'
      document.getElementById('cam-ph').style.display = 'none'
      _startCamLoop()
    }
    img.src = 'data:image/jpeg;base64,' + m.image
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

  // ── Vision detections ─────────────────────────────────────
  if (m.type === 'vision') {
    mapTurkeys = m.turkeys ?? []
    mapHumans  = m.humans  ?? []
    camBboxes  = m.bboxes  ?? []
    // rAF loop picks up new bboxes on its next tick — no manual redraw needed
  }
}

// ═══════════════════════════════════════════════════════════════════
// CAMERA — two-layer canvas rendering
//
//  #cam-bg  (background layer)
//    Draws: raw frame, scanlines, vignette, tint
//    Redraws: ONLY when _camBgDirty is true (new frame arrived)
//    This layer never flickers — it just holds the last good image.
//
//  #cam-hud (HUD overlay layer)
//    Draws: bboxes, lock brackets, diagnostic panels, chrome text
//    Redraws: every tick at ~15fps — transparent background so bg shows through
//
// Nothing outside this block calls drawBg or drawHud directly.
// ═══════════════════════════════════════════════════════════════════

let _camRafId      = null
let _camFrameCount = 0
let _camLastDraw   = 0
const CAM_FPS      = 15
const CAM_INTERVAL = 1000 / CAM_FPS

// ── Helpers ───────────────────────────────────────────────────────
function _fitCanvas(canvas) {
  // Resize canvas backing store to match CSS size.
  // Returns {ctx, cssW, cssH} or null if not visible.
  // IMPORTANT: only resizes if dimensions actually changed —
  // assigning canvas.width even to the same value resets the canvas.
  const dpr  = window.devicePixelRatio || 1
  const cssW = canvas.clientWidth
  const cssH = canvas.clientHeight
  if (cssW === 0 || cssH === 0) return null
  const needW = Math.round(cssW * dpr)
  const needH = Math.round(cssH * dpr)
  if (canvas.width !== needW || canvas.height !== needH) {
    canvas.width  = needW
    canvas.height = needH
    return { ctx: canvas.getContext('2d'), cssW, cssH, resized: true }
  }
  return { ctx: canvas.getContext('2d'), cssW, cssH, resized: false }
}

// ── Background layer ──────────────────────────────────────────────
// Draws the camera image, scanlines, vignette.
// Called once per new frame — never called again until next frame arrives.
function _drawBg() {
  const canvas = document.getElementById('cam-bg')
  if (!canvas || !camFrame) return

  const dpr  = window.devicePixelRatio || 1
  const fit  = _fitCanvas(canvas)
  if (!fit) return
  const { ctx, cssW, cssH } = fit
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const scale = Math.min(cssW / camFrameW, cssH / camFrameH)
  const drawW = camFrameW * scale
  const drawH = camFrameH * scale
  const offX  = (cssW - drawW) / 2
  const offY  = (cssH - drawH) / 2

  // Black letterbox
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, cssW, cssH)

  // Camera frame
  ctx.drawImage(camFrame, offX, offY, drawW, drawH)

  // Cyan tint
  ctx.fillStyle = 'rgba(0,180,220,0.06)'
  ctx.fillRect(offX, offY, drawW, drawH)

  // Scanlines — drawn once, sit on top of image permanently
  ctx.fillStyle = 'rgba(0,0,0,0.15)'
  for (let y = offY; y < offY + drawH; y += 3) {
    ctx.fillRect(offX, y, drawW, 1)
  }

  // Vignette
  const vig = ctx.createRadialGradient(cssW/2, cssH/2, cssH*0.25, cssW/2, cssH/2, cssH*0.85)
  vig.addColorStop(0, 'rgba(0,0,0,0)')
  vig.addColorStop(1, 'rgba(0,0,0,0.55)')
  ctx.fillStyle = vig
  ctx.fillRect(0, 0, cssW, cssH)
}

// ── HUD overlay layer ─────────────────────────────────────────────
// Draws bboxes, chrome, text — on a fully transparent canvas.
// Clears and redraws every tick. Background shows through underneath.
function _drawHud(t) {
  const canvas = document.getElementById('cam-hud')
  if (!canvas || !camFrame) return

  const dpr  = window.devicePixelRatio || 1
  const fit  = _fitCanvas(canvas)
  if (!fit) return
  const { ctx, cssW, cssH, resized } = fit

  // If HUD canvas was resized, bg needs a redraw too
  if (resized) _camBgDirty = true

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)   // transparent — bg canvas shows through

  const hasTargets = camBboxes.length > 0
  const scale = Math.min(cssW / camFrameW, cssH / camFrameH)
  const drawW = camFrameW * scale
  const drawH = camFrameH * scale
  const offX  = (cssW - drawW) / 2
  const offY  = (cssH - drawH) / 2

  // ── Scrolling scan sweep ───────────────────────────────────────
  const sweepY = offY + ((t * 0.3) % 1) * drawH
  const swGrd  = ctx.createLinearGradient(0, sweepY - 12, 0, sweepY + 12)
  swGrd.addColorStop(0,   'rgba(0,0,0,0)')
  swGrd.addColorStop(0.5, 'rgba(0,180,220,0.07)')
  swGrd.addColorStop(1,   'rgba(0,0,0,0)')
  ctx.fillStyle = swGrd
  ctx.fillRect(offX, sweepY - 12, drawW, 24)

  // ── Target lock boxes ──────────────────────────────────────────
  camBboxes.forEach((box, idx) => {
    const { label = '?', conf = 0, x1=0, y1=0, x2=0, y2=0 } = box
    const lbl      = String(label).toLowerCase()
    const isTurkey = lbl.includes('turkey')
    const isHuman  = lbl.includes('human')
    const base     = isTurkey ? '0,212,255' : isHuman ? '255,30,60' : '0,212,255'

    const bx1 = offX + x1 * scale, by1 = offY + y1 * scale
    const bx2 = offX + x2 * scale, by2 = offY + y2 * scale
    const bw  = bx2 - bx1,         bh  = by2 - by1
    const bcx = bx1 + bw / 2,      bcy = by1 + bh / 2

    const pulse  = 0.75 + 0.25 * Math.sin(t * 4 + idx * 2.1)
    const locked = ((t * 0.5) % 1.2) >= 0.95

    // Box fill
    ctx.fillStyle = `rgba(${base},${locked ? 0.06 : 0.03})`
    ctx.fillRect(bx1, by1, bw, bh)

    // Glow outline
    ctx.strokeStyle = `rgba(${base},${0.25 * pulse})`
    ctx.lineWidth   = 8
    ctx.shadowColor = `rgba(${base},1)`
    ctx.shadowBlur  = 16
    ctx.strokeRect(bx1, by1, bw, bh)
    ctx.shadowBlur  = 0
    ctx.strokeStyle = `rgba(${base},${pulse})`
    ctx.lineWidth   = 1
    ctx.strokeRect(bx1, by1, bw, bh)

    // Corner brackets
    const BK    = Math.min(16, bw * 0.18, bh * 0.18)
    const cDefs = [[bx1,by1,1,1],[bx2,by1,-1,1],[bx2,by2,-1,-1],[bx1,by2,1,-1]]
    ctx.strokeStyle = `rgba(${base},${pulse})`
    ctx.lineWidth   = 2
    ctx.shadowColor = `rgba(${base},1)`
    ctx.shadowBlur  = 8
    cDefs.forEach(([cx, cy, dx, dy], i) => {
      ctx.save()
      ctx.translate(cx, cy)
      ctx.rotate(locked ? 0 : Math.sin(t * 2 + i) * 0.18)
      ctx.beginPath()
      ctx.moveTo(dx * BK, 0); ctx.lineTo(0, 0); ctx.lineTo(0, dy * BK)
      ctx.stroke()
      ctx.restore()
    })
    ctx.shadowBlur = 0

    // Centre crosshair (when locked)
    if (locked) {
      const crs = Math.min(bw, bh) * 0.12
      ctx.strokeStyle = `rgba(${base},${0.5 * pulse})`
      ctx.lineWidth   = 1
      ctx.setLineDash([2, 4])
      ctx.beginPath()
      ctx.moveTo(bcx - crs, bcy); ctx.lineTo(bcx + crs, bcy)
      ctx.moveTo(bcx, bcy - crs); ctx.lineTo(bcx, bcy + crs)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = `rgba(${base},${pulse})`
      ctx.shadowColor = `rgba(${base},1)`; ctx.shadowBlur = 6
      ctx.beginPath(); ctx.arc(bcx, bcy, 2, 0, Math.PI*2); ctx.fill()
      ctx.shadowBlur = 0
    }

    // Diagnostic panel (inside top of box)
    const panW   = bw - 4, panH = 42
    const panX   = bx1 + 2, panY = by1 + 2, panPad = 7
    ctx.fillStyle = 'rgba(4,10,18,0.72)'
    ctx.fillRect(panX, panY, panW, panH)
    ctx.fillStyle = `rgba(${base},0.8)`
    ctx.fillRect(panX, panY, panW, 2)

    ctx.textAlign = 'left'; ctx.textBaseline = 'top'
    const px = panX + panPad
    let   py = panY + 7

    ctx.font        = 'bold 10px Courier New'
    ctx.fillStyle   = `rgba(${base},1)`
    ctx.shadowColor = `rgba(${base},1)`; ctx.shadowBlur = 6
    ctx.fillText(`TGT-${String(idx+1).padStart(2,'0')}  ${locked ? '◉ LOCKED' : '◎ ACQUIRING'}`, px, py)
    ctx.shadowBlur  = 0
    py += 14

    ctx.font      = '8px Courier New'
    ctx.fillStyle = `rgba(${base},0.75)`
    ctx.fillText(label.toUpperCase(), px, py)

    const barOffX = px + 52
    const barW    = panW - panPad - 52 - 26
    ctx.fillStyle = 'rgba(20,40,60,0.9)'
    ctx.fillRect(barOffX, py + 1, barW, 6)
    ctx.fillStyle   = `rgba(${base},0.9)`
    ctx.shadowColor = `rgba(${base},1)`; ctx.shadowBlur = 4
    ctx.fillRect(barOffX, py + 1, barW * conf, 6)
    ctx.shadowBlur  = 0
    ctx.fillStyle   = `rgba(${base},0.85)`
    ctx.fillText(`${Math.round(conf*100)}%`, barOffX + barW + 4, py)
    ctx.textBaseline = 'alphabetic'
  })

  // ── Global chrome ──────────────────────────────────────────────
  const stamp  = new Date().toTimeString().slice(0,8)
  const fCount = (_camFrameCount++).toString().padStart(6,'0')

  // Corner brackets
  const CL = 18
  const corners = [[0,0,1,1],[cssW,0,-1,1],[cssW,cssH,-1,-1],[0,cssH,1,-1]]
  ctx.strokeStyle = 'rgba(0,180,220,0.6)'
  ctx.lineWidth   = 1.5
  corners.forEach(([cx,cy,dx,dy]) => {
    ctx.beginPath()
    ctx.moveTo(cx+dx*CL, cy); ctx.lineTo(cx,cy); ctx.lineTo(cx,cy+dy*CL)
    ctx.stroke()
  })

  // Top-left
  ctx.font = 'bold 10px Courier New'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'
  ctx.fillStyle = 'rgba(0,212,255,0.9)'
  ctx.shadowColor = 'rgba(0,212,255,1)'; ctx.shadowBlur = 8
  ctx.fillText('CDS · VISION SYS v1.0', 6, 6)
  ctx.shadowBlur = 0
  ctx.font = '8px Courier New'
  ctx.fillStyle = 'rgba(0,212,255,0.6)'
  ctx.fillText(`CH.01  AUTO  ${stamp}`, 6, 20)

  // Top-right
  ctx.textAlign = 'right'
  if (hasTargets) {
    const wp = 0.7 + 0.3 * Math.sin(t * 6)
    ctx.font = 'bold 15px Courier New'
    ctx.fillStyle   = `rgba(255,60,0,${wp})`
    ctx.shadowColor = 'rgba(255,60,0,1)'; ctx.shadowBlur = 18
    ctx.fillText('⚠ TARGET ACQUIRED', cssW - 6, 16)
    ctx.shadowBlur = 0
    ctx.font = 'bold 9px Courier New'
    ctx.fillStyle   = `rgba(255,140,0,${0.6 + 0.4 * Math.sin(t * 4)})`
    ctx.shadowColor = 'rgba(255,140,0,1)'; ctx.shadowBlur = 8
    ctx.fillText(`${camBboxes.length} AVIAN THREAT${camBboxes.length>1?'S':''} DETECTED`, cssW - 6, 30)
    ctx.shadowBlur = 0
  } else {
    ctx.font = 'bold 10px Courier New'
    ctx.fillStyle   = 'rgba(0,212,255,0.65)'
    ctx.shadowColor = 'rgba(0,212,255,0.8)'; ctx.shadowBlur = 6
    ctx.fillText('SCANNING...', cssW - 6, 16)
    ctx.shadowBlur = 0
    ctx.font = '8px Courier New'
    ctx.fillStyle = 'rgba(0,180,220,0.4)'
    ctx.fillText('NO TARGETS', cssW - 6, 28)
  }

  // Bottom bar
  const barY = cssH - 14
  ctx.fillStyle = 'rgba(0,0,0,0.45)'
  ctx.fillRect(0, barY, cssW, 14)
  ctx.strokeStyle = 'rgba(0,180,220,0.25)'; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0,barY); ctx.lineTo(cssW,barY); ctx.stroke()
  ctx.font = '8px Courier New'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
  ctx.fillStyle = 'rgba(0,212,255,0.7)'
  ctx.fillText(`FRAME ${fCount}`, 6, barY + 7)
  ctx.textAlign = 'center'
  ctx.fillStyle = 'rgba(30,210,250,0.75)'
  ctx.shadowColor = 'rgba(0,180,220,0.8)'; ctx.shadowBlur = 6
  ctx.fillText('ME450 · TURKEYSQUAD · CANINE DETERRENT SYS', cssW/2, barY + 7)
  ctx.shadowBlur = 0
  ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic'
}

// ── rAF loop ──────────────────────────────────────────────────────
function _startCamLoop() {
  if (_camRafId) return
  function loop(ts) {
    _camRafId = requestAnimationFrame(loop)
    if (!camFrame) return
    if (ts - _camLastDraw < CAM_INTERVAL) return
    _camLastDraw = ts
    if (_camBgDirty) {
      _drawBg()
      _camBgDirty = false
    }
    _drawHud(ts / 1000)
  }
  _camRafId = requestAnimationFrame(loop)
}

// Resize observer: just mark bg dirty so it redraws at correct size next tick
;(function() {
  const bg = document.getElementById('cam-bg')
  if (bg) new ResizeObserver(() => { _camBgDirty = true }).observe(bg)
})()

// rAF loop starts lazily on first frame decode

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
  ctx.save()
  ctx.shadowBlur = 0
  ctx.setLineDash([])
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
  ctx.restore()
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