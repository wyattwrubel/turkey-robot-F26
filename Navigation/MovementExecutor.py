import asyncio
import time
import numpy as np
import sys
sys.path.insert(0, '/home/turkeysquad/ME450/MainDesign/casadi')
if 'casadi' in sys.modules:
    del sys.modules['casadi']
import casadi as ca

# =============================================================================
# Config
# =============================================================================
WHEELBASE = 0.7
L         = WHEELBASE
DT        = 0.1
N         = 8

V_MAX =  3.0
V_MIN = -3.0

FINAL_DIST  = 0.8
CHASE_DIST  = 3.0

STALL_WINDOW       = 30
STALL_DELTA        = 0.3
STALL_ARRIVAL_MULT = 2.0

CHASE_HANDOFF_DIST  = 10.0
LOOKAHEAD_DIST      = 2.5

PIVOT_THRESHOLD_DEG = 10.0
PIVOT_ALIGN_DEG     =  3.0
PIVOT_KP            =  3.0

W_XY_RUN     =  5.0
W_XY_TERM    = 80.0
W_FOV        =  8.0
W_HEADING    = 12.0
W_OMEGA      =  1.0
W_INPUT_DIFF =  0.5

N_STATIC_RECTS       = 8
N_DYN_CIRCLES        = 5
STATIC_MARGIN        = 0.5
DYN_R_TOTAL          = 2.0
RECT_ACTIVATION_DIST = 4.0

# Obstacle contact detection: if EDT stays <= this threshold for this many
# consecutive ticks, trigger an immediate replan regardless of stall window.
OBSTACLE_CONTACT_MARGIN = STATIC_MARGIN + 0.1   # 0.6 m
OBSTACLE_CONTACT_TICKS  = 10                     # 1 s at DT=0.1

# =============================================================================
# Helpers
# =============================================================================
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def get_state(parent):
    raw = np.array(parent.Current_State, dtype=np.float64)
    return np.array([raw[0], raw[1], wrap(raw[2])])

def _rect_dist(rect, x, y):
    x0, x1, y0, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return np.sqrt(dx*dx + dy*dy)

_DUMMY_RECT   = [999.0, 1000.0, 999.0, 1000.0]
_DUMMY_CIRCLE = [999.0, 999.0]

def _pack_static_rects(obstacle_rects, rx, ry):
    out = []; n_active = 0
    for rect in obstacle_rects[:N_STATIC_RECTS]:
        if _rect_dist(rect, rx, ry) <= RECT_ACTIVATION_DIST:
            x0, x1, y0, y1 = rect
            out.extend([float(x0), float(x1), float(y0), float(y1)])
            n_active += 1
        else:
            out.extend(_DUMMY_RECT)
    while len(out) < 4 * N_STATIC_RECTS:
        out.extend(_DUMMY_RECT)
    return np.array(out, dtype=np.float64), n_active

def _pack_dyn_circles(dynamic_obs):
    out = []
    for cx, cy in dynamic_obs[:N_DYN_CIRCLES]:
        out.extend([float(cx), float(cy)])
    while len(out) < 2 * N_DYN_CIRCLES:
        out.extend(_DUMMY_CIRCLE)
    return np.array(out, dtype=np.float64)

# =============================================================================
# MPC solver
# =============================================================================
# Scan solver — pure trajectory tracking, no obstacle constraints
# Parameter vector: X0(3) | lookahead(2) | chase=0(1) | turkey_unused(2) | uprev(2)
# =============================================================================
def build_scan_solver(n_steps=N):
    n = n_steps; ns = 3; nu = 2

    xs,ys,ths = ca.SX.sym('x'),ca.SX.sym('y'),ca.SX.sym('th')
    vls,vrs   = ca.SX.sym('vl'),ca.SX.sym('vr')
    vs=(vls+vrs)/2; ws=(vrs-vls)/L
    f = ca.Function('f',
        [ca.vertcat(xs,ys,ths), ca.vertcat(vls,vrs)],
        [ca.vertcat(xs,ys,ths)+DT*ca.vertcat(vs*ca.cos(ths),-vs*ca.sin(ths),ws)])

    X=ca.SX.sym('X',ns,n+1); U=ca.SX.sym('U',nu,n)
    PX0=ca.SX.sym('X0',ns); Pl=ca.SX.sym('l',2); Pup=ca.SX.sym('up',nu)
    p=ca.vertcat(PX0,Pl,Pup)

    obj=0; geq=[X[:,0]-PX0]
    for k in range(n):
        xk=X[0,k]; yk=X[1,k]; thk=X[2,k]; vlk=U[0,k]; vrk=U[1,k]
        obj += W_XY_RUN*((xk-Pl[0])**2+(yk-Pl[1])**2)
        wdx=Pl[0]-xk; wdy=Pl[1]-yk; wreg=ca.sqrt(wdx**2+wdy**2+1e-6)
        aw=ca.atan2(-wdy/wreg,wdx/wreg)
        obj += W_HEADING*ca.atan2(ca.sin(aw-thk),ca.cos(aw-thk))**2
        obj += W_OMEGA*((vrk-vlk)/L)**2
        upk=Pup if k==0 else U[:,k-1]
        obj += W_INPUT_DIFF*ca.sumsqr(U[:,k]-upk)
        geq.append(X[:,k+1]-f(X[:,k],U[:,k]))
    obj += W_XY_TERM*((X[0,n]-Pl[0])**2+(X[1,n]-Pl[1])**2)

    ov=ca.vertcat(ca.reshape(X,-1,1),ca.reshape(U,-1,1))
    nd=ov.numel(); cs=ns*(n+1)
    lbx=-np.inf*np.ones(nd); ubx=np.inf*np.ones(nd)
    lbx[cs:]=V_MIN; ubx[cs:]=V_MAX
    gc=ca.vertcat(*geq); ne=gc.numel()
    lbg=np.zeros(ne); ubg=np.zeros(ne)

    slv=ca.nlpsol('scan_solver','ipopt',{'f':obj,'x':ov,'g':gc,'p':p},
        {'ipopt.print_level':0,'print_time':0,'ipopt.tol':1e-2,
         'ipopt.max_iter':50,'ipopt.warm_start_init_point':'yes',
         'ipopt.mu_strategy':'adaptive'})

    return dict(solver=slv,lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg,
                n_dec=nd,ctrl_start=cs,ns=ns,nu=nu,N=n,n_p=p.numel())


# =============================================================================
# Chase solver — hard obstacle constraints active
# Parameter vector: X0(3) | lookahead(2) | chase=1(1) | turkey(2) | uprev(2)
#                 | srects(4*N_SR) | dcents(2*N_DC)
# =============================================================================
def build_chase_solver(n_steps=N):
    n=n_steps; ns=3; nu=2; nsr=N_STATIC_RECTS; ndc=N_DYN_CIRCLES

    xs,ys,ths = ca.SX.sym('x'),ca.SX.sym('y'),ca.SX.sym('th')
    vls,vrs   = ca.SX.sym('vl'),ca.SX.sym('vr')
    vs=(vls+vrs)/2; ws=(vrs-vls)/L
    f = ca.Function('f',
        [ca.vertcat(xs,ys,ths), ca.vertcat(vls,vrs)],
        [ca.vertcat(xs,ys,ths)+DT*ca.vertcat(vs*ca.cos(ths),-vs*ca.sin(ths),ws)])

    X=ca.SX.sym('X',ns,n+1); U=ca.SX.sym('U',nu,n)
    PX0=ca.SX.sym('X0',ns); Pl=ca.SX.sym('l',2); Pt=ca.SX.sym('t',2)
    Pup=ca.SX.sym('up',nu); Psr=ca.SX.sym('sr',4*nsr); Pdc=ca.SX.sym('dc',2*ndc)
    p=ca.vertcat(PX0,Pl,Pt,Pup,Psr,Pdc)

    obj=0; geq=[X[:,0]-PX0]; giq=[]
    for k in range(n):
        xk=X[0,k]; yk=X[1,k]; thk=X[2,k]; vlk=U[0,k]; vrk=U[1,k]
        obj += W_XY_RUN*((xk-Pl[0])**2+(yk-Pl[1])**2)
        tdx=Pt[0]-xk; tdy=Pt[1]-yk; treg=ca.sqrt(tdx**2+tdy**2+1e-6)
        at=ca.atan2(-tdy/treg,tdx/treg)
        obj += W_FOV*ca.atan2(ca.sin(at-thk),ca.cos(at-thk))**2
        obj += W_OMEGA*((vrk-vlk)/L)**2
        upk=Pup if k==0 else U[:,k-1]
        obj += W_INPUT_DIFF*ca.sumsqr(U[:,k]-upk)
        geq.append(X[:,k+1]-f(X[:,k],U[:,k]))
        # Static rectangle hard constraints
        for i in range(nsr):
            xmn=Psr[4*i]; xmx=Psr[4*i+1]; ymn=Psr[4*i+2]; ymx=Psr[4*i+3]
            dx=ca.fmax(xmn-xk,ca.fmax(0.0,xk-xmx))
            dy=ca.fmax(ymn-yk,ca.fmax(0.0,yk-ymx))
            giq.append(ca.sqrt(dx**2+dy**2+1e-6)-STATIC_MARGIN)
        # Dynamic circle hard constraints
        for j in range(ndc):
            cx=Pdc[2*j]; cy=Pdc[2*j+1]
            giq.append(ca.sqrt((xk-cx)**2+(yk-cy)**2+1e-6)-DYN_R_TOTAL)
    obj += W_XY_TERM*((X[0,n]-Pl[0])**2+(X[1,n]-Pl[1])**2)

    ov=ca.vertcat(ca.reshape(X,-1,1),ca.reshape(U,-1,1))
    nd=ov.numel(); cs=ns*(n+1)
    lbx=-np.inf*np.ones(nd); ubx=np.inf*np.ones(nd)
    lbx[cs:]=V_MIN; ubx[cs:]=V_MAX
    geqc=ca.vertcat(*geq); giqc=ca.vertcat(*giq); ga=ca.vertcat(geqc,giqc)
    ne=geqc.numel(); ni=giqc.numel()
    lbg=np.concatenate([np.zeros(ne),np.zeros(ni)])
    ubg=np.concatenate([np.zeros(ne),np.inf*np.ones(ni)])

    slv=ca.nlpsol('chase_solver','ipopt',{'f':obj,'x':ov,'g':ga,'p':p},
        {'ipopt.print_level':0,'print_time':0,'ipopt.tol':1e-2,
         'ipopt.max_iter':50,'ipopt.warm_start_init_point':'yes',
         'ipopt.mu_strategy':'adaptive'})

    return dict(solver=slv,lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg,
                n_dec=nd,ctrl_start=cs,ns=ns,nu=nu,N=n,n_p=p.numel())

# =============================================================================
# Warm-start helpers
# =============================================================================
def _seed_guess(state, mpc):
    x0 = np.zeros(mpc['n_dec'])
    for k in range(mpc['N']+1):
        x0[k*mpc['ns']:k*mpc['ns']+mpc['ns']] = state
    return x0

def _shift_guess(sol_flat, mpc):
    ns,nu,n,cs = mpc['ns'],mpc['nu'],mpc['N'],mpc['ctrl_start']
    X=sol_flat[:cs].reshape(ns,n+1,order='F')
    U=sol_flat[cs:].reshape(nu,n,  order='F')
    x0=np.zeros(mpc['n_dec'])
    x0[:cs]=np.hstack([X[:,1:],X[:,-1:]]).flatten(order='F')
    x0[cs:]=np.hstack([U[:,1:],U[:,-1:]]).flatten(order='F')
    return x0

# =============================================================================
# Lookahead — forward-direction check
# =============================================================================
def _lookahead_point(robot_xy, robot_theta, waypoints, dist=LOOKAHEAD_DIST):
    """
    Project a point dist metres ahead along the A* path.
    Skips waypoints behind the robot to avoid U-turn demands in tight corridors.
    """
    if not waypoints:
        return np.array(robot_xy)
    pts = [np.array(wp, dtype=float) for wp in waypoints]
    fwd = np.array([np.cos(robot_theta), -np.sin(robot_theta)])

    start = len(pts)-1
    for i,pt in enumerate(pts):
        if np.dot(fwd, pt-robot_xy) >= 0:
            start=i; break

    remaining=dist
    for i in range(start,len(pts)-1):
        seg=pts[i+1]-pts[i]; sl=np.linalg.norm(seg)
        if sl<1e-6: continue
        if sl>=remaining: return pts[i]+seg*(remaining/sl)
        remaining-=sl
    return pts[-1]

# =============================================================================
# Controller
# =============================================================================
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode       = "idle"
        self._waypoints  = []
        self._target     = None
        self._turkey_pos = None
        self._chase_mode = False
        self._astar_approach_active = False

        self._pivot_target_hdg   = 0.0
        self._dist_history       = []
        self._obstacle_contact_n = 0
        self._solving            = False

        # Separate warm starts — scan and chase have different param layouts
        self._scan_guess  = None
        self._chase_guess = None
        self._u_prev      = np.zeros(2)

        print(f"[MPC] Building scan solver  (N={N}, no constraints)...")
        self._scan_mpc  = build_scan_solver(N)
        print(f"[MPC] Building chase solver (N={N}, rects={N_STATIC_RECTS}, circles={N_DYN_CIRCLES})...")
        self._chase_mpc = build_chase_solver(N)
        print(f"[MPC] Solvers ready. scan n_dec={self._scan_mpc['n_dec']}, "
              f"chase n_dec={self._chase_mpc['n_dec']}")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_turkey(self, tx, ty):
        self._turkey_pos = np.array([float(tx), float(ty)])
        self._chase_mode = True

    def clear_turkey(self):
        self._turkey_pos = None; self._chase_mode = False
        self._astar_approach_active = False

    # ── Protocol 6 ───────────────────────────────────────────────────────────

    async def _pub_reference(self, wps):
        await self.MPCFeedback_pub.publish(
            {"Id":"Reference","Cmd":[[float(x),float(y)] for x,y in wps]})

    async def _pub_clear(self):
        await self.MPCFeedback_pub.publish({"Id":"ClearTrajectory"})

    # ── Command receivers ─────────────────────────────────────────────────────

    async def Orient_CMD_Receiver(self, data):
        self._mode = "pivot"
        self._pivot_target_hdg = wrap(float(data[2]))

    async def Move_CMD_Receiver(self, waypoints, target_xy,
                                chase=False, astar_approach=False):
        if not waypoints: return

        dest_changed = (
            self._target is None
            or np.linalg.norm(np.array(target_xy)-np.array(self._target)) > 3.0
        )

        self._waypoints             = waypoints
        self._target                = target_xy
        self._astar_approach_active = astar_approach
        self._scan_guess            = None   # invalidate on new plan
        self._chase_guess           = None

        if dest_changed:
            self._dist_history       = []
            self._obstacle_contact_n = 0

        state = get_state(self._parent); rxy = state[:2]
        fwd   = np.array([np.cos(state[2]), -np.sin(state[2])])

        if not chase:
            pivot_wp = next(
                (np.array(wp, dtype=float) for wp in waypoints
                 if np.dot(fwd, np.array(wp, dtype=float)-rxy) > 0),
                np.array(waypoints[-1], dtype=float)
            )
            dx,dy = pivot_wp[0]-rxy[0], pivot_wp[1]-rxy[1]
            hdg   = np.arctan2(-dy, dx); err = wrap(hdg-state[2])
            if abs(np.degrees(err)) > PIVOT_THRESHOLD_DEG:
                self._pivot_target_hdg = hdg; self._mode = "pivot"
                print(f"[MPC] Pivot engaged: {np.degrees(err):.1f}deg")
                await self._pub_reference(waypoints); return

        self._mode = "chase" if chase else "trajectory"
        await self._pub_reference(waypoints)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def MPC_Main(self, logger=None):
        if self._solving: return
        self._solving = True
        try:
            await self._MPC_Main_inner(logger=logger)
        finally:
            self._solving = False

    async def _MPC_Main_inner(self, logger=None):

        # ── Pivot ──────────────────────────────────────────────────────────────
        if self._mode == "pivot":
            state = get_state(self._parent)
            err   = wrap(self._pivot_target_hdg-state[2])
            if abs(np.degrees(err)) < PIVOT_ALIGN_DEG:
                print("[MPC] Pivot complete.")
                self._mode       = "chase" if self._chase_mode else "trajectory"
                self._u_prev     = np.zeros(2)
                self._scan_guess = self._chase_guess = None
            else:
                omega = PIVOT_KP*err
                u = np.array([np.clip(-omega*L/2,V_MIN,V_MAX),
                               np.clip( omega*L/2,V_MIN,V_MAX)])
                await self._send(u)
                if logger:
                    logger.log(state=state, mode="pivot",
                        waypoint=tuple(self._target) if self._target is not None else (0,0),
                        turkey_pos=tuple(self._turkey_pos) if self._turkey_pos is not None else None,
                        edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                        vl=float(u[0]), vr=float(u[1]),
                        n_dynamic=len(self._parent._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            return

        if self._mode not in ("trajectory","chase"):
            return

        state = get_state(self._parent); rxy = state[:2]

        # ── Live A* → chase handoff ────────────────────────────────────────────
        if (self._mode=="trajectory" and self._astar_approach_active
                and self._turkey_pos is not None):
            dtk = np.linalg.norm(rxy-self._turkey_pos)
            if dtk <= CHASE_HANDOFF_DIST:
                print(f"[MPC] Handoff A*→chase dist={dtk:.1f}m")
                self._mode = "chase"; self._astar_approach_active = False
                self._dist_history = []; self._obstacle_contact_n = 0
                self._chase_guess  = None

        # ── Target ────────────────────────────────────────────────────────────
        if self._mode == "chase":
            if self._turkey_pos is None: self._mode="idle"; return
            ct=self._turkey_pos; ad=CHASE_DIST
        else:
            if self._target is None: self._mode="idle"; return
            ct=self._target; ad=FINAL_DIST

        dist_to_target = np.linalg.norm(rxy-ct)
        edt = self._parent.Map.min_rect_dist(*rxy)

        # ── Obstacle contact (chase only — scan trusts A*) ────────────────────
        if self._mode == "chase":
            if edt <= OBSTACLE_CONTACT_MARGIN:
                self._obstacle_contact_n += 1
            else:
                self._obstacle_contact_n = 0
            if self._obstacle_contact_n >= OBSTACLE_CONTACT_TICKS:
                print(f"[MPC] Chase obstacle contact replan (EDT={edt:.2f}m)")
                self._obstacle_contact_n = 0; self._dist_history = []
                self._chase_guess = None
                asyncio.create_task(self._parent.replan_to(self._target))
        else:
            self._obstacle_contact_n = 0   # not tracked in scan mode

        # ── Stall detection ───────────────────────────────────────────────────
        self._dist_history.append(dist_to_target)
        if len(self._dist_history) > STALL_WINDOW:
            self._dist_history.pop(0)
        stalled = (len(self._dist_history)==STALL_WINDOW
                   and (self._dist_history[0]-dist_to_target) < STALL_DELTA)

        # ── Arrival ───────────────────────────────────────────────────────────
        if dist_to_target < ad:
            print(f"[MPC] Arrived dist={dist_to_target:.2f}m")
            self._mode="idle"; self._dist_history=[]; self._obstacle_contact_n=0
            await self._send(np.zeros(2)); await self._pub_clear()
            if logger:
                logger.log(state=state, mode="idle", waypoint=tuple(ct),
                    turkey_pos=tuple(self._turkey_pos) if self._turkey_pos is not None else None,
                    edt_distance=edt, vl=0.0, vr=0.0,
                    n_dynamic=len(self._parent._dynamic_obs),
                    solve_time_ms=0.0, warm_start=False,
                    replan=False, n_waypoints=len(self._waypoints))
            return

        # ── Stall recovery ────────────────────────────────────────────────────
        if stalled:
            if dist_to_target < STALL_ARRIVAL_MULT*ad:
                print(f"[MPC] Arrived (stall, close) dist={dist_to_target:.2f}m")
                self._mode="idle"; self._dist_history=[]; self._obstacle_contact_n=0
                await self._send(np.zeros(2)); await self._pub_clear()
                return
            else:
                print(f"[MPC] Stall replan (dist={dist_to_target:.2f}m)")
                self._dist_history=[]; self._scan_guess=None; self._chase_guess=None
                self._obstacle_contact_n=0
                asyncio.create_task(self._parent.replan_to(self._target))

        # ── Select solver and build param vector ──────────────────────────────
        if self._mode == "chase":
            mpc  = self._chase_mpc
            lxy  = self._turkey_pos
            warm = self._chase_guess is not None
            if not warm:
                self._chase_guess = _seed_guess(state, mpc)
            guess = self._chase_guess

            srp, n_act = _pack_static_rects(self._parent.Map.Obstacle_Rects, rxy[0], rxy[1])
            dcp        = _pack_dyn_circles(self._parent._dynamic_obs)
            pvec = np.concatenate([state, lxy, self._turkey_pos, self._u_prev, srp, dcp])
        else:
            mpc  = self._scan_mpc
            lxy  = _lookahead_point(rxy, state[2], self._waypoints)
            warm = self._scan_guess is not None
            if not warm:
                self._scan_guess = _seed_guess(state, mpc)
            guess  = self._scan_guess
            n_act  = 0   # no constraints active in scan
            pvec   = np.concatenate([state, lxy, self._u_prev])

        # ── Solve ─────────────────────────────────────────────────────────────
        t0  = time.perf_counter()
        sol = mpc['solver'](x0=guess,
            lbx=mpc['lbx'], ubx=mpc['ubx'],
            lbg=mpc['lbg'], ubg=mpc['ubg'], p=pvec)
        ms  = (time.perf_counter()-t0)*1000

        flat = sol['x'].full().flatten()
        u    = flat[mpc['ctrl_start']:mpc['ctrl_start']+mpc['nu']]
        new_guess = _shift_guess(flat, mpc)

        if self._mode == "chase":
            self._chase_guess = new_guess
        else:
            self._scan_guess  = new_guess
        self._u_prev = u.copy()

        if logger:
            logger.log(state=state, mode=self._mode, waypoint=tuple(lxy),
                turkey_pos=tuple(self._turkey_pos) if self._turkey_pos is not None else None,
                edt_distance=edt, vl=float(u[0]), vr=float(u[1]),
                n_dynamic=len(self._parent._dynamic_obs),
                solve_time_ms=ms, warm_start=warm,
                replan=False, n_waypoints=len(self._waypoints))

        print(f"[MPC] mode={self._mode} dist={dist_to_target:.2f}m edt={edt:.2f}m "
              f"state=({state[0]:.2f},{state[1]:.2f},{np.degrees(state[2]):.1f}deg) "
              f"u=[{u[0]:.2f},{u[1]:.2f}] solve={ms:.1f}ms warm={warm} "
              f"{'rects_active='+str(n_act) if self._mode=='chase' else 'scan'}")

        await self._send(u)

    async def _send(self, u):
        await self.Drivetrain_pub.publish(
            {"Id":"System","Cmd":np.array(u,dtype=np.float32).tolist()})

    async def DriveTrain_Sender(self, logger=None):
        while True:
            t0 = asyncio.get_event_loop().time()
            await self.MPC_Main(logger=logger)
            await asyncio.sleep(max(0.0, DT-(asyncio.get_event_loop().time()-t0)))