"""Offline checks for the opponent set-play wall."""
import math
import sys
import types
from types import SimpleNamespace

booster = types.ModuleType("booster_agent_framework")
booster.AgentBase = type("AgentBase", (), {})
booster.AgentFeatures = type("AgentFeatures", (), {})
sys.modules["booster_agent_framework"] = booster

from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D
from src.main import _act_opp_set_play
from src.param import (
    INACTIVITY_CLEAR_M, OPP_SET_GUARD_GATE_M, OPP_SET_PLAY_FIELD_MARGIN_M,
    OPP_SET_WALL_FAR_M, OPP_SET_WALL_NEAR_M, OPP_SET_WALL_SPREAD_M,
)

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    if not ok:
        failures.append(name)


FIELD = ADULT_FIELD_DIMENSIONS
half_w = FIELD.width / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M


def make_ctx(ball_xy):
    return Context(
        now=0.0, dt=1.0 / 30.0, team_id=1, field=FIELD,
        ball=BallState(x=ball_xy[0], y=ball_xy[1]) if ball_xy else None,
    )


def make_player(pid, xy):
    p = SimpleNamespace(id=pid, pose=Pose2D(xy[0], xy[1], 0.0), action="",
                        is_fallen=False, is_penalized=False)
    p.calls = []
    p.guard = lambda: p.calls.append(("guard",))
    p.stop = lambda: p.calls.append(("stop",))
    p.walk_to = lambda t, **kw: p.calls.append(("walk", t, kw))
    p.move_to_position = lambda t: p.calls.append(("move", t))
    return p


def run(ball_xy, player_xys):
    ctx = make_ctx(ball_xy)
    players = [make_player(i + 1, xy) for i, xy in enumerate(player_xys)]
    _act_opp_set_play(ctx, players, SimpleNamespace())
    return players


def lane_perp(ball_xy, t):
    gx, gy = -FIELD.length / 2.0, 0.0
    lx, ly = gx - ball_xy[0], gy - ball_xy[1]
    ll = math.hypot(lx, ly)
    px, py = t[0] - ball_xy[0], t[1] - ball_xy[1]
    return abs(px * ly - py * lx) / ll


def wall_targets(p):
    return [c[1] for c in p.calls if c[0] == "walk"]


# --- A. Throw-in on our flank: two lane blockers plus a keeper.
players = run((-2.0, 4.2), [(-3.0, 0.0), (-5.0, 1.0), (-1.0, 1.0)])
guards = [p for p in players if any(c[0] == "guard" for c in p.calls)]
check("A: one keeper guards the goal", len(guards) == 1)
walkers = [p for p in players if not any(c[0] == "guard" for c in p.calls)]
targets = [t for p in walkers for t in wall_targets(p)]
check("A: two wall blockers", len(targets) == 2, f"targets={targets}")
ok_dist = all(math.hypot(t[0] + 2.0, t[1] - 4.2) >= INACTIVITY_CLEAR_M - 1e-9 for t in targets)
check("A: every wall point beyond the inactivity radius", ok_dist)
ok_lane = all(lane_perp((-2.0, 4.2), t) < 0.05 for t in targets)
check("A: both blockers sit on the goal lane", ok_lane,
      f"perps={[round(lane_perp((-2.0, 4.2), t), 3) for t in targets]}")
ok_in = all(abs(t[0]) <= FIELD.length / 2.0 - 0.3 + 1e-9 and abs(t[1]) <= half_w + 1e-9
            for t in targets)
check("A: wall points inside the field", ok_in)
sep = math.dist(targets[0], targets[1]) if len(targets) == 2 else 0.0
check("A: blockers do not stack on one spot", sep >= OPP_SET_WALL_FAR_M - OPP_SET_WALL_NEAR_M - 0.05,
      f"sep={sep:.2f}")

# --- B. Ball on our goal line: no keeper spot exists, everyone spreads legally.
players = run((-6.7, 0.0), [(-5.0, 0.5), (-6.0, -1.0), (-4.0, 0.0)])
check("B: nobody guards when the goal line is the ball's spot",
      not any(any(c[0] == "guard" for c in p.calls) for p in players))
all_t = [t for p in players for t in wall_targets(p)]
ok = all(math.hypot(t[0] + 6.7, t[1]) >= INACTIVITY_CLEAR_M - 1e-9 for t in all_t)
check("B: every spread point clears the inactivity radius", ok, f"targets={all_t}")
ok = all(abs(t[0]) <= FIELD.length / 2.0 - 0.3 + 1e-9 and abs(t[1]) <= half_w + 1e-9
         for t in all_t)
check("B: spread points inside the field", ok)

# --- C. A robot inside the clearance zone exits radially first.
players = run((0.0, 0.0), [(0.8, 0.0)])
t1 = wall_targets(players[0])[0]
check("C: trapped robot exits radially outward",
      abs(t1[0] - OPP_SET_WALL_NEAR_M) < 1e-9 and abs(t1[1]) < 1e-9,
      f"target=({t1[0]:.2f},{t1[1]:.2f})")

# --- C2. With a full team, the trapped robot may instead take the keeper
#     role, whose walk home also increases ball distance.
players = run((0.0, 0.0), [(0.8, 0.0), (-5.0, 0.0), (-6.5, 0.0)])
p1 = players[0]
got_guard = any(c[0] == "guard" for c in p1.calls)
t1 = wall_targets(p1)
check("C2: trapped robot exits by walking home or radially",
      got_guard or (t1 and math.hypot(t1[0][0], t1[0][1]) >= INACTIVITY_CLEAR_M - 1e-9),
      f"guard={got_guard} targets={t1}")


# --- Gate. The keeper posts only when the goal is at least
#     OPP_SET_GUARD_GATE_M from the ball: inside that radius a parked keeper
#     would sit within the M-05 3 m/10 s trap while the opponent stalls.
inside_gate = -FIELD.length / 2.0 + OPP_SET_GUARD_GATE_M - 0.2
outside_gate = -FIELD.length / 2.0 + OPP_SET_GUARD_GATE_M + 0.2
players = run((inside_gate, 0.0), [(-4.0, 0.0), (-5.0, 1.0), (-6.0, -1.0)])
check("gate: no keeper when the ball sits inside the gate radius",
      not any(any(c[0] == "guard" for c in p.calls) for p in players))
players = run((outside_gate, 0.0), [(-4.0, 0.0), (-5.0, 1.0), (-6.0, -1.0)])
check("gate: keeper posts once the ball clears the gate radius",
      any(any(c[0] == "guard" for c in p.calls) for p in players))

# --- D. No ball visible: everyone stands still.
players = run(None, [(0.0, 0.0), (-2.0, 0.0), (-4.0, 0.0)])
check("D: missing ball stops everyone",
      all(any(c[0] == "stop" for c in p.calls) for p in players))

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
