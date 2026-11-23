"""Offline checks for the strategy pass: kickoff legality and clearance."""
import math
import sys
import types
from types import SimpleNamespace

# Stub the platform module so src.main imports without ROS.
booster = types.ModuleType("booster_agent_framework")


class AgentBase:
    pass


class AgentFeatures:
    pass


booster.AgentBase = AgentBase
booster.AgentFeatures = AgentFeatures
sys.modules["booster_agent_framework"] = booster

from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D
from src.main import _act_opp_kickoff, _act_opp_set_play, _act_our_kickoff, Phase
from src.param import (
    INACTIVITY_CLEAR_M, KICKOFF_BALL_MOVED_M, KICKOFF_PASS_X_M,
    KICKOFF_PASS_Y_M, KICK_POWER_RESTART_PASS,
)

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    if not ok:
        failures.append(name)


def make_ctx(ball_xy):
    return Context(
        now=0.0, dt=1.0 / 30.0, team_id=1,
        field=ADULT_FIELD_DIMENSIONS,
        ball=BallState(x=ball_xy[0], y=ball_xy[1]) if ball_xy else None,
    )


def make_player(pid, xy):
    p = SimpleNamespace(id=pid, pose=Pose2D(xy[0], xy[1], 0.0), action="",
                        is_fallen=False, is_penalized=False, is_kicking=False)
    p.calls = []
    p.attack = lambda *a, **k: p.calls.append(("attack",))
    p.kick = lambda d, pw: p.calls.append(("kick", d, pw))
    p.guard = lambda: p.calls.append(("guard",))
    p.stop = lambda: p.calls.append(("stop",))
    p.walk_to = lambda t, **kw: p.calls.append(("walk", t, kw))
    p.move_to_position = lambda t: p.calls.append(("move", t))
    return p


# ---------------------------------------------------------------- our kickoff
store = SimpleNamespace(prev_phase=None, kickoff_taker=None)
ctx = make_ctx((0.0, 0.0))
players = [make_player(1, (-1.5, 0.0)), make_player(2, (-4.0, 1.0)), make_player(3, (-4.0, -1.0))]
_act_our_kickoff(ctx, players, store)

taker = next(p for p in players if p.id == store.kickoff_taker)
kick_calls = [c for c in taker.calls if c[0] == "kick"]
check("kickoff: taker passes on first touch", len(kick_calls) == 1)
# Farthest teammate from our goal (-7, 0) is player 2 at (-4.0, 1.0).
expected_dir = math.atan2(1.0, -4.0)
check(
    "kickoff: pass goes to the teammate farthest from our goal",
    abs(kick_calls[0][1] - expected_dir) < 1e-9,
    f"dir={kick_calls[0][1]:.4f} expected={expected_dir:.4f}",
)
check(
    "kickoff: pass power is soft so the ball stays in play",
    kick_calls[0][2] == KICK_POWER_RESTART_PASS,
    f"power={kick_calls[0][2]}",
)
receiver = max(
    (p for p in players if p is not taker),
    key=lambda p: math.hypot(p.pose.x + 7.0, p.pose.y),
)
check(
    "kickoff: receiver holds position to meet the pass",
    receiver.action == "kickoff:receive" and any(c[0] == "stop" for c in receiver.calls),
    f"action={receiver.action}",
)
rest = [p for p in players if p is not taker and p is not receiver]
check(
    "kickoff: remaining robot takes the guard role",
    all(any(c[0] == "guard" for c in p.calls) for p in rest),
)

# No teammate available: fall back to the straight tap.
store_t = SimpleNamespace(prev_phase=Phase.OUR_KICKOFF, kickoff_taker=1)
solo = [make_player(1, (-1.5, 0.0))]
ctx_t = make_ctx((0.0, 0.0))
_act_our_kickoff(ctx_t, solo, store_t)
solo_kick = [c for c in solo[0].calls if c[0] == "kick"]
check(
    "kickoff: no teammate -> straight tap fallback",
    len(solo_kick) == 1 and abs(solo_kick[0][1]) < 1e-9
    and solo[0].action == "kickoff:tap",
    f"calls={[c[0] for c in solo[0].calls]}",
)

# Strike gate: a rolling ball must not be struck again.
store3 = SimpleNamespace(
    prev_phase=Phase.OUR_KICKOFF, kickoff_taker=taker.id, prev_ball=(0.25, 0.15),
)
ctx3 = make_ctx((0.3, 0.2))          # 0.36 m from the spot, still rolling
taker.is_kicking = True
taker.calls.clear()
_act_our_kickoff(ctx3, [taker], store3)
check(
    "kickoff: rolling ball is not re-struck",
    not any(c[0] == "kick" for c in taker.calls) and taker.action == "kickoff:striking",
    f"action={taker.action}",
)
store3.prev_ball = (0.3, 0.2)        # ball came to rest below the threshold
ctx4 = make_ctx((0.3, 0.2))
taker.calls.clear()
_act_our_kickoff(ctx4, [taker], store3)
check(
    "kickoff: stopped ball allows another strike",
    any(c[0] == "kick" for c in taker.calls),
)
taker.is_kicking = False

# ------------------------------------------------------------- opp kickoff
ctx3 = make_ctx((0.0, 0.0))
players3 = [make_player(1, (-6.0, 0.0)), make_player(2, (-3.0, 1.0)), make_player(3, (-2.0, -1.0))]
_act_opp_kickoff(ctx3, players3)
walk_targets = [c[1] for p in players3 for c in p.calls if c[0] == "walk"]
for t in walk_targets:
    d = math.hypot(t[0], t[1])
    check(
        f"opp kickoff: slot {t} clear",
        d >= INACTIVITY_CLEAR_M
        and math.hypot(t[0], t[1]) > ADULT_FIELD_DIMENSIONS.circle_radius,
        f"dist={d:.2f}",
    )

# ------------------------------------------------------------ opp set play
def run_wall(start_xy, ball_xy, frames=60):
    """Simulate the wall behaviour frame by frame for one robot."""
    ctx_n = make_ctx(ball_xy)
    p = make_player(1, start_xy)
    last = None
    for _ in range(frames):
        p.calls.clear()
        _act_opp_set_play(ctx_n, [p], SimpleNamespace())
        kinds = [c[0] for c in p.calls]
        if "stop" in kinds:
            last = ("stop", None)
            break
        step = next(c[1] for c in p.calls if c[0] == "walk")
        last = ("walk", step)
        p.pose = Pose2D(step[0], step[1], 0.0)
    return last, p.pose


# A single robot must end up ON the ball-to-goal lane, not scattered away.
last, end = run_wall((-1.0, 0.0), (0.0, 0.0))
check(
    "opp set play: solo robot blocks the lane",
    last[0] == "walk" and abs(end.y) < 1e-6 and end.x <= -INACTIVITY_CLEAR_M,
    f"end=({end.x:.2f},{end.y:.2f})",
)
_, mid = run_wall((-1.0, 0.0), (0.0, 0.0), frames=1)
check("opp set play: first frame already past 1.45 m",
      math.hypot(mid.x, mid.y) >= 1.45, f"dist={math.hypot(mid.x, mid.y):.2f}")

# A robot already clear moves onto the wall instead of standing still.
last2, end2 = run_wall((-5.0, 0.0), (0.0, 0.0))
check(
    "opp set play: clear robot joins the wall",
    last2[0] == "walk" and abs(end2.x) >= INACTIVITY_CLEAR_M - 1e-6 and abs(end2.y) < 1e-6,
    f"end=({end2.x:.2f},{end2.y:.2f})",
)

# Touchline ball: the wall point must stay inside the field.
last3, end3 = run_wall((0.5, 3.4), (0.0, 3.8))
check(
    "opp set play: touchline wall stays in field",
    abs(end3.y) <= ADULT_FIELD_DIMENSIONS.width / 2.0 - 0.3 + 1e-6
    and math.hypot(end3.x, end3.y - 3.8) >= INACTIVITY_CLEAR_M - 1e-6,
    f"end=({end3.x:.2f},{end3.y:.2f})",
)

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
