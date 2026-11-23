"""Checks for attack(): shot aim, staging point, and strike alignment."""
import math
import sys
from types import SimpleNamespace

from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState
from src.param import CHASE_FIELD_MARGIN_M, KICK_POWER_DEFAULT, GOAL_CORNER_INSET_M
from src.player import Player, _behind_ball
from src.utils import clamp_inside_field

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    if not ok:
        failures.append(name)


FIELD = ADULT_FIELD_DIMENSIONS
goal_x = FIELD.length / 2.0
ball_xy = (2.0, 1.0)


def make(our_xy, ball=ball_xy, opponents=()):
    ctx = Context(
        now=1.0, dt=1.0 / 30.0, team_id=0, field=FIELD,
        ball=BallState(x=ball[0], y=ball[1]),
        teammates={1: RobotState(player_id=1, pose=Pose2D(*our_xy, 0.0))},
        opponents={
            i + 1: RobotState(player_id=i + 1, pose=Pose2D(*xy, 0.0))
            for i, xy in enumerate(opponents)
        },
    )
    p = Player(1, SimpleNamespace(team_id=0), None)
    p.context = ctx
    p.calls = []
    p.walk_to = lambda target, **kw: p.calls.append(("walk", target, kw))
    p.kick = lambda d, power: p.calls.append(("kick", d, power))
    return p


# 1. Default aim is the goal center candidate when nothing blocks it.
p = make((0.0, 0.0))
direction, power = p.plan_kick()
expected = math.atan2(0.0 - ball_xy[1], goal_x - ball_xy[0])
check("aim: open goal -> center", abs(direction - expected) < 1e-9,
      f"dir={direction:.4f} expected={expected:.4f}")
check("aim: shot power is the hard default", power == KICK_POWER_DEFAULT,
      f"power={power}")

# 2. A covered center moves the aim to a corner, with a real post margin.
p = make((0.0, 0.0), opponents=[(6.8, 0.0)])
direction, _ = p.plan_kick()
corner_y = FIELD.goal_width / 2.0 - GOAL_CORNER_INSET_M
expected = math.atan2(corner_y - ball_xy[1], goal_x - ball_xy[0])
check("aim: covered center -> corner with margin",
      abs(direction - expected) < 1e-9,
      f"dir={direction:.4f} expected={expected:.4f}")
check("aim: corner target keeps clear of the post",
      abs(FIELD.goal_width / 2.0 - corner_y - GOAL_CORNER_INSET_M) < 1e-9,
      f"margin={FIELD.goal_width / 2.0 - corner_y:.2f}")

# 3. Custom target changes the aim.
target = (goal_x, 1.5)
p = make((0.0, 0.0))
direction, _ = p.plan_kick(target)
expected = math.atan2(1.5 - ball_xy[1], goal_x - ball_xy[0])
check("aim: custom target is honoured", abs(direction - expected) < 1e-9,
      f"dir={direction:.4f}")

# 4. Out of range: walk to the clamped staging point, no strike.
p = make((0.0, 0.0))
p.attack(target)
kind, chase, kw = p.calls[-1]
expected_point = clamp_inside_field(
    p.context, *_behind_ball(ball_xy[0], ball_xy[1], target, 0.35),
    margin=CHASE_FIELD_MARGIN_M,
)
check("chase: staging point behind the ball",
      kind == "walk" and math.dist(chase, expected_point) < 1e-9,
      f"target={chase}")
check("chase: robot avoidance enabled", kw.get("avoid_robots") is True)

# 5. In range and behind the ball: strike with the planned direction.
p = make((1.7, 1.0))
p.attack(target)
kind, direction, power = p.calls[-1]
check("strike: aligned robot kicks",
      kind == "kick" and abs(direction - expected) < 1e-9,
      f"call={p.calls[-1]}")

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
