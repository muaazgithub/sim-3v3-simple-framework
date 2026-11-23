"""Offline checks for the own-goal fix: danger-zone clearance and clamped chase."""
import math
import sys
import types
from types import SimpleNamespace

booster = types.ModuleType("booster_agent_framework")


class AgentBase:
    pass


booster.AgentBase = AgentBase
booster.AgentFeatures = type("AgentFeatures", (), {})
sys.modules["booster_agent_framework"] = booster

from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState
from src.param import (
    CHASE_FIELD_MARGIN_M, DEFENSE_CLEAR_LATERAL_M, DEFENSE_CLEAR_RANGE_M,
    DEFENSE_DANGER_DEPTH_M, KICK_POWER_BACKFIELD,
)
from src.player import Player

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    if not ok:
        failures.append(name)


FIELD = ADULT_FIELD_DIMENSIONS
half_l = FIELD.length / 2.0


def make_ctx(ball_xy, our_xy):
    return Context(
        now=0.0, dt=1.0 / 30.0, team_id=1, field=FIELD,
        ball=BallState(x=ball_xy[0], y=ball_xy[1]) if ball_xy else None,
        teammates={1: RobotState(player_id=1, pose=Pose2D(*our_xy, 0.0))},
        opponents={},
    )


def clear_direction(ball_xy):
    p = Player(1, SimpleNamespace(team_id=1), None)
    p.context = make_ctx(ball_xy, (0.0, 0.0))
    plan = p.plan_kick()
    return plan


# 1. Danger-zone kicks are forward clearances, never backward.
plan = clear_direction((-6.5, 0.0))
expected = math.atan2(DEFENSE_CLEAR_LATERAL_M, DEFENSE_CLEAR_RANGE_M)
check("danger: kicks forward with flank bias",
      plan is not None
      and abs(plan[0] - expected) < 1e-9
      and plan[1] == KICK_POWER_BACKFIELD,
      f"dir={plan[0]:.4f} expected={expected:.4f}")

plan = clear_direction((-6.5, -1.0))
expected = math.atan2(-DEFENSE_CLEAR_LATERAL_M, DEFENSE_CLEAR_RANGE_M)
check("danger: lower ball clears to the lower flank",
      abs(plan[0] - expected) < 1e-9, f"dir={plan[0]:.4f}")

all_forward = True
worst = None
for xi in range(-70, -35, 5):
    for yi in range(-40, 41, 10):
        ball = (xi / 10.0, yi / 10.0)
        direction, _ = clear_direction(ball)
        if math.cos(direction) <= 0.0:
            all_forward = False
            worst = (ball, direction)
check("danger: sweep of deep balls never kicks backward", all_forward,
      f"worst={worst}")

# 1b. An explicit power override applies to the danger-zone clearance.
p2 = Player(1, SimpleNamespace(team_id=1), None)
p2.context = make_ctx((-6.5, 0.0), (0.0, 0.0))
direction, forced = p2.plan_kick(power=5.0)
check("danger: power override is honoured",
      abs(direction - math.atan2(DEFENSE_CLEAR_LATERAL_M, DEFENSE_CLEAR_RANGE_M)) < 1e-9
      and forced == 5.0, f"power={forced}")

# 2. Ball outside the danger zone still uses goal aiming.
plan = clear_direction((-2.0, 1.0))
check("normal: mid-field ball still aims at the goal",
      plan is not None and abs(plan[0] - math.atan2(-1.0, half_l - (-2.0))) < 1e-6,
      f"dir={plan[0]:.4f}")

# 3. The chase target is always clamped inside the field, and in the danger
#    zone it sits upfield of the ball so contact pushes the ball away.
p = Player(1, SimpleNamespace(team_id=1), None)
recorded = []
p.walk_to = lambda target, **kw: recorded.append((target, kw))
ball = (-6.8, 0.5)
p.context = make_ctx(ball, (-3.0, 0.5))   # far enough to stay in chase mode
p.attack()
chase = recorded[-1][0]
margin = CHASE_FIELD_MARGIN_M
check("danger: chase point inside the field",
      chase[0] >= -half_l + margin - 1e-9 and chase[0] <= half_l - margin + 1e-9
      and abs(chase[1]) <= FIELD.width / 2.0 - margin + 1e-9,
      f"chase=({chase[0]:.3f},{chase[1]:.3f})")
check("danger: chase point upfield of the ball", chase[0] > ball[0],
      f"chase_x={chase[0]:.3f} ball_x={ball[0]:.3f}")

# 4. Non-danger chase is the usual behind-the-ball staging, also clamped.
p2 = Player(1, SimpleNamespace(team_id=1), None)
recorded2 = []
p2.walk_to = lambda target, **kw: recorded2.append((target, kw))
ball = (5.9, 3.9)   # near the opponent corner: raw chase would leave the field
p2.context = make_ctx(ball, (4.0, 3.0))
p2.attack()
chase2 = recorded2[-1][0]
check("normal: chase point near opponent corner stays in field",
      chase2[0] <= half_l - margin + 1e-9 and chase2[1] <= FIELD.width / 2.0 - margin + 1e-9,
      f"chase=({chase2[0]:.3f},{chase2[1]:.3f})")

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
