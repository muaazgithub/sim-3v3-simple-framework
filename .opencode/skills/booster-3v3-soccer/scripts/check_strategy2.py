"""Offline checks for set-play dispatch, shot aiming, guard, and support."""
import math
import sys
import types
from types import SimpleNamespace

booster = types.ModuleType("booster_agent_framework")


class AgentBase:
    pass


class AgentFeatures:
    pass


booster.AgentBase = AgentBase
booster.AgentFeatures = AgentFeatures
sys.modules["booster_agent_framework"] = booster

from src.framework.types import (
    ADULT_FIELD_DIMENSIONS, BallState, Context, GameControlState, GameState,
    KICKING_TEAM_NONE, Pose2D, RobotState, SetPlay, TeamState,
)
from src.main import (_act_our_set_play, _act_normal, _match_urgency,
                      get_phase, Phase)
from src.param import (
    GOAL_CORNER_INSET_M, INACTIVITY_CLEAR_M, KICKOFF_PASS_X_M, KICKOFF_PASS_Y_M,
    KICK_POWER_RESTART_PASS, SET_PLAY_PASS_AHEAD_M,
    SET_PLAY_HOLD_AT_BALL_S, SET_PLAY_PASS_EDGE_MARGIN_M,
    SET_PLAY_READY_MAX_S, SET_PLAY_SETTLE_ARRIVE_M, SET_PLAY_SETTLE_M,
    SUPPORT_DEFEND_DEPTH_M, URGENT_ATTACK_SECS, URGENT_DEFEND_SECS,
    URGENCY_ATTACK, URGENCY_DEFEND, URGENCY_NORMAL,
    THROW_IN_AHEAD_M, THROW_IN_INWARD_M,
    THROW_IN_POWER_MIN, THROW_IN_RAMP_M, GOAL_KICK_POWER,
)
from src.player import Player

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    if not ok:
        failures.append(name)


FIELD = ADULT_FIELD_DIMENSIONS


def make_ctx(ball_xy, set_play=SetPlay.NONE, opponents=(), now=0.0):
    game = GameControlState(state=GameState.PLAYING, set_play=set_play)
    return Context(
        now=now, dt=1.0 / 30.0, team_id=1, field=FIELD,
        game=game,
        ball=BallState(x=ball_xy[0], y=ball_xy[1]) if ball_xy else None,
        teammates={1: RobotState(player_id=1, pose=Pose2D(0, 0, 0))},
        opponents={i + 1: RobotState(player_id=i + 1, pose=Pose2D(*xy, 0.0))
                   for i, xy in enumerate(opponents)},
    )


def make_player(pid, xy):
    p = SimpleNamespace(id=pid, pose=Pose2D(xy[0], xy[1], 0.0), action="",
                        is_fallen=False, is_penalized=False, is_kicking=False)
    p.calls = []
    p.attack = lambda *a, **k: p.calls.append(("attack", a, k))
    p.kick = lambda d, pw: p.calls.append(("kick", d, pw))
    p.guard = lambda: p.calls.append(("guard",))
    p.stop = lambda: p.calls.append(("stop",))
    p.support = lambda: p.calls.append(("support",))
    p.walk_to = lambda t, **kw: p.calls.append(("walk", t, kw))
    p.move_to_position = lambda t: p.calls.append(("move", t))
    return p


def run_set_play(ball_xy, set_play, store_overrides, players, now=None,
                 entering=False):
    if now is None:
        now = SET_PLAY_READY_MAX_S + 1.0
    ctx = make_ctx(ball_xy, set_play, now=now)
    store_dict = {
        "set_play_spot": None, "set_play_taker": None,
        "prev_phase": None if entering else Phase.OUR_SET_PLAY,
        "setplay_entered_at": None if entering else 0.0,
        "setplay_settled_at": None if entering else 0.0,
    }
    store_dict.update(store_overrides)
    store = SimpleNamespace(**store_dict)
    _act_our_set_play(ctx, players, store)
    return ctx, store


# ---------------------------------------------------- goal kick: pass, no punt
players = [make_player(1, (-5.0, 0.3)), make_player(2, (-2.0, 1.0)),
           make_player(3, (1.5, -1.5))]
run_set_play((-6.0, 0.0), SetPlay.GOAL_KICK, {}, players)
taker, receiver, helper = players[0], players[2], players[1]
kick = [c for c in taker.calls if c[0] == "kick"]
check("goal kick: taker passes instead of punting",
      len(kick) == 1 and taker.action == "goalkick:pass",
      f"action={taker.action} calls={[c[0] for c in taker.calls]}")
expected_dir = math.atan2(-1.5 - 0.0, 1.5 - (-6.0))
check("goal kick: pass heads for the most advanced teammate",
      abs(kick[0][1] - expected_dir) < 1e-9,
      f"dir={kick[0][1]:.4f} expected={expected_dir:.4f}")
check("goal kick: measured pass power", kick[0][2] == GOAL_KICK_POWER,
      f"power={kick[0][2]}")
check("goal kick: receiver keeps his spot for the pass",
      not any(c[0] in ("move", "guard") for c in receiver.calls))
check("goal kick: another robot still guards",
      any(c[0] == "guard" for c in helper.calls))

# Only the taker active: fall back to the safe softer clearance.
players = [make_player(1, (-5.0, 0.3))]
run_set_play((-6.0, 0.0), SetPlay.GOAL_KICK, {}, players)
fallback = [c for c in players[0].calls if c[0] == "attack"]
check("goal kick: no receiver -> soft staged clearance",
      fallback and fallback[0][2].get("power") == GOAL_KICK_POWER,
      f"calls={[c[0] for c in players[0].calls]}")

# -------------------------------------- attacking indirect: pass to receivers
spot = (3.0, 0.5)
players = [make_player(1, (2.5, 0.5)), make_player(2, (-2.0, 0.0)), make_player(3, (0.5, -1.0))]
ctx, store = run_set_play(spot, SetPlay.INDIRECT_FREE_KICK, {}, players)
taker = players[0]
kicks = [c for c in taker.calls if c[0] == "kick"]
check("indirect: taker passes, not shoots", len(kicks) == 1)
pass_dir = kicks[0][1]
deep = (FIELD.length / 2 - FIELD.penalty_dist, 0.0)
receiver_spots = [c[1] for p in players[1:] for c in p.calls if c[0] == "move"]
expected_dirs = [
    math.atan2(t[1] - spot[1], t[0] - spot[0]) for t in receiver_spots
]
check(
    "indirect: attacking spot passes to a receiver spot",
    any(abs(pass_dir - d) < 1e-9 for d in expected_dirs) and math.cos(pass_dir) > 0.0,
    f"dir={pass_dir:.4f} expected={expected_dirs}",
)
check("indirect: restart pass is soft", kicks[0][2] == KICK_POWER_RESTART_PASS)
check("indirect: passer is remembered", store.set_play_taker == taker.id)
receivers = [c[1] for p in players[1:] for c in p.calls if c[0] == "move"]
check("indirect: both receivers take up spots", len(receivers) == 2,
      f"spots={receivers}")
check("indirect: receiver spots clear of the ball",
      all(math.hypot(t[0] - spot[0], t[1] - spot[1]) >= INACTIVITY_CLEAR_M - 1e-6
          for t in receivers))

# Ball moved past the spot: a teammate (never the passer) takes the 2nd touch.
players = [make_player(1, (2.5, 0.5)), make_player(2, (4.2, 0.6)), make_player(3, (-3.0, 0.0))]
ctx, store = run_set_play((3.8, 0.5), SetPlay.INDIRECT_FREE_KICK,
                          {"prev_phase": Phase.OUR_SET_PLAY,
                           "set_play_spot": spot,
                           "set_play_taker": 1}, players)
striker = players[1]
taker = players[0]
check("indirect: teammate attacks the live ball", any(c[0] == "attack" for c in striker.calls))
check("indirect: passer never re-touches", not any(c[0] in ("kick", "attack") for c in taker.calls))
check("indirect: everyone else supports", all(
    any(c[0] == "support" for c in p.calls) for p in (players[0], players[2])))

# ------------------------------------------------- attacking corner: pass, not shoot
corner = (FIELD.length / 2 - 0.1, FIELD.width / 2 - 0.1)
players = [make_player(1, (6.0, 4.0)), make_player(2, (4.0, 1.0)), make_player(3, (-2.0, 0.0))]
ctx, store = run_set_play(corner, SetPlay.CORNER_KICK, {}, players)
taker = players[0]
kicks = [c for c in taker.calls if c[0] == "kick"]
check("corner: taker passes instead of shooting", len(kicks) == 1)
pass_dir = kicks[0][1]
check(
    "corner: pass heads for the penalty-spot receiver",
    abs(pass_dir - math.atan2(deep[1] - corner[1], deep[0] - corner[0])) < 1e-9,
    f"dir={pass_dir:.4f}",
)
check("corner: pass is soft", kicks[0][2] == KICK_POWER_RESTART_PASS)
receivers = [c[1] for p in players[1:] for c in p.calls if c[0] == "move"]
check("corner: both receivers take up spots", len(receivers) == 2,
      f"spots={receivers}")
check("corner: receiver spots clear of the ball",
      all(math.hypot(t[0] - corner[0], t[1] - corner[1]) >= INACTIVITY_CLEAR_M - 1e-6
          for t in receivers))

# ------------------------------------- defensive throw-in: clear, do not pass
players = [make_player(1, (-2.5, 4.0)), make_player(2, (-4.0, 0.0)), make_player(3, (2.0, -2.0))]
run_set_play((-2.0, 4.2), SetPlay.THROW_IN, {}, players)
kick_calls = [c for c in players[0].calls if c[0] == "kick"]
check("defensive throw-in: taker plays a directed throw",
      len(kick_calls) == 1 and players[0].action == "throwin:clear",
      f"action={players[0].action} calls={[c[0] for c in players[0].calls]}")
check("defensive throw-in: someone guards",
      any(any(c[0] == "guard" for c in p.calls) for p in players[1:]))

# ------------------------------------------------------- guard tracking
p = Player(2, SimpleNamespace(team_id=1), None)
recorded = []
p.walk_to = lambda target, **kw: recorded.append((target, kw))

ctx = make_ctx((-5.0, 0.5))
ctx.teammates[2] = RobotState(player_id=2, pose=Pose2D(-6.6, 0.0, 0.0))
p.context = ctx
p.guard()
target, kw = recorded[-1]
home_x = -FIELD.length / 2 + FIELD.goal_area_length / 2
check("guard: threatened tracks the mouth", p.action == "guard:track"
      and abs(target[0] - home_x) < 1e-9 and target[1] > 0.0,
      f"target=({target[0]:.2f},{target[1]:.2f})")
check("guard: faces the ball", abs(kw.get("face", 0.0) - math.atan2(0.5, 1.6)) < 1e-6)

ctx = make_ctx((3.0, 0.0))
ctx.teammates[2] = RobotState(player_id=2, pose=Pose2D(-6.0, 0.0, 0.0))
p.context = ctx
p.guard()
target, _ = recorded[-1]
check("guard: calm ball returns home", p.action == "guard:home"
      and abs(target[0] - home_x) < 1e-9 and abs(target[1]) < 1e-9)

# Ball at the mouth: the keeper sweeps it instead of watching.
swept = []
p.attack = lambda: swept.append(True)
p._guard_prev_ball = None             # fresh guard: no stale velocity
ctx = make_ctx((-6.0, 0.3))
ctx.teammates[2] = RobotState(player_id=2, pose=Pose2D(-6.4, 0.0, 0.0))
p.context = ctx
p.guard()
check("guard: ball at the mouth is swept",
      p.action == "guard:clear" and swept, f"action={p.action}")

# Shot prediction: meet the ball where it crosses the keeper's line.
ctx = make_ctx((-4.0, 0.2))
ctx.teammates[2] = RobotState(player_id=2, pose=Pose2D(-6.5, 0.0, 0.0))
p.context = ctx
p._guard_prev_ball = (-3.7, 0.1)      # moving 9 m/s toward goal, 3 m/s wide
p.guard()
target, _ = recorded[-1]
expected_y = 0.2 + 3.0 * ((-6.5 + 4.0) / -9.0)
check("guard: predicts the shot crossing point",
      abs(target[1] - expected_y) < 1e-6 and target[1] > 0.2,
      f"target_y={target[1]:.3f} expected={expected_y:.3f}")

# ------------------------------------------------- support near own goal
p2 = Player(3, SimpleNamespace(team_id=1), None)
moved_to = []
p2.move_to_position = lambda t: moved_to.append(t)
ctx = make_ctx((-6.0, 0.0))
ctx.teammates[3] = RobotState(player_id=3, pose=Pose2D(-5.0, 0.0, 0.0))
p2.context = ctx
p2.support()
t = moved_to[-1]
d = math.hypot(t[0] + 6.0, t[1])
check("support: goal-kick ball keeps clear distance", d >= INACTIVITY_CLEAR_M - 1e-6,
      f"target=({t[0]:.2f},{t[1]:.2f}) dist={d:.2f}")

# ------------------------------------------------------- shot aiming
def shot_case(opponents):
    p3 = Player(1, SimpleNamespace(team_id=1), None)
    ctx = make_ctx((4.0, 0.5), opponents=opponents)
    ctx.teammates[1] = RobotState(player_id=1, pose=Pose2D(4.0, 0.5, 0.0))
    p3.context = ctx
    return p3._choose_shot_target()


goal_x = FIELD.length / 2
t = shot_case([(6.8, 0.0)])
corner_y = FIELD.goal_width / 2 - GOAL_CORNER_INSET_M
check("aim: covered center -> corner", abs(t[0] - goal_x) < 1e-9 and abs(abs(t[1]) - corner_y) < 1e-9,
      f"target=({t[0]:.2f},{t[1]:.2f})")
t = shot_case([])
check("aim: empty field -> goal center", abs(t[0] - goal_x) < 1e-9 and abs(t[1]) < 1e-9,
      f"target=({t[0]:.2f},{t[1]:.2f})")
t = shot_case([(6.8, 0.0), (6.0, 0.9), (5.5, -0.5)])
check("aim: all lanes covered -> center fallback",
      abs(t[1]) < 1e-9 and goal_x <= t[0] <= goal_x + 0.25,
      f"target=({t[0]:.2f},{t[1]:.2f})")

# ------------------------------------ defensive throw-in: power ramps with depth
from src.param import KICK_POWER_BACKFIELD  # noqa: E402


def throw_in_case(ball_xy, **overrides):
    players = [make_player(1, (0.0, 4.0)), make_player(2, (-4.0, 0.0)),
               make_player(3, (2.0, -2.0))]
    ctx, store = run_set_play(ball_xy, SetPlay.THROW_IN,
                              {**overrides}, players)
    return players[0], ctx, store


def expected_throw_in(ball_xy):
    ratio = max(0.0, -ball_xy[0]) / THROW_IN_RAMP_M
    power = THROW_IN_POWER_MIN + (KICK_POWER_BACKFIELD - THROW_IN_POWER_MIN) * min(ratio, 1.0)
    sign = 1.0 if ball_xy[1] >= 0.0 else -1.0
    target = (ball_xy[0] + THROW_IN_AHEAD_M, ball_xy[1] - THROW_IN_INWARD_M * sign)
    return power, math.atan2(target[1] - ball_xy[1], target[0] - ball_xy[0])


taker, _, _ = throw_in_case((-0.5, 4.2))
kick = [c for c in taker.calls if c[0] == "kick"][0]
power, direction = expected_throw_in((-0.5, 4.2))
check("throwin: near halfway is soft",
      abs(kick[2] - power) < 1e-9 and abs(kick[1] - direction) < 1e-9
      and taker.action == "throwin:clear",
      f"power={kick[2]:.2f} expected={power:.2f} dir={kick[1]:.3f}")

taker, _, _ = throw_in_case((-3.0, 4.2))
kick = [c for c in taker.calls if c[0] == "kick"][0]
power, direction = expected_throw_in((-3.0, 4.2))
check("throwin: deeper means stronger",
      abs(kick[2] - power) < 1e-9 and power > THROW_IN_POWER_MIN + 4.0,
      f"power={kick[2]:.2f} expected={power:.2f}")

taker, _, _ = throw_in_case((-4.0, 4.2))
check("throwin: deep in our end uses the staged clearance",
      any(c[0] == "attack" for c in taker.calls) and taker.action == "attack",
      f"action={taker.action}")

players = [make_player(1, (0.0, 4.0)), make_player(2, (-4.0, 0.0)),
           make_player(3, (2.0, -2.0))]
players[0].is_kicking = True
ctx, store = run_set_play((-0.5, 4.2), SetPlay.THROW_IN,
                          {"prev_ball": (-0.2, 4.2)}, players)
taker = players[0]
check("throwin: rolling ball is not re-struck",
      not any(c[0] == "kick" for c in taker.calls)
      and taker.action == "throwin:striking",
      f"action={taker.action}")

# The aim point must sit inside the field on BOTH touchlines - an inverted
# inward sign once aimed every defensive throw-in out of play.
for ball_xy in [(-0.5, 4.5), (-0.5, -4.5), (-2.5, 4.5), (-2.5, -4.5)]:
    players = [make_player(1, (0.0, 4.0)), make_player(2, (-4.0, 0.0)),
               make_player(3, (2.0, -2.0))]
    run_set_play(ball_xy, SetPlay.THROW_IN, {}, players)
    taker = next(p for p in players if any(c[0] == "kick" for c in p.calls))
    kick = [c for c in taker.calls if c[0] == "kick"][0]
    direction = kick[1]
    into_field = (direction < 0.0) if ball_xy[1] > 0.0 else (direction > 0.0)
    check(f"throwin: aim from y={ball_xy[1]} points into the field",
          into_field, f"dir={direction:.3f}")
    target_y = ball_xy[1] + math.tan(direction) * THROW_IN_AHEAD_M
    check(f"throwin: aim from y={ball_xy[1]} stays inside the field",
          abs(target_y) <= ADULT_FIELD_DIMENSIONS.width / 2.0 - 0.3 + 1e-6,
          f"target_y={target_y:.2f}")


# ------------------------------------------------ hold-off: set up first
# Frame sequence: entering the phase starts the clocks; the taker walks to
# a settle spot AT the ball; the hold counts from that arrival.
def holdoff_run(ball_xy, set_play, players, strike_now):
    ctx = make_ctx(ball_xy, set_play, now=0.0)
    store = SimpleNamespace(set_play_spot=None, set_play_taker=None,
                            prev_phase=None)
    _act_our_set_play(ctx, players, store)
    entry_calls = {p.id: list(p.calls) for p in players}
    entry_actions = {p.id: p.action for p in players}
    store.prev_phase = Phase.OUR_SET_PLAY
    ctx2 = make_ctx(ball_xy, set_play, now=strike_now)
    _act_our_set_play(ctx2, players, store)
    return entry_calls, entry_actions


def settle_of(ball_xy, set_play, players):
    """The taker's settle target, read from the entry frame's walk call."""
    ctx = make_ctx(ball_xy, set_play, now=0.0)
    store = SimpleNamespace(set_play_spot=None, set_play_taker=None,
                            prev_phase=None)
    _act_our_set_play(ctx, players, store)
    walks = [c[1] for c in players[0].calls if c[0] == "walk"]
    return walks[0] if walks else None


def in_field(stage):
    return stage is not None and (
        abs(stage[0]) <= ADULT_FIELD_DIMENSIONS.length / 2.0 - 0.3 + 1e-9
        and abs(stage[1]) <= ADULT_FIELD_DIMENSIONS.width / 2.0 - 0.3 + 1e-9
    )


# Goal kick, taker starting away from the ball: he walks to a settle spot
# roughly SET_PLAY_SETTLE_M behind the ball, and the strike waits for the
# entry fallback.
players = [make_player(1, (-5.0, 0.3)), make_player(2, (-2.0, 1.0)),
           make_player(3, (1.5, -1.5))]
stage = settle_of((-6.0, 0.0), SetPlay.GOAL_KICK, players)
check("goalkick hold-off: taker walks to a spot at the ball",
      stage is not None and 0.4 <= math.dist(stage, (-6.0, 0.0)) <= 1.2,
      f"stage={stage} dist={math.dist(stage, (-6.0, 0.0)) if stage else None:.2f}")
check("goalkick hold-off: staging spot inside the field", in_field(stage),
      f"stage={stage}")
entry_calls, entry_actions = holdoff_run(
    (-6.0, 0.0), SetPlay.GOAL_KICK, players, SET_PLAY_READY_MAX_S + 0.5)
check("goalkick hold-off: no strike while the hold runs",
      not any(c[0] == "kick" for c in entry_calls[players[0].id])
      and entry_actions[players[0].id] == "goalkick:hold",
      f"action={entry_actions[players[0].id]} "
      f"calls={[c[0] for c in entry_calls[players[0].id]]}")
check("goalkick hold-off: guard role still assigned",
      any(c[0] == "guard" for c in entry_calls[players[1].id]))
kick = [c for c in players[0].calls if c[0] == "kick"]
check("goalkick hold-off: fallback strike once the entry fallback expires",
      len(kick) == 1 and kick[0][2] == GOAL_KICK_POWER,
      f"calls={[c[0] for c in players[0].calls]}")

# The delay itself starts when the bot is standing at the ball: with him
# already settled, the strike comes one AT-BALL hold later, not one
# phase-entry fallback later.
players = [make_player(1, (-5.0, 0.3)), make_player(2, (-2.0, 1.0)),
           make_player(3, (1.5, -1.5))]
stage = settle_of((-6.0, 0.0), SetPlay.GOAL_KICK, players)
players = [make_player(1, stage), make_player(2, (-2.0, 1.0)),
           make_player(3, (1.5, -1.5))]
entry_calls, entry_actions = holdoff_run(
    (-6.0, 0.0), SetPlay.GOAL_KICK, players, SET_PLAY_HOLD_AT_BALL_S + 0.5)
check("goalkick hold-off: settled bot strikes one at-ball hold later",
      not any(c[0] == "kick" for c in entry_calls[players[0].id])
      and any(c[0] == "kick" for c in players[0].calls),
      f"entry={[c[0] for c in entry_calls[players[0].id]]} "
      f"all={[c[0] for c in players[0].calls]}")

# Defensive throw-in: same settle guarantees.
players = [make_player(1, (0.0, 4.0)), make_player(2, (-4.0, 0.0)),
           make_player(3, (2.0, -2.0))]
stage = settle_of((-2.5, 4.5), SetPlay.THROW_IN, players)
check("throwin hold-off: taker walks to a spot at the ball",
      stage is not None and 0.4 <= math.dist(stage, (-2.5, 4.5)) <= 1.2,
      f"stage={stage} dist={math.dist(stage, (-2.5, 4.5)) if stage else None:.2f}")
check("throwin hold-off: staging spot inside the field", in_field(stage),
      f"stage={stage}")

# Attacking throw-in: taker holds at the ball while receivers still walk
# to their spots near the goal.
players = [make_player(1, (2.0, 3.0)), make_player(2, (1.0, 0.0)),
           make_player(3, (3.0, -1.0))]
entry_calls, entry_actions = holdoff_run(
    (4.0, 4.5), SetPlay.THROW_IN, players, SET_PLAY_READY_MAX_S + 0.5)
check("attack throwin hold-off: taker holds at the ball, receivers still move",
      not any(c[0] == "kick" for c in entry_calls[players[0].id])
      and entry_actions[players[0].id] == "setplay:hold"
      and any(c[0] == "move" for c in entry_calls[players[1].id]
              + entry_calls[players[2].id]),
      f"actions={[entry_actions[p.id] for p in players]}")

# Corners and indirect free kicks are NOT delayed: they strike on frame one
# even at now=0.
players = [make_player(1, (5.0, 3.0)), make_player(2, (3.0, 0.0)),
           make_player(3, (4.0, -1.0))]
run_set_play((6.5, 4.4), SetPlay.CORNER_KICK, {}, players, now=0.0)
check("corner: not delayed, passes on frame one",
      any(c[0] == "kick" for c in players[0].calls),
      f"action={players[0].action}")


# ------------------------------------------------------- late-game urgency
def score_game(our, their, secs):
    return GameControlState(
        state=GameState.PLAYING,
        teams=(TeamState(team_number=1, score=our),
               TeamState(team_number=2, score=their)),
        secs_remaining=secs,
    )


def urgency_ctx(game, ball_xy=(0.0, 0.0)):
    return Context(
        now=0.0, dt=1.0 / 30.0, team_id=1, field=FIELD,
        game=game,
        ball=BallState(x=ball_xy[0], y=ball_xy[1]) if ball_xy else None,
        teammates={}, opponents={},
    )


check("urgency: missing broadcast -> normal",
      _match_urgency(urgency_ctx(None)) == URGENCY_NORMAL)
check("urgency: trailing late -> attack",
      _match_urgency(urgency_ctx(score_game(0, 1, 100))) == URGENCY_ATTACK)
check("urgency: trailing early -> normal",
      _match_urgency(urgency_ctx(score_game(0, 1, URGENT_ATTACK_SECS + 10)))
      == URGENCY_NORMAL)
check("urgency: leading late -> defend",
      _match_urgency(urgency_ctx(score_game(2, 0, 80))) == URGENCY_DEFEND)
check("urgency: leading early -> normal",
      _match_urgency(urgency_ctx(score_game(2, 0, URGENT_DEFEND_SECS + 10)))
      == URGENCY_NORMAL)
check("urgency: level -> normal",
      _match_urgency(urgency_ctx(score_game(1, 1, 50))) == URGENCY_NORMAL)
check("urgency: zeroed clock -> normal",
      _match_urgency(urgency_ctx(score_game(0, 1, 0))) == URGENCY_NORMAL)


def urgency_run(game):
    players = [make_player(1, (-2.0, 1.0)), make_player(2, (-5.0, 0.0)),
               make_player(3, (3.0, -1.0))]
    for p in players:
        p.support = lambda **kw: p.calls.append(("support", kw))
    ctx = urgency_ctx(game)
    _act_normal(ctx, players, SimpleNamespace())
    return players


players = urgency_run(score_game(0, 1, 60))
check("urgency play: trailing support stands ahead",
      ("support", {"ahead": True}) in players[2].calls
      and players[2].action == "support:ahead",
      f"calls={players[2].calls}")
check("urgency play: presser and guard unchanged",
      any(c[0] == "attack" for c in players[0].calls)
      and any(c[0] == "guard" for c in players[1].calls))

players = urgency_run(score_game(2, 0, 80))
check("urgency play: leading support parks deep",
      ("support", {"defend": True}) in players[2].calls
      and players[2].action == "support:defend",
      f"calls={players[2].calls}")

players = urgency_run(None)
check("urgency play: normal support otherwise",
      ("support", {}) in players[2].calls
      and players[2].action == "support",
      f"calls={players[2].calls}")

# Support geometry: ahead stands on the ball-to-their-goal line; defend
# parks at SUPPORT_DEFEND_DEPTH_M from our goal when the ball is far.
p2 = Player(2, SimpleNamespace(team_id=1), None)
moved_to = []
p2.move_to_position = lambda t: moved_to.append(t)
p2.context = urgency_ctx(score_game(0, 1, 60), ball_xy=(-6.0, 0.0))
p2.support(ahead=True)
t = moved_to[-1]
check("support ahead: on the ball-to-their-goal line at SUPPORT_DIST_M",
      abs(t[0] - (-2.7)) < 1e-6 and abs(t[1]) < 1e-6, f"target={t}")

p2.context = urgency_ctx(score_game(2, 0, 80), ball_xy=(5.0, 0.0))
p2.support(defend=True)
t = moved_to[-1]
check("support defend: parked SUPPORT_DEFEND_DEPTH_M from our goal",
      abs(t[0] - (-4.2)) < 1e-6 and abs(t[1]) < 1e-6, f"target={t}")


# ------------------------------------ unknown kicker reads as opponents'
game = GameControlState(state=GameState.PLAYING, set_play=SetPlay.THROW_IN)
check("phase: unknown kicker -> opponents' set play (never touch)",
      get_phase(urgency_ctx(game)) == Phase.OPP_SET_PLAY)
game = GameControlState(state=GameState.PLAYING, set_play=SetPlay.THROW_IN,
                        kicking_team=1)
check("phase: our kicker -> our set play",
      get_phase(urgency_ctx(game)) == Phase.OUR_SET_PLAY)
game = GameControlState(state=GameState.PLAYING, set_play=SetPlay.THROW_IN,
                        kicking_team=2)
check("phase: their kicker -> opponents' set play",
      get_phase(urgency_ctx(game)) == Phase.OPP_SET_PLAY)

# ----------------------------------------------- taker lock: no more fights
players = [make_player(1, (-5.0, 0.3)), make_player(2, (-2.0, 1.0)),
           make_player(3, (1.5, -1.5))]
ctx = make_ctx((-6.0, 0.0), SetPlay.GOAL_KICK, now=0.0)
store = SimpleNamespace(set_play_spot=None, set_play_taker=None, prev_phase=None)
_act_our_set_play(ctx, players, store)
store.prev_phase = Phase.OUR_SET_PLAY
players[1].pose = Pose2D(-5.9, 0.1, 0.0)
ctx2 = make_ctx((-6.0, 0.0), SetPlay.GOAL_KICK,
                now=SET_PLAY_READY_MAX_S + 0.5)
_act_our_set_play(ctx2, players, store)
check("taker lock: closer teammate does not steal the kick",
      not any(c[0] == "kick" for c in players[1].calls),
      f"p2 calls={[c[0] for c in players[1].calls]}")
check("taker lock: locked taker still takes the kick",
      any(c[0] == "kick" for c in players[0].calls),
      f"p1 calls={[c[0] for c in players[0].calls]}")

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
