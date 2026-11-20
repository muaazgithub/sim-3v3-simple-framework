"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- utils/: movement, geometry, and obstacle-avoidance helpers such as
  ``opponent_goal``, ``dist``, and ``angle_to``.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import angle_to, clamp, clamp_inside_field, dist, opponent_goal, own_goal


_log = logging.getLogger(__name__)


# ======================================================================
# Phase state machine: match phase classification
# ======================================================================


class Phase(Enum):
    """Top-level match phase controlling normal play, restarts, and stops."""
    NORMAL = "normal"              # Normal contest while PLAYING
    OUR_KICKOFF = "our_kickoff"    # Our kickoff during early SET/PLAYING
    OPP_KICKOFF = "opp_kickoff"    # Opponent kickoff; keep clear
    OUR_SET_PLAY = "our_set_play"  # Our free kick, corner, or goal kick
    OPP_SET_PLAY = "opp_set_play"  # Opponent set play; keep clear
    READY = "ready"                # Move into READY positions
    STOPPED = "stopped"            # Non-kickoff SET, INITIAL, FINISHED, or stopped


def get_phase(context: Context) -> Phase:
    """Determine the current phase from the GameController state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: move to ready positions.
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal play or an active kickoff/set play.
    if state == GameState.PLAYING and not g.stopped:
        # For a set play, kicking_team identifies the team taking it.
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # During the kickoff countdown, kicking_team identifies the taker.
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal play.
        return Phase.NORMAL

    # Remain stationary during SET, INITIAL, FINISHED, or a stoppage.
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """Return the active set-play type, or ``SetPlay.NONE`` when unavailable.

    This reads the GameController's ``set_play`` field without identifying the
    taking team. :func:`get_phase` distinguishes OUR_SET_PLAY from OPP_SET_PLAY;
    this function only identifies the type.

    The seven possible values from framework.types.SetPlay are:
    - ``NONE``: no set play, such as normal play or a kickoff
    - ``DIRECT_FREE_KICK``: a direct free kick that may score immediately
    - ``INDIRECT_FREE_KICK``: another player must touch the ball before a goal
    - ``PENALTY_KICK``: a penalty kick
    - ``THROW_IN``: a kick-in from touch
    - ``GOAL_KICK``: a goal kick
    - ``CORNER_KICK``: a corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.normal_attacker = None
        store.normal_defending = False  # Hysteresis for the defensive assignment
        store.set_play_spot = None    # Ball position where the set play started
        store.set_play_taker = None   # Passer of an indirect free kick
        store.prev_ball = None        # Previous known ball position (x, y)
        store.prev_ball_t = None      # Timestamp of the previous known ball spot
        store.ball_vel = None         # Smoothed ball velocity estimate (vx, vy)

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # Publish the shared ball velocity estimate to every player; the
        # keeper's shot prediction, the field interception, and the shot
        # gate all read it.
        ball_vel = getattr(store, "ball_vel", None)
        for p in players:
            p.ball_vel = ball_vel

        # Handle readiness and retain only players that can act this frame.
        # ensure_ready asynchronously gets up or switches to walk mode without
        # moving. Penalized players also prepare so they can rejoin immediately.
        # Penalized or unready players are excluded from role assignment to avoid
        # selecting an attacker who cannot move.
        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # May prepare, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Exclude players with unknown positions
                p.stop()
            else:
                active.append(p)

        # Dispatch the team once per phase; _act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw every teammate last, including penalized, unready, and stopped
        # players, so markers remain visible during states such as SET.
        for p in players:
            _draw_teammate_marker(p)

        # Estimate ball velocity from consecutive known positions, smoothed
        # to suppress tracking noise. A gap longer than BALL_VEL_MAX_DT_S
        # (missed detections) invalidates the estimate. Consumers: keeper
        # shot prediction, field interception, and the shot-stillness gate.
        # All store reads use getattr: a store persisted across a strategy
        # reload may lack newer fields, and an exception here makes the
        # framework stop every player for the tick — never crash the loop.
        ball = context.ball
        now = getattr(context, "now", None)
        prev = getattr(store, "prev_ball", None)
        prev_t = getattr(store, "prev_ball_t", None)
        vel = None
        if (
            ball is not None and prev is not None and prev_t is not None
            and now is not None
        ):
            dt = now - prev_t
            if 1e-3 < dt <= BALL_VEL_MAX_DT_S:
                raw = ((ball.x - prev[0]) / dt, (ball.y - prev[1]) / dt)
                old = getattr(store, "ball_vel", None)
                if old is None:
                    vel = raw
                else:
                    a = BALL_VEL_SMOOTHING
                    vel = (
                        a * raw[0] + (1.0 - a) * old[0],
                        a * raw[1] + (1.0 - a) * old[1],
                    )
        store.ball_vel = vel
        store.prev_ball = (ball.x, ball.y) if ball is not None else None
        store.prev_ball_t = now if ball is not None else None


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _player_dist_to_position(p: Player, target: tuple[float, float]) -> float:
    """Return infinity when the player's position is unknown."""
    pose = p.pose
    if pose is None:
        return math.inf
    return dist(pose.x, pose.y, *target)


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to the ball's current position."""
    ball = context.ball
    return (
        _player_dist_to_position(p, (ball.x, ball.y)) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _ball_stationary(context: Context, store) -> bool:
    """Return whether the ball has barely moved since the previous frame.

    Restart first touches re-issue ``kick()`` only while the ball is
    stationary: re-striking a ball that is already rolling piles strike
    distance onto strike distance and sends it out of play.
    """
    ball = context.ball
    prev = getattr(store, "prev_ball", None)
    if ball is None:
        return False
    if prev is None:
        return True                   # No history yet; allow the strike.
    return dist(ball.x, ball.y, *prev) < BALL_STILL_EPS_M


def _assign_support_roles(context: Context, taker: Player, players: list[Player]) -> None:
    """Give the players other than ``taker`` a guard and safe waiting spots.

    The player nearest our goal guards; the rest hold waiting slots outside
    the center circle and at least ``INACTIVITY_CLEAR_M`` from the ball, so
    standing there cannot draw the inactivity penalty.
    """
    rest = [p for p in players if p is not taker]
    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    r = context.field.circle_radius
    slot_x = -max(r + 0.5, INACTIVITY_CLEAR_M)
    slots = [(slot_x, 1.8), (slot_x - 0.9, -1.8)]
    for p, target in zip(rest, slots):
        p.action = "wait"
        p.move_to_position(target)


def _opponent_holds_ball(context: Context) -> bool:
    """Return whether an opponent is close enough to the ball to be holding it.

    A held ball cannot be won by circling; the attacker should tackle it.
    """
    ball = context.ball
    if ball is None:
        return False
    for opponent in context.opponents.values():
        pose = opponent.pose
        if pose is None:
            continue
        if dist(ball.x, ball.y, pose.x, pose.y) <= OPP_BALL_HOLD_DIST_M:
            return True
    return False


def _act_normal(context: Context, players: list[Player], store) -> None:
    """Assign roles for normal play, with a defensive keeper assignment.

    Once the ball threatens our goal (inside our half beyond the hysteresis
    lines, or rolling toward our goal), the player nearest our goal keeps
    guarding and is excluded from attacker selection — a defending keeper
    must never be pulled out of the goal by a chasing role. An opponent
    holding the ball is tackled instead of circled.
    """
    if not players:
        return

    ball = context.ball
    vel = getattr(store, "ball_vel", None)
    toward_goal = (
        ball is not None and vel is not None and vel[0] < -INTERCEPT_MIN_VX_MPS
    )

    defending = getattr(store, "normal_defending", False)
    if ball is not None:
        if not defending and (ball.x < DEFEND_ENTER_X_M or toward_goal):
            defending = True
        elif defending and ball.x > DEFEND_EXIT_X_M and not toward_goal:
            defending = False
    store.normal_defending = defending

    if defending:
        # Defensive shape: keeper guards, nearest of the rest attacks, the
        # remaining player supports (lane wall or interception).
        gx, gy = own_goal(context)
        keeper = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        keeper.action = "defend:keeper"
        keeper.guard()
        rest = [p for p in players if p is not keeper]

        attacker = _select_closest_attacker(
            context, rest, getattr(store, "normal_attacker", None),
        )
        store.normal_attacker = attacker.id
        if _opponent_holds_ball(context):
            attacker.action = "tackle"
            attacker.tackle()
        else:
            attacker.action = "attack"
            attacker.attack()
        for p in rest:
            if p is attacker:
                continue
            p.action = "support"
            p.support()
        return

    # Attacking phase: the baseline division of labour.
    attacker = _select_closest_attacker(
        context, players, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = attacker.id
    if _opponent_holds_ball(context):
        attacker.action = "tackle"
        attacker.tackle()
    else:
        attacker.action = "attack"
        attacker.attack()

    # The remaining player nearest our goal becomes the guard.
    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    # All other players support.
    for p in rest:
        p.action = "support"
        p.support()


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Lock the kickoff taker, pass on the first touch, and keep others clear.

    A goal straight from kickoff does not count, so while the ball is still
    inside ``KICKOFF_BALL_MOVED_M`` of the kickoff spot the taker passes to a
    forward point instead of shooting, and only chases and shoots through
    ``attack()`` after the ball has left that radius. Before the touch,
    teammates stay out of the center circle and at least ``INACTIVITY_CLEAR_M``
    from the ball so the inactivity rule cannot bite.
    """
    if not players:
        return

    active_ids = {p.id for p in players}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in active_ids:
        # Select a new taker when entering the kickoff phase.
        store.kickoff_taker = _select_closest_attacker(context, players).id

    attacker_id = store.kickoff_taker
    attacker = next((p for p in players if p.id == attacker_id), None)
    if attacker is None:
        return

    ball = context.ball
    moved = (
        ball is not None
        and dist(ball.x, ball.y, 0.0, 0.0) > KICKOFF_BALL_MOVED_M
    )
    if moved:
        # The ball is live: the nearest player takes it, whoever that is —
        # usually the taker chasing his own touch, sometimes the teammate it
        # rolled to.
        striker = _select_closest_attacker(context, players)
        striker.action = "kickoff:attack"
        striker.attack()
        _assign_support_roles(context, striker, players)
        return

    if attacker.is_kicking and not _ball_stationary(context, store):
        # A strike is already in flight; issuing kick() again would strike
        # the moving ball a second time and pile distance onto the touch.
        attacker.action = "kickoff:striking"
    else:
        # First touch: a soft pass toward a shallow forward point, never the
        # goal.
        attacker.action = "kickoff:pass"
        attacker.kick(
            angle_to(0.0, 0.0, KICKOFF_PASS_X_M, KICKOFF_PASS_Y_M),
            KICK_POWER_RESTART_PASS,
        )
    _assign_support_roles(context, attacker, players)


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """Guard with one player while the others wait clear of ball and circle.

    Waiting slots sit outside the center circle and at least
    ``INACTIVITY_CLEAR_M`` from the ball so standing there cannot draw the
    inactivity penalty.
    """
    if not players:
        return
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    guard.guard()

    rest = [p for p in players if p is not guard]
    r = context.field.circle_radius
    slot_x = -max(r + 0.5, INACTIVITY_CLEAR_M)
    slots = [(slot_x, 0.0), (slot_x - 0.9, 0.5)]
    for p, target in zip(rest, slots):
        p.action = "opp_kickoff:ready"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _push_clear_of_ball(
    context: Context, point: tuple[float, float], ball,
) -> tuple[float, float]:
    """Push a point away from the ball until it clears the inactivity radius.

    Clamping near a boundary can drag a stand-off point back next to the ball,
    where standing still draws the inactivity penalty. The point is pushed
    along its own ball-to-point direction on whichever side stays inside the
    field; the farther of the two results wins.
    """
    bx, by = ball.x, ball.y
    if dist(point[0], point[1], bx, by) >= INACTIVITY_CLEAR_M:
        return point
    ux, uy = point[0] - bx, point[1] - by
    norm = math.hypot(ux, uy)
    if norm >= 1e-6:
        ux, uy = ux / norm, uy / norm
    else:
        ux, uy = 1.0, 0.0
    half_l = context.field.length / 2.0 - 0.3
    half_w = context.field.width / 2.0 - 0.3
    options = []
    for sign in (1.0, -1.0):
        ox = clamp(bx + sign * ux * INACTIVITY_CLEAR_M, -half_l, half_l)
        oy = clamp(by + sign * uy * INACTIVITY_CLEAR_M, -half_w, half_w)
        options.append((dist(ox, oy, bx, by), ox, oy))
    # Both pushed options reach the inactivity radius unless clamping cut one
    # short. Round the capped distance so near-ties compare equal, then prefer
    # the upfield option on x — a one-ULP distance difference must not flip
    # the spot behind the ball.
    _, x, y = max(
        options,
        key=lambda o: (round(min(o[0], INACTIVITY_CLEAR_M), 6), o[1]),
    )
    return (x, y)


def _receiver_spots(context: Context, ball) -> list[tuple[float, float]]:
    """Deep and short receiving spots for an attacking restart.

    The deep spot waits at the penalty spot; the short spot lies along the
    ball-to-goal lane a little over 3 m from the ball. Both are clamped inside
    the field and pushed clear of the ball so waiting there cannot draw the
    inactivity penalty.
    """
    deep = clamp_inside_field(
        context, context.field.length / 2.0 - context.field.penalty_dist, 0.0,
        margin=0.3,
    )
    gx, gy = opponent_goal(context)
    dx, dy = gx - ball.x, gy - ball.y
    d = math.hypot(dx, dy)
    if d >= 1e-6:
        along = min(SET_PLAY_RECEIVE_DIST_M, max(0.5 * d, d - 1.2))
        short = (ball.x + dx / d * along, ball.y + dy / d * along)
    else:
        short = deep
    short = clamp_inside_field(context, *short, margin=0.3)
    return [
        _push_clear_of_ball(context, deep, ball),
        _push_clear_of_ball(context, short, ball),
    ]


def _pass_to_receivers(
    context: Context, taker: Player, players: list[Player], store,
    origin: tuple[float, float],
) -> None:
    """Pass an attacking restart to a receiver waiting near the goal.

    Two receivers walk to a deep spot at the penalty spot and a short spot on
    the ball-to-goal lane; the taker passes to whichever receiver is already
    closest to his spot, so the ball arrives as he gets there.
    """
    spots = _receiver_spots(context, context.ball)
    remaining = [p for p in players if p is not taker]
    assigned = []
    for spot in spots:
        if not remaining:
            break
        who = min(remaining, key=lambda p: _player_dist_to_position(p, spot))
        remaining.remove(who)
        assigned.append((who, spot))
    if assigned:
        _, pass_target = min(
            assigned, key=lambda item: _player_dist_to_position(item[0], item[1]),
        )
    else:
        pass_target = (KICKOFF_PASS_X_M, KICKOFF_PASS_Y_M)
    if taker.is_kicking and not _ball_stationary(context, store):
        # A strike is in flight; re-striking a rolling ball compounds the
        # touch and can send it out of play.
        taker.action = "setplay:striking"
    else:
        taker.action = "setplay:pass"
        taker.kick(
            angle_to(origin[0], origin[1], *pass_target),
            KICK_POWER_RESTART_PASS,
        )
    store.set_play_taker = taker.id
    for who, spot in assigned:
        who.action = "setplay:receive"
        who.move_to_position(spot)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Dispatch our set play by type.

    Direct restarts (direct free kick, penalty, goal kick, and throw-ins or
    corners deep in our own half) resume the normal division of labour with
    the taker attacking: ``attack()`` aims at the opponent's goal, so the
    first touch never goes toward our own goal. Attacking restarts —
    throw-ins and corners in the opponent half, plus every indirect free
    kick — pass instead of shooting: two receivers take up positions near the
    goal and, once the ball has left the spot, the nearest teammate who is
    not the passer takes the next touch while the passer drops into support.
    """
    if not players:
        return

    ball = context.ball
    if store.prev_phase != Phase.OUR_SET_PLAY:
        # Entering the phase: remember the restart spot so "ball moved" is
        # measured from where the referee placed the ball, and forget any
        # previous passer.
        store.set_play_spot = (ball.x, ball.y) if ball is not None else None
        store.set_play_taker = None
    elif getattr(store, "set_play_spot", None) is None and ball is not None:
        store.set_play_spot = (ball.x, ball.y)
    spot = getattr(store, "set_play_spot", None)
    if spot is not None:
        origin = spot
    elif ball is not None:
        origin = (ball.x, ball.y)
    else:
        origin = (0.0, 0.0)

    moved = (
        spot is not None and ball is not None
        and dist(ball.x, ball.y, *spot) > KICKOFF_BALL_MOVED_M
    )

    taker = _select_closest_attacker(context, players)
    set_play = get_set_play_type(context)

    attacking = ball is not None and ball.x > 0.0
    pass_first = set_play == SetPlay.INDIRECT_FREE_KICK or (
        set_play in (SetPlay.CORNER_KICK, SetPlay.THROW_IN) and attacking
    )
    if not pass_first:
        taker.action = "attack"
        taker.attack()
        _assign_support_roles(context, taker, players)
        return

    former_taker = getattr(store, "set_play_taker", None)
    if not moved:
        if attacking:
            _pass_to_receivers(context, taker, players, store, origin)
        else:
            # Indirect free kick deep in our half: a shallow forward pass,
            # with the rest of the team holding safe positions.
            if spot is not None and spot[0] >= KICKOFF_PASS_X_M:
                pass_target = (
                    min(
                        spot[0] + SET_PLAY_PASS_AHEAD_M,
                        context.field.length / 2.0 - SET_PLAY_PASS_EDGE_MARGIN_M,
                    ),
                    KICKOFF_PASS_Y_M if spot[1] >= 0.0 else -KICKOFF_PASS_Y_M,
                )
            else:
                pass_target = (KICKOFF_PASS_X_M, KICKOFF_PASS_Y_M)
            if taker.is_kicking and not _ball_stationary(context, store):
                # Let the in-flight strike finish before re-striking.
                taker.action = "setplay:striking"
            else:
                taker.action = "setplay:pass"
                taker.kick(
                    angle_to(origin[0], origin[1], *pass_target),
                    KICK_POWER_RESTART_PASS,
                )
            store.set_play_taker = taker.id
            _assign_support_roles(context, taker, players)
        return

    # Ball live after the pass: the passer must not re-touch before a
    # teammate does, so keep him out of attacker selection until the phase
    # ends. The nearest other player takes the next touch while everyone else
    # supports behind the ball.
    candidates = [p for p in players if p.id != former_taker] or players
    striker = _select_closest_attacker(context, candidates)
    striker.action = "attack"
    striker.attack()
    rest = [p for p in players if p is not striker]
    for p in rest:
        p.action = "support"
        p.support()


def _circle_points_in_field(
    context: Context, ball, radius: float,
) -> list[tuple[float, float]]:
    """In-field points exactly ``radius`` from the ball.

    Returns the circle's intersections with the in-field boundary rectangle;
    empty when the circle lies entirely outside, which cannot happen for the
    radii used here on this field.
    """
    half_l = context.field.length / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    candidates = []
    # The direct point on the ball-to-goal lane at this radius.
    gx, gy = own_goal(context)
    lx, ly = gx - ball.x, gy - ball.y
    line_len = math.hypot(lx, ly)
    if line_len >= 1e-6:
        candidates.append((
            ball.x + lx / line_len * radius,
            ball.y + ly / line_len * radius,
        ))
    for x in (-half_l, half_l):
        remaining = radius ** 2 - (x - ball.x) ** 2
        if remaining >= 0.0:
            offset = math.sqrt(remaining)
            candidates.extend((x, ball.y + s * offset) for s in (-1.0, 1.0))
    for y in (-half_w, half_w):
        remaining = radius ** 2 - (y - ball.y) ** 2
        if remaining >= 0.0:
            offset = math.sqrt(remaining)
            candidates.extend((ball.x + s * offset, y) for s in (-1.0, 1.0))
    return [
        t for t in candidates
        if t == clamp_inside_field(context, *t, margin=OPP_SET_PLAY_FIELD_MARGIN_M)
    ]


def _pick_wall_slot(
    context: Context, ball,
    radius: float,
    exclude: tuple[float, float] | None = None,
    min_separation: float = 0.0,
) -> tuple[float, float] | None:
    """Best blocking point on a circle around the ball for our goal's lane.

    Candidates lie ``radius`` from the ball, inside the field, on our goal's
    side of it. The point hugging the ball-to-goal line wins, so the wall
    blocks the direct shot or pass rather than open grass.
    """
    gx, gy = own_goal(context)
    lx, ly = gx - ball.x, gy - ball.y
    line_len = math.hypot(lx, ly)
    if line_len < 1e-6:
        return None
    best: tuple[float, float] | None = None
    best_key: tuple[float, float] | None = None
    for t in _circle_points_in_field(context, ball, radius):
        if exclude is not None and dist(t[0], t[1], *exclude) < min_separation:
            continue
        px, py = t[0] - ball.x, t[1] - ball.y
        along = (px * lx + py * ly) / line_len
        if along <= 0.0:
            continue                      # Behind the ball blocks nothing.
        perp = abs(px * ly - py * lx) / line_len
        key = (-perp, along)
        if best_key is None or key > best_key:
            best, best_key = t, key
    return best


def _wall_move(p: Player, context: Context, ball, target: tuple[float, float]) -> None:
    """Send one player to a wall point, retreating radially when too close.

    A player inside the wall circle exits to the nearest point on it, which
    only ever increases the distance to the ball, so the 1.45 m clearance is
    never crossed by cutting past the ball.
    """
    pose = p.pose
    if pose is None:
        p.stop()
        return
    if dist(pose.x, pose.y, ball.x, ball.y) < OPP_SET_WALL_NEAR_M:
        # Inside the wall circle: exit along the ray away from the ball,
        # which only ever increases the distance to the ball and never
        # crosses it. When the ray lands outside the field, fall back to the
        # nearest in-field point on the circle.
        outward = angle_to(ball.x, ball.y, pose.x, pose.y)
        radial = clamp_inside_field(
            context,
            ball.x + math.cos(outward) * OPP_SET_WALL_NEAR_M,
            ball.y + math.sin(outward) * OPP_SET_WALL_NEAR_M,
            margin=OPP_SET_PLAY_FIELD_MARGIN_M,
        )
        if dist(radial[0], radial[1], ball.x, ball.y) >= INACTIVITY_CLEAR_M:
            target = radial
        else:
            points = _circle_points_in_field(context, ball, OPP_SET_WALL_NEAR_M)
            if points:
                target = min(points, key=lambda t: dist(pose.x, pose.y, *t))
        p.action = "opp_set:retreat"
    else:
        p.action = "opp_set:wall"
    p.walk_to(
        target, face=angle_to(pose.x, pose.y, ball.x, ball.y),
        avoid_ball=True, avoid_robots=True,
    )


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Block the ball-to-goal lane while keeping every rule distance.

    The set-play ball is stationary until the opponent touches it, so instead
    of scattering, the team forms a wall on the lane toward our goal: one
    blocker at ``OPP_SET_WALL_NEAR_M``, one at ``OPP_SET_WALL_FAR_M``, and
    either a keeper or a laterally spread third blocker. Every standing point
    stays at least ``OPP_SET_WALL_NEAR_M`` from the ball, above both the
    1.45 m clearance line and the 3 m inactivity line.
    """
    ball = context.ball
    if ball is None:
        for p in players:
            p.action = "opp_set:wait"
            p.stop()
        return

    gx, gy = own_goal(context)
    near = _pick_wall_slot(context, ball, OPP_SET_WALL_NEAR_M)
    far = _pick_wall_slot(context, ball, OPP_SET_WALL_FAR_M)
    assignments: list[tuple[str, tuple[float, float] | None]] = []
    if near is not None:
        assignments.append(("wall", near))
    if far is not None:
        assignments.append(("wall", far))
    if dist(gx, gy, ball.x, ball.y) >= INACTIVITY_CLEAR_M + 0.2:
        # The keeper's post is clear of the ball, so someone can guard.
        assignments.append(("guard", None))
    else:
        # No legal keeper spot: spread a third blocker wide of the near one.
        spread = _pick_wall_slot(
            context, ball, OPP_SET_WALL_NEAR_M,
            exclude=near, min_separation=OPP_SET_WALL_SPREAD_M,
        )
        if spread is not None:
            assignments.append(("wall", spread))

    remaining = list(players)
    for kind, target in assignments:
        if not remaining:
            break
        if kind == "guard" or target is None:
            who = min(remaining, key=lambda p: _player_dist_to_position(p, (gx, gy)))
            remaining.remove(who)
            who.action = "opp_set:guard"
            who.guard()
        else:
            # Bind the narrowed target as a default argument: Pyright does
            # not carry loop-variable narrowing into lambda bodies.
            who = min(
                remaining,
                key=lambda p, t=target: _player_dist_to_position(p, t),
            )
            remaining.remove(who)
            _wall_move(who, context, ball, target)

    # Extra players, or no legal slots at all: hold spread-out points on the
    # near circle so nobody stands inside the rule distances.
    placed: list[tuple[float, float]] = [
        t for _, t in assignments if t is not None
    ]
    for p in remaining:
        pose = p.pose
        if pose is None:
            p.stop()
            continue
        points = [
            t for t in _circle_points_in_field(context, ball, OPP_SET_WALL_NEAR_M)
            if all(dist(t[0], t[1], q[0], q[1]) >= OPP_SET_WALL_SPREAD_M for q in placed)
        ] or _circle_points_in_field(context, ball, OPP_SET_WALL_NEAR_M)
        if not points:
            p.action = "opp_set:wait"
            p.stop()
            continue
        px, py = pose.x, pose.y
        target = min(points, key=lambda t: dist(px, py, *t))
        placed.append(target)
        _wall_move(p, context, ball, target)


def _act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to a ready position."""
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field = context.field
    # Our kickoff priority: center-circle edge, goal-area center, then left side.
    if our_kickoff:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-0.5, field.circle_radius + 2),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
    # Opponent kickoff priority: outside center circle, goal area, then penalty area.
    else:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius - 0.5, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-field.length / 2.0 + field.penalty_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates in red, using cubes while kicking and spheres otherwise.

    This runs for every player each frame regardless of phase, penalty, or
    readiness. The label contains the player ID and current high-level action
    (``p.action``), with ``[KICK]`` appended while kicking. Shape also indicates
    whether the player is in the kicking state.
    """
    from .framework import debugdraw

    pose = p.pose
    if pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(pose.x, pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(pose.x, pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(pose.x, pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Calculate player-to-ball distances and draw them each frame.

    This no longer depends on the analysis module; distances use the ball's
    current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Nothing can be drawn when the ball is not visible.
    if ball is None:
        return

    # 1. Draw the ball's current position as a green point.
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Draw distance labels: red for teammates and blue for opponents.
    for p in players:
        pose = p.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        pose = r.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )
