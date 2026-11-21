"""Player control handle for one robot, designed for direct user editing.

Platform primitives such as ``set_velocity``, ``kick``, ``release_kick``,
``request_mode``, and ``get_up`` delegate to the injected framework
``_backend``. State properties such as ``pose``, ``mode``, ``is_fallen``, and
``penalty`` read from ``self.context`` or the backend.

Movement behaviors such as ``walk_to``, ``face_to``, and ``ensure_ready`` are
also Player methods because they command this player and may need cross-frame
state for hysteresis or avoidance. Pure coordinate calculations such as
``dist``, ``angle_to``, and goal coordinates belong in utils/geom.

Instances live for the entire match, while the framework replaces
``self.context`` every frame. Add custom skills directly to this class.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Context, Penalty, Pose2D
from .framework import debugdraw

from .param import *
# The keeper-save constants are read defensively rather than relying on the
# star import: some container-based Pyright setups keep analyzing a stale
# param.py copy and flag freshly added star-imported names as undefined,
# which blocks the build. getattr falls back to the same defaults as
# param.py, while the real robot reads the tuned values at runtime.
from . import param as _param
KEEPER_LUNGE_TIME_S = getattr(_param, "KEEPER_LUNGE_TIME_S", 1.0)
KEEPER_JUMP_MODE = getattr(_param, "KEEPER_JUMP_MODE", None)
# Roll calibration for distance-based kick power (meters of roll per unit of
# kick power on this field's friction); defensive read like the constants
# above so a stale analyzer cache cannot flag it.
_ROLL_M_PER_POWER = getattr(_param, "KICK_ROLL_M_PER_POWER", 3.8)
from .utils.geom import (
    angle_to,
    clamp,
    clamp_inside_field,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
)
from .utils.obstacles import collect_obstacles
from .utils.path_planner import plan_global_path

if TYPE_CHECKING:
    from .framework.config import SoccerConfig
    from .framework.robot_backend import RobotBackend


__all__ = ["Player"]


_log = logging.getLogger(__name__)

def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list,
) -> float:
    """Return clearance along a lookahead ray for local obstacle avoidance.

    Only obstacles ahead of (px, py), with projection t > 0, block this heading.
    Ignoring obstacles behind or to the rear prevents a nearby rear obstacle
    from reducing clearance in every direction. Return infinity when clear;
    larger values mean more space and negative values indicate a collision.
    """
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue                      # Rear obstacles do not block this heading.
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """Return a position ``offset`` behind the ball relative to ``aim``.

    This is used as the chase target so the ball lies between the robot and
    ``aim``, naturally aligning the robot on arrival. If ``aim`` coincides with
    the ball and the direction is undefined, return the ball position.
    """
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d              # Unit vector from the ball to aim.
    return (ball_x - ux * offset, ball_y - uy * offset)   # Move away from aim.


def roll_power(distance: float, margin: float = 1.0) -> float:
    """Return the kick power that rolls ``distance * margin`` meters.

    Meters of roll per unit of power come from the calibration constant; the
    margin makes the ball arrive at the target with pace instead of dying
    short of it. Result is clamped to the firmware's power range.
    """
    roll = distance * margin
    return max(KICK_POWER_MIN, min(KICK_POWER_MAX, roll / _ROLL_M_PER_POWER))


class Player:
    """Handle for one player. Add new skills here.
    """

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: "RobotBackend | None",
    ) -> None:
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self._backend = _backend           # Robot control wrapper; rarely changed
        self.context: Context | None = None

        # Current high-level action for visualization and debugging. Strategy
        # dispatch updates it each frame; actions such as guard refine substates
        # internally. The visualization pass in main.py uses it as a label.
        self.action: str = "init"

        # SDK cache fields updated automatically by the framework backend.
        self._mode: str | None = None
        self._fall_down_state: str | None = None

        # Cross-frame detour-side memory; None means no active detour.
        self._avoid_side: float | None = None

        # Cross-frame kick hysteresis state.
        self._kicking: bool = False

        # Cross-frame block/guard hysteresis state.
        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        self._keeper_clearing: bool = False
        # Cross-frame state for driving the ball over the goal line.
        self._goal_line_push: bool = False
        # Cross-frame state for temporarily attacking when support gets stuck.
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

        # Cross-frame pass decision state for blocked shots.
        self._passing: bool = False
        self._pass_target_id: int | None = None
        # Cross-frame ball-possession hysteresis for the keeper.
        self._enemy_possession: bool = False
        # Latched aim seam so the shot direction stays stable across frames.
        self._aim_target: tuple[float, float] | None = None
        # Side memory for the dribble touch direction (1.0 / -1.0 / None).
        self._dribble_side: float | None = None
        # Ball velocity estimate for this frame, published by the strategy
        # in main.py from the shared ball position history.
        self.ball_vel: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def is_kicking(self) -> bool:
        """Return whether the player is currently kicking."""
        return self._kicking

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        if ctx is None:
            return None
        robot = ctx.teammates.get(self.id)
        return None if robot is None else robot.pose

    @property
    def mode(self) -> str | None:
        if self._backend is not None:
            return self._backend.mode
        return self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        if self._backend is not None:
            return getattr(self._backend, "fall_down_state", None)
        return self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        game = ctx.game if ctx is not None else None
        if game is None:
            return Penalty.NONE
        state = game.get_player_state(self.config.team_id, self.id)
        return Penalty.NONE if state is None else state.penalty

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    # ------------------------------------------------------------------
    # Chassis control
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                self.id, vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Kicking
    # ------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
    ) -> None:
        pose = self.pose
        if pose is None:
            _log.warning("player %d kick skipped: pose unknown", self.id)
            return

        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ball is None:
            _log.warning("player %d kick skipped: ball unknown", self.id)
            return

        if kick_direction is None:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                _log.warning("player %d kick skipped: kick plan unavailable", self.id)
                return
            kick_direction, power = kick_plan

        if self._backend is None:
            _log.debug(
                "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                self.id, ball.x, ball.y, kick_direction,
            )
            return
        # Transform field coordinates to body coordinates using the current pose.
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        ball_x_body = dx * cos_t + dy * sin_t
        ball_y_body = -dx * sin_t + dy * cos_t
        kick_direction = normalize_angle(kick_direction)
        direction_body = normalize_angle(kick_direction - pose.theta)
        power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
        self._kicking = True
        self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)

    def plan_kick(
        self, kick_target: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Calculate kick direction and power.

        Aim from the current ball position toward ``kick_target``, which
        defaults to the opponent's goal center. With the default target the
        aim point is scanned across the goal mouth for the widest open gap;
        when every lane is blocked, the kick becomes a pass to the best open
        teammate instead of a hopeful center shot. A ball deep in our end is
        cleared upfield instead of shot. Return
        ``(kick_direction, kick_power)``, or None when the ball or context is
        unavailable.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        if kick_target is None and self._ball_in_danger():
            return self._clear_plan(ball)

        if kick_target is None:
            kick_target, best_clear = self._best_shot_target(ball)
            # Hysteresis: once passing, stay in pass mode until the shot lane
            # reopens by PASS_EXIT_MARGIN_M, so the kick task does not flip
            # between shot and pass while an opponent hovers on the line.
            exit_margin = PASS_EXIT_MARGIN_M if self._passing else 0.0
            shot_open = best_clear >= SHOT_MIN_CLEARANCE_M + exit_margin
            # Shot gates: shooting range, angle cone, and a still-enough ball.
            # Long, acute, or moving-ball attempts are wild; carry or pass.
            if shot_open and not self._shot_allowed(ball):
                shot_open = False
            if not shot_open:
                if self._passing or best_clear < SHOT_MIN_CLEARANCE_M:
                    pass_plan = self._pass_plan(ball)
                    if pass_plan is not None:
                        return pass_plan
                self._passing = False
                dribble_plan = self._dribble_plan(ball)
                if dribble_plan is not None:
                    return dribble_plan
            else:
                self._passing = False
                self._pass_target_id = None

        kick_direction = angle_to(ball.x, ball.y, *kick_target)
        # The marker shows where the shot crosses the goal line, not the
        # caller's (possibly deeper) target point.
        self._draw_kick_target(self._goal_target_for_direction(kick_direction))
        # Power rolls the ball exactly to the aim point (plus pace margin):
        # on this low-friction field a fixed near-cap power sails out of play.
        kick_power = roll_power(
            dist(ball.x, ball.y, *kick_target), SHOT_ROLL_MARGIN,
        )

        return kick_direction, kick_power

    def _ball_in_danger(self) -> bool:
        """Return whether the ball sits deep enough in our end to force a clearance.

        Inside this zone the shot staging point would lie between the ball and
        our goal line, where any miscue risks our own goal.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        return ball.x < -ctx.field.length / 2.0 + DEFENSE_DANGER_DEPTH_M

    def _clear_plan(self, ball) -> tuple[float, float]:
        """Plan an upfield clearance biased toward the nearer open flank.

        The aim point always lies ahead of the ball, so the kick direction can
        never point back toward our own goal.
        """
        lateral = DEFENSE_CLEAR_LATERAL_M * (1.0 if ball.y >= 0.0 else -1.0)
        target = (ball.x + DEFENSE_CLEAR_RANGE_M, ball.y + lateral)
        self._draw_kick_target(target)
        return (
            angle_to(ball.x, ball.y, *target),
            roll_power(dist(ball.x, ball.y, *target), SHOT_ROLL_MARGIN),
        )

    def _best_shot_target(
        self, ball,
    ) -> tuple[tuple[float, float], float]:
        """Scan aim points across the goal mouth; return target and clearance.

        ``GOAL_AIM_SAMPLES`` points span the mouth between the post insets, so
        narrow seams between defenders are found, not just the center and the
        corners. Points the ball cannot reach on a straight line are skipped;
        among the rest the widest gap past any opponent wins when it clears
        ``SHOT_MIN_CLEARANCE_M``. With nothing open, the goal center is
        returned together with the best clearance found, so the caller can
        fall back to a pass or a hopeful center shot.
        """
        ctx = self.context
        if ctx is None:
            # Unreachable from plan_kick, which checks the context first.
            return (0.0, 0.0), -math.inf
        if ball is None:
            return opponent_goal(ctx), math.inf

        goal_x = ctx.field.length / 2.0
        half_goal = ctx.field.goal_width / 2.0
        inset = min(GOAL_CORNER_INSET_M, max(half_goal - 0.05, 0.0))
        lo, hi = -(half_goal - inset), (half_goal - inset)
        samples = max(GOAL_AIM_SAMPLES, 1)

        candidates: list[tuple[float, tuple[float, float]]] = []
        for k in range(samples):
            ratio = k / (samples - 1) if samples > 1 else 0.5
            y = lo + (hi - lo) * ratio
            direction = angle_to(ball.x, ball.y, goal_x, y)
            if not self.kick_can_score(direction):
                continue
            clear = self._shot_clearance(ball, (goal_x, y))
            candidates.append((clear, (goal_x, y)))
        if not candidates:
            self._aim_target = None
            return opponent_goal(ctx), 0.0
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_clear, best_target = candidates[0]

        # Hysteresis: keep the latched seam while it still qualifies and no
        # candidate is clearly wider, so the kick direction does not flicker
        # between aim points as opponents drift along the line.
        latched = self._aim_target
        if latched is not None:
            latch = next((c for c in candidates if c[1] == latched), None)
            if (
                latch is not None
                and latch[0] >= SHOT_MIN_CLEARANCE_M
                and latch[0] >= best_clear - AIM_SWITCH_MARGIN_M
            ):
                best_clear, best_target = latch

        if best_clear >= SHOT_MIN_CLEARANCE_M:
            self._aim_target = best_target
            return best_target, best_clear
        self._aim_target = None
        return opponent_goal(ctx), max(best_clear, 0.0)

    def _pass_plan(self, ball) -> tuple[float, float] | None:
        """Plan a pass to the best open teammate when the shot is blocked.

        Candidates are teammates with known poses at least
        ``PASS_BACKWARD_MARGIN_M`` upfield of the ball, never deep inside our
        defensive danger zone, with a pass lane that clears every opponent by
        ``PASS_MIN_LANE_CLEAR_M``. The current pass target is kept unless a
        clearly closer candidate appears, so the kick direction does not flap
        between teammates. Return ``(kick_direction, kick_power)`` or None
        when no teammate offers an open lane.
        """
        ctx = self.context
        if ctx is None:
            return None

        candidates: list[tuple[float, int, float, float]] = []
        for mate_id, mate in ctx.teammates.items():
            if mate_id == self.id:
                continue
            pose = mate.pose
            if pose is None:
                continue
            # Never pass deep into our defensive danger zone: a blocked shot
            # must not turn into a pass toward our own goal.
            if pose.x < -ctx.field.length / 2.0 + DEFENSE_DANGER_DEPTH_M:
                continue
            # The pass must go upfield of the ball, never backward.
            if pose.x < ball.x - PASS_BACKWARD_MARGIN_M:
                continue
            if self._shot_clearance(ball, (pose.x, pose.y)) < PASS_MIN_LANE_CLEAR_M:
                continue
            candidates.append((
                dist(ball.x, ball.y, pose.x, pose.y),
                mate_id, pose.x, pose.y,
            ))
        if not candidates:
            self._pass_target_id = None
            return None

        candidates.sort()
        chosen = candidates[0]
        if self._pass_target_id is not None:
            current = next(
                (c for c in candidates if c[1] == self._pass_target_id), None,
            )
            if (
                current is not None
                and current[0] <= candidates[0][0] + PASS_TARGET_SWITCH_M
            ):
                chosen = current
        self._pass_target_id = chosen[1]
        self._passing = True

        target = (chosen[2], chosen[3])
        power = roll_power(chosen[0], PASS_ROLL_MARGIN)
        self._draw_kick_target(target)
        debugdraw.line(
            [(ball.x, ball.y), target], rgb=(0.2, 0.8, 1.0), ns="pass_lane",
        )
        return angle_to(ball.x, ball.y, *target), power

    def _shot_allowed(self, ball) -> bool:
        """Return whether position and ball motion allow a shot attempt.

        The ball must be within ``SHOOT_MAX_RANGE_FRAC`` of the field length
        from the opponent goal center, the shot direction within
        ``SHOOT_MAX_ANGLE_RAD`` of the +x axis, and the ball slower than
        ``SHOT_MAX_BALL_SPEED_MPS``. Long or acute attempts and kicks on
        fast rolling balls spray the ball wildly; carry or pass instead.
        """
        ctx = self.context
        if ctx is None:
            return False
        gx, gy = opponent_goal(ctx)
        if dist(ball.x, ball.y, gx, gy) > SHOOT_MAX_RANGE_FRAC * ctx.field.length:
            return False
        if abs(angle_to(ball.x, ball.y, gx, gy)) > SHOOT_MAX_ANGLE_RAD:
            return False
        vel = self.ball_vel
        if vel is not None and math.hypot(vel[0], vel[1]) > SHOT_MAX_BALL_SPEED_MPS:
            return False
        return True

    def _dribble_plan(self, ball) -> tuple[float, float] | None:
        """Plan a soft dribble touch that advances around obstacles.

        Candidate directions fan out from the direct line to the opponent
        goal; the first candidate on each side whose probe stays clear of
        robots by ``DRIBBLE_CLEARANCE_M`` wins. The previous slant side is
        kept until it becomes blocked, so successive touches steer one way
        around a defender instead of jittering left and right. Returns
        ``(kick_direction, kick_power)``; even a boxed-in robot keeps the
        ball moving straight ahead, which beats a hopeful blocked shot.
        """
        ctx = self.context
        if ctx is None:
            return None

        goal_dir = angle_to(ball.x, ball.y, *opponent_goal(ctx))
        obstacles = collect_obstacles(
            ctx, self.id, ball=False, robots=True, goals=False,
        )

        plus_h: float | None = None
        minus_h: float | None = None
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offset = k * PLAN_STEP
            if plus_h is None and _heading_clearance(
                ball.x, ball.y, goal_dir + offset, obstacles,
            ) >= DRIBBLE_CLEARANCE_M:
                plus_h = goal_dir + offset
            if minus_h is None and _heading_clearance(
                ball.x, ball.y, goal_dir - offset, obstacles,
            ) >= DRIBBLE_CLEARANCE_M:
                minus_h = goal_dir - offset
            if plus_h is not None and minus_h is not None:
                break
            k += 1

        side = self._dribble_side
        if side == 1.0 and plus_h is not None:
            chosen, self._dribble_side = plus_h, 1.0
        elif side == -1.0 and minus_h is not None:
            chosen, self._dribble_side = minus_h, -1.0
        elif plus_h is not None:
            chosen, self._dribble_side = plus_h, 1.0
        elif minus_h is not None:
            chosen, self._dribble_side = minus_h, -1.0
        else:
            chosen, self._dribble_side = goal_dir, None

        self._draw_kick_target(self._goal_target_for_direction(chosen))
        return normalize_angle(chosen), KICK_POWER_DRIBBLE

    def tackle(self) -> None:
        """Go straight at an opponent-held ball and win it back.

        Normal attacking steers around robots, which never wins a ball an
        opponent is shielding; the tackle walks directly at the ball with
        avoidance off and, once in kick range, plays the regular
        ``plan_kick`` decision (clearance, pass, or dribble), so the poke is
        always safe for our own goal.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return

        d = dist(pose.x, pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                self.stop()
                return
            kick_direction, kick_power = kick_plan
            body_err = abs(normalize_angle(kick_direction - pose.theta))
            if body_err <= SHOT_MAX_BODY_ERR_RAD:
                self.kick(kick_direction, kick_power)
                return
            # Misaligned: get behind the ball facing the poke direction.
            self.release_kick()
            aim = (
                ball.x + math.cos(kick_direction),
                ball.y + math.sin(kick_direction),
            )
            chase = _behind_ball(ball.x, ball.y, aim, CHASE_BEHIND_M)
            self.walk_to(
                clamp_inside_field(ctx, *chase, margin=CHASE_FIELD_MARGIN_M),
                face=kick_direction,
            )
        else:
            self.release_kick()
            # Approach the ball directly: the opponent shielding it must be
            # met, not circled.
            self.walk_to(
                (ball.x, ball.y), avoid_ball=False, avoid_robots=False,
            )

    def _shot_clearance(
        self, ball, target: tuple[float, float],
    ) -> float:
        """Return the narrowest gap between the shot line and any opponent.

        The value is the smallest perpendicular distance from an opponent to
        the ball-to-target segment; opponents beyond the segment are ignored.
        Infinity means no opponent stands on the line.
        """
        ctx = self.context
        if ctx is None:
            return math.inf
        vx, vy = target[0] - ball.x, target[1] - ball.y
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return math.inf
        length = math.sqrt(length2)
        best = math.inf
        for opponent in ctx.opponents.values():
            pose = opponent.pose
            if pose is None:
                continue
            rx, ry = pose.x - ball.x, pose.y - ball.y
            t = (rx * vx + ry * vy) / length2
            if t < 0.0 or t > 1.0:
                continue
            best = min(best, abs(rx * vy - ry * vx) / length)
        return best

    def _goal_target_for_direction(
        self, kick_direction: float,
    ) -> tuple[float, float]:
        """Project the shot direction onto the opponent's goal line for display."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return (0.0, 0.0)

        dx = math.cos(kick_direction)
        if dx <= 1e-6:
            return opponent_goal(ctx)
        goal_x = ctx.field.length / 2.0
        t = max(0.0, (goal_x - ball.x) / dx)
        return (goal_x, ball.y + math.sin(kick_direction) * t)

    def _in_backfield(self) -> bool:
        """Return whether the ball is in our half, where kicks use more power."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        # own_penalty_edge_x = -ctx.field.length / 2.0 + ctx.field.penalty_area_length
        # return ball.x < own_penalty_edge_x
        return ball.x < 0

    def _draw_kick_target(self, target: tuple[float, float]) -> None:
        """Mark the kick target selected by plan_kick with an X."""
        from .framework import debugdraw

        x, y = target
        s = KICK_TARGET_MARK_SIZE_M
        debugdraw.line(
            [(x - s, y - s), (x + s, y + s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )
        debugdraw.line(
            [(x - s, y + s), (x + s, y - s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )

    def release_kick(self) -> None:
        self._kicking = False  # Clear kick hysteresis and cube visualization.
        self._passing = False  # Ending the kick task also ends the pass mode.
        self._pass_target_id = None
        self._aim_target = None  # The next kick picks its aim seam fresh.
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    def kick_can_score(self, kick_direction: float) -> bool:
        """Return whether a straight kick in ``kick_direction`` can score.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False

        goal_x = ctx.field.length / 2.0
        dx = math.cos(kick_direction)
        dy = math.sin(kick_direction)
        if dx <= 1e-6:
            return False

        half_goal = ctx.field.goal_width / 2.0
        if half_goal <= 0.0:
            return False
        if ball.x >= goal_x:
            y_at_goal = ball.y
        else:
            t = (goal_x - ball.x) / dx
            if t < 0.0:
                return False
            y_at_goal = ball.y + dy * t
        return -half_goal <= y_at_goal <= half_goal

    # ------------------------------------------------------------------
    # Slow operations, invoked asynchronously to avoid blocking
    # ------------------------------------------------------------------

    @property
    def is_getting_up(self) -> bool:
        """Return whether recovery is queued or still executing in the SDK."""
        return self._backend is not None and self._backend.is_getting_up

    def request_mode(self, mode: str) -> None:
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", self.id, mode)
            return
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", self.id)
            return
        self._backend.get_up()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def ensure_ready(self) -> bool:
        """Recover from falls and switch to walk mode asynchronously.

        Return whether the player is ready to act during this frame. Mode
        changes must wait for the get-up task, not just a normal fall snapshot.
        """
        if self.is_getting_up:
            return False
        if self.is_fallen:
            self.get_up()
            return False
        if self.mode != "walk":
            self.request_mode("walk")
            return False
        return True

    def face_to(self, target_theta: float) -> None:
        """Turn in place to the target heading."""
        pose = self.pose
        if pose is None:
            self.stop()
            return
        err = normalize_angle(target_theta - pose.theta)
        self.set_velocity(0.0, 0.0, self._angular(err))

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
    ) -> bool:
        """Walk toward a target point and return whether it has been reached.

        With ``avoid_ball`` or ``avoid_robots`` enabled, the simplified VFH
        local planner models the ball, robots, and goals as circular obstacles.
        It scans candidate headings and selects the one closest to the target
        that remains collision-free over ``PLAN_LOOKAHEAD``. This handles
        multiple obstacles, concave structures, and symmetric conflicts more
        reliably than single-obstacle detours or potential-field repulsion.

        Nearby targets use omnidirectional walking; distant targets use a
        turn-walk-turn sequence. ``face`` sets the heading near or at the target.
        """
        self.release_kick() # Walking is overridden while kicking, so release first.
        pose = self.pose
        if pose is None:
            self.stop()
            return False

        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y
        distance = math.hypot(dx, dy)

        if distance < arrive_dist:
            # At the target, turn to the requested heading if needed.
            if face is not None:
                err = normalize_angle(face - pose.theta)
                if abs(err) > 0.1:
                    self.set_velocity(0.0, 0.0, self._angular(err))
                else:
                    self.stop()
            else:
                self.stop()
            return True

        # Prefer global A* planning and fall back to the legacy local planner.
        goal_dir = math.atan2(dy, dx)
        planned_path: list[tuple[float, float]] | None = None
        waypoint: tuple[float, float] | None = None
        ctx = self.context
        if (avoid_ball or avoid_robots) and ctx is not None:
            obstacles = collect_obstacles(
                ctx, self.id,
                ball=avoid_ball, robots=avoid_robots,
                goals=(avoid_ball or avoid_robots),
            )
            if USE_GLOBAL_PATH_PLANNER:
                planned_path = plan_global_path(
                    ctx,
                    (pose.x, pose.y),
                    (tx, ty),
                    obstacles,
                )
                if planned_path is not None:
                    waypoint = self._path_waypoint(pose, planned_path)
                    heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                else:
                    heading = self._plan_heading(pose, goal_dir, obstacles)
            else:
                heading = self._plan_heading(pose, goal_dir, obstacles)
        else:
            heading = goal_dir

        # Visualize the target (green), direct line (gray), planned heading
        # (yellow), and forward lookahead probe (cyan).
        from .framework import debugdraw
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

        if distance <= OMNI_DIST:
            # Nearby: translate along heading while turning toward face.
            wdx, wdy = math.cos(heading) * distance, math.sin(heading) * distance
            cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
            vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
            vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
            speed = math.hypot(vx, vy)
            if speed > MAX_LINEAR:
                vx *= MAX_LINEAR / speed
                vy *= MAX_LINEAR / speed
            vyaw = (
                self._angular(normalize_angle(face - pose.theta))
                if face is not None else 0.0
            )
            self.set_velocity(vx, vy, vyaw)
        else:
            # Distant: turn, walk, and turn along heading.
            angle_err = normalize_angle(heading - pose.theta)
            if abs(angle_err) > TURN_THRESHOLD:
                self.set_velocity(0.0, 0.0, self._angular(angle_err))
            else:
                vx = clamp(
                    LINEAR_GAIN * distance * math.cos(angle_err), 0.0, MAX_LINEAR,
                )
                self.set_velocity(vx, 0.0, self._angular(angle_err))
        return False

    def _path_waypoint(
        self, pose: Pose2D, path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Pick a short lookahead waypoint from a planned global path."""
        if not path:
            return (pose.x, pose.y)
        prev = (pose.x, pose.y)
        points = path[1:] if len(path) > 1 else path
        for point in points:
            seg_len = dist(prev[0], prev[1], point[0], point[1])
            if seg_len >= GLOBAL_PATH_LOOKAHEAD_M:
                ratio = GLOBAL_PATH_LOOKAHEAD_M / max(seg_len, 1e-6)
                return (
                    prev[0] + (point[0] - prev[0]) * ratio,
                    prev[1] + (point[1] - prev[1]) * ratio,
                )
            prev = point
        return path[-1]

    def _plan_heading(
        self, pose: Pose2D, goal_dir: float, obstacles: list,
    ) -> float:
        """Choose the clearest candidate heading nearest the target direction.

        Candidates are tested by increasing absolute offset from the target.
        Player ID parity chooses which side is tried first, breaking symmetry
        when two players avoid each other.
        """
        sign_first = 1.0 if self.id % 2 == 0 else -1.0
        best_h = goal_dir
        best_clear = -math.inf

        offsets = [0.0]
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offsets.append(sign_first * k * PLAN_STEP)
            offsets.append(-sign_first * k * PLAN_STEP)
            k += 1

        for off in offsets:
            h = goal_dir + off
            clear = _heading_clearance(pose.x, pose.y, h, obstacles)
            if clear >= PLAN_CLEARANCE:
                return h
            if clear > best_clear:
                best_clear, best_h = clear, h
        return best_h

    # ------------------------------------------------------------------
    # High-level actions called directly by the strategy in main.py
    # ------------------------------------------------------------------


    def block_path_projection(
        self, opponent_id: int,
    ) -> tuple[float, float, float, float] | None:
        """Project this player onto an opponent-to-ball segment.

        Return ``(x, y, perpendicular_distance, segment_parameter_t)``.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        opponent = ctx.opponents.get(opponent_id) if ctx is not None else None
        if ctx is None or pose is None or ball is None or opponent is None:
            return None
        opponent_pose = opponent.pose
        if opponent_pose is None:
            return None

        ax, ay = opponent_pose.x, opponent_pose.y
        bx, by = ball.x, ball.y
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return None

        raw_t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / length2
        t = clamp(raw_t, 0.0, 1.0)
        tx = ax + vx * t
        ty = ay + vy * t
        return tx, ty, dist(pose.x, pose.y, tx, ty), raw_t

    def _chase_align_target(
        self, ctx: Context, kick_target: tuple[float, float],
    ) -> tuple[float, float]:
        """Return the point the chase should align behind the ball toward.

        While a pass is in progress the chase aligns toward the pass target
        instead of the goal, so the approach geometry matches the planned
        kick direction and the alignment gate can open sooner.
        """
        if self._passing and self._pass_target_id is not None:
            mate = ctx.teammates.get(self._pass_target_id)
            pose = mate.pose if mate is not None else None
            if pose is not None:
                return (pose.x, pose.y)
        return kick_target

    def attack(self, kick_target: tuple[float, float] | None = None) -> None:
        """Chase the ball and shoot toward ``kick_target`` or the opponent goal.

        Kick hysteresis enters below ENTER and exits above EXIT, where EXIT is
        greater than ENTER, to prevent threshold oscillation. The chase avoids
        robots but not the ball. The chase target lies ``CHASE_BEHIND_M`` behind
        the ball on the ball-to-target line, naturally aligning the player for a
        shot on arrival, and the same target also sets the shot direction.

        While the planned shot direction still points farther off the robot's
        heading than ``SHOT_MAX_BODY_ERR_RAD``, the kick is withheld and the
        robot repositions behind the ball facing the shot instead: firing
        sideways or backwards wastes the shot and can turn it into an own
        goal. With no open shot (``kick_target=None``), ``plan_kick`` returns
        a pass to the best open teammate instead of a blocked center shot.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(ctx)

        d = dist(pose.x, pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            kick_plan = self.plan_kick(kick_target)
            if kick_plan is None:
                self.stop()
                return
            kick_direction, kick_power = kick_plan
            body_err = abs(normalize_angle(kick_direction - pose.theta))
            if body_err <= SHOT_MAX_BODY_ERR_RAD:
                self.kick(kick_direction, kick_power)
                return
            # Misaligned: reposition behind the ball on the shot line and
            # face the shot direction, so the next frames can fire cleanly.
            self.release_kick()
            aim = (
                ball.x + math.cos(kick_direction),
                ball.y + math.sin(kick_direction),
            )
            chase = _behind_ball(ball.x, ball.y, aim, CHASE_BEHIND_M)
            self.walk_to(
                clamp_inside_field(ctx, *chase, margin=CHASE_FIELD_MARGIN_M),
                face=kick_direction, avoid_robots=True,
            )
        else:
            self.release_kick()
            if self._ball_in_danger():
                # Stage upfield of the ball instead of between the ball and
                # our goal line: any body contact then pushes the ball away
                # from our goal, never into it.
                chase = (ball.x + CHASE_BEHIND_M, ball.y)
            else:
                align = self._chase_align_target(ctx, kick_target)
                chase = _behind_ball(ball.x, ball.y, align, CHASE_BEHIND_M)
            self.walk_to(
                clamp_inside_field(ctx, *chase, margin=CHASE_FIELD_MARGIN_M),
                avoid_robots=True,
            )

    def _enemy_has_ball(self) -> bool:
        """Update and return whether the nearest robot to the ball is an opponent.

        Possession is a nearest-opponent-versus-nearest-teammate race; the
        ``GUARD_POSSESSION_MARGIN_M`` hysteresis keeps the verdict from
        flipping every frame on near ties. Robots with unknown poses are
        ignored, and with no ball the previous verdict stands.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return self._enemy_possession
        opp_d = min(
            (
                dist(ball.x, ball.y, o.pose.x, o.pose.y)
                for o in ctx.opponents.values() if o.pose is not None
            ),
            default=math.inf,
        )
        team_d = min(
            (
                dist(ball.x, ball.y, pose.x, pose.y)
                for pose in (
                    t.pose for t in ctx.teammates.values() if t.pose is not None
                )
            ),
            default=math.inf,
        )
        if opp_d + GUARD_POSSESSION_MARGIN_M < team_d:
            self._enemy_possession = True
        elif team_d + GUARD_POSSESSION_MARGIN_M < opp_d:
            self._enemy_possession = False
        return self._enemy_possession

    def guard(self, allow_clear: bool = True) -> None:
        """Guard the goal, sprinting back when the opponents have the ball.

        When the ball comes within ``KEEPER_CLEAR_ENTER_M`` the keeper shoots
        it upfield at once (``allow_clear=False`` disables this during the
        opponents' restarts, where touching the ball early is a foul).
        With the ball in enemy possession the keeper runs straight to the
        goal mouth and tracks the ball's y across it, so a shot or pass
        toward our goal meets a blocker as early as possible. Facing is only
        imposed near the blocking spot, so the long run goes straight at
        maximum walk speed. Away from danger the keeper waits at the
        goal-area center, and once the ball crosses into our half
        (hysteresis on ``GUARD_THREAT_ENTER_X`` and ``GUARD_THREAT_EXIT_X``)
        the keeper tracks the ball across the mouth, which improves shot
        blocking and keeps the keeper moving instead of standing inside the
        3 m inactivity radius.
        """
        ctx = self.context
        pose = self.pose
        if ctx is None or pose is None:
            self.action = "guard:stop"
            self.stop()
            return
        home = own_goal_area_center(ctx)

        ball = ctx.ball

        # Ball inside the clear radius: shoot it upfield immediately — the
        # kick task walks the last stretch itself, so this fires as soon as
        # the ball is near. Disabled during the opponents' restarts, where
        # touching the ball early is a foul.
        if allow_clear and ball is not None:
            d_ball = dist(pose.x, pose.y, ball.x, ball.y)
            if d_ball <= (
                KEEPER_CLEAR_EXIT_M if self._keeper_clearing
                else KEEPER_CLEAR_ENTER_M
            ):
                self._keeper_clearing = True
                self.action = "guard:clear"
                kick_direction, kick_power = self._clear_plan(ball)
                self.kick(kick_direction, kick_power)
                return
        self._keeper_clearing = False

        face_ball = (
            angle_to(pose.x, pose.y, ball.x, ball.y)
            if GUARD_FACE_BALL and ball is not None else 0.0
        )

        # Predictive intercept: once the ball travels toward our goal fast
        # enough, sprint to where its path crosses the goal line instead of
        # chasing its current side of the mouth. Blocking means getting the
        # body onto the shot line, so the ball is not treated as an obstacle.
        vel = self.ball_vel
        if (
            ball is not None and vel is not None
            and vel[0] < -INTERCEPT_MIN_VX_MPS
            and math.hypot(vel[0], vel[1]) >= KEEPER_INTERCEPT_MIN_SPEED_MPS
        ):
            vx, vy = vel
            goal_line_x = -ctx.field.length / 2.0
            t_cross = (goal_line_x - ball.x) / vx      # vx < 0, so t > 0
            y_cross = ball.y + vy * t_cross
            mouth = ctx.field.goal_width / 2.0 + GUARD_TRACK_MARGIN_M
            target = clamp_inside_field(
                ctx, home[0], clamp(y_cross, -mouth, mouth), margin=0.3,
            )
            self.action = "guard:intercept"
            if t_cross <= KEEPER_LUNGE_TIME_S:
                # Imminent shot. If the robot firmware provides a real
                # jump/dive save mode, use it; note the mode must return to
                # walk by itself (like get-up), since ensure_ready requests
                # walk again on any other mode.
                if KEEPER_JUMP_MODE:
                    self.request_mode(KEEPER_JUMP_MODE)
                    return
                # Otherwise lunge: a direct maximum-speed omnidirectional
                # burst at the crossing point — no turn phases, no
                # avoidance. Throwing the body onto the line matters more
                # than posture, and the framework recovers a fall for free.
                dx_l, dy_l = target[0] - pose.x, target[1] - pose.y
                d_l = math.hypot(dx_l, dy_l)
                if d_l >= 1e-6:
                    ux_l, uy_l = dx_l / d_l, dy_l / d_l
                    cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
                    self.set_velocity(
                        MAX_LINEAR * (ux_l * cos_t + uy_l * sin_t),
                        MAX_LINEAR * (-ux_l * sin_t + uy_l * cos_t),
                        0.0,
                    )
                    debugdraw.line(
                        [(pose.x, pose.y), (target[0], target[1])],
                        rgb=(1.0, 0.4, 0.0), ns="keeper_lunge",
                    )
                    return
            face = (
                face_ball if dist(pose.x, pose.y, *target) <= OMNI_DIST
                else None
            )
            avoid_ball = KEEPER_BLOCK_AVOID_BALL
        elif ball is not None and self._enemy_has_ball():
            mouth = ctx.field.goal_width / 2.0 + GUARD_TRACK_MARGIN_M
            target = clamp_inside_field(
                ctx, home[0], clamp(ball.y, -mouth, mouth), margin=0.3,
            )
            self.action = "guard:block"
            face = (
                face_ball if dist(pose.x, pose.y, *target) <= OMNI_DIST
                else None
            )
            avoid_ball = KEEPER_BLOCK_AVOID_BALL
        else:
            if ball is not None:
                if ball.x < GUARD_THREAT_ENTER_X:
                    self._guard_threatened = True
                elif ball.x > GUARD_THREAT_EXIT_X:
                    self._guard_threatened = False
            if self._guard_threatened and ball is not None:
                mouth = ctx.field.goal_width / 2.0 + GUARD_TRACK_MARGIN_M
                target = clamp_inside_field(
                    ctx, home[0], clamp(ball.y, -mouth, mouth), margin=0.3,
                )
                self.action = "guard:track"
            else:
                target = home
                self.action = "guard:home"
            face = face_ball
            avoid_ball = True

        # Face the ball for faster reactions, or the opponent's goal if unseen.
        debugdraw.point(
            target[0], target[1], rgb=(0.0, 0.6, 1.0), scale=0.2, ns="guard_home",
        )
        self.walk_to(target, face=face, avoid_ball=avoid_ball, avoid_robots=True)

    def support(self) -> None:
        """Support the ball-to-own-goal line; block the lane when defending.

        With the ball in our half the support becomes a second blocker on the
        ball-to-goal lane at ``DEFEND_WALL_DIST_M``, denying the direct shot.
        When a ball is rolling toward our goal fast enough, the support meets
        it on its path ``INTERCEPT_LEAD_M`` ahead instead of waiting on the
        lane. With the ball in the opponent half it holds the classic support
        spot at ``SUPPORT_DIST_M`` from the ball.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            self.stop()
            return

        gx, gy = own_goal(ctx)
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d

        vel = self.ball_vel
        if (
            ball.x < 0.0 and vel is not None
            and vel[0] < -INTERCEPT_MIN_VX_MPS
            and math.hypot(vel[0], vel[1]) >= KEEPER_INTERCEPT_MIN_SPEED_MPS
        ):
            # Rolling toward our goal: meet the ball on its predicted path
            # ahead of it, where a body in the way stops the attack early.
            speed = math.hypot(vel[0], vel[1])
            tx = bx + vel[0] / speed * INTERCEPT_LEAD_M
            ty = by + vel[1] / speed * INTERCEPT_LEAD_M
            self.move_to_position(clamp_inside_field(ctx, tx, ty, margin=0.3))
            return

        if ball.x < 0.0:
            # Defending: hold the ball-to-goal lane as a second blocker.
            along = min(DEFEND_WALL_DIST_M, max(d - 0.8, 0.0))
        else:
            along = min(SUPPORT_DIST_M, d)
        tx = bx + ux * along
        ty = by + uy * along

        # Stay 0.3 m inside our goal line and clamp laterally inside the field.
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0 - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)

        # Clamping near our goal line can pull the point back next to the ball
        # (goal kicks, pinned balls). Standing there draws the inactivity
        # penalty, so push the point away from the ball on whichever side
        # stays inside the field.
        if dist(tx, ty, bx, by) < INACTIVITY_CLEAR_M:
            ux_t, uy_t = tx - bx, ty - by
            norm = math.hypot(ux_t, uy_t)
            ux_t, uy_t = (
                (ux_t / norm, uy_t / norm) if norm >= 1e-6 else (-1.0, 0.0)
            )
            options = []
            for sign in (1.0, -1.0):
                ox = clamp(
                    bx + sign * ux_t * INACTIVITY_CLEAR_M,
                    -half_l + 0.3, half_l,
                )
                oy = clamp(
                    by + sign * uy_t * INACTIVITY_CLEAR_M, -half_w, half_w,
                )
                options.append((dist(ox, oy, bx, by), ox, oy))
            _, tx, ty = max(options)
        self.move_to_position((tx, ty))

    def take_kickoff(self, kick_target: tuple[float, float] | None = None) -> None:
        """Stage behind the ball for our restart, then approach and kick."""
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(ctx)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        rel_x, rel_y = pose.x - ball.x, pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k          # Negative is our side.
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.attack(kick_target)

    def move_to_position(self, target: tuple[float, float] | None) -> None:
        """Move to a support/defense position while facing and avoiding the ball."""
        if target is None:
            self.stop()
            return
        face = None
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ball is not None and pose is not None:
            face = angle_to(pose.x, pose.y, ball.x, ball.y)
        self.release_kick()
        self.walk_to(target, face=face, avoid_ball=True, avoid_robots=True)

    @staticmethod
    def _angular(err: float) -> float:
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)
