"""Central configuration point for strategy tuning.

This module contains tunable strategy, movement, kicking, and obstacle
avoidance parameters. Existing values were moved here from their original
modules; make future parameter adjustments here first.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power to this range. Unsupported values may cause a fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Kick power during normal play. Values near the 10.0 cap make robots fall in
# simulation; if players start dropping after their own kicks, lower these.
KICK_POWER_DEFAULT = 8.0
KICK_POWER_BACKFIELD = 9.0

# Soft first touch for restarts (kickoff, indirect free kick, attacking
# throw-ins and corners): a full-power pass from the spot sails out of play.
KICK_POWER_RESTART_PASS = 2.0


# ======================================================================
# Player movement control
# ======================================================================

ARRIVE_DIST = 0.15             # Target arrival threshold (m)
OMNI_DIST = 1.0                # Use omnidirectional control below this distance
TURN_THRESHOLD = 0.5           # Turn in place above this heading error (rad)
MAX_LINEAR = 2.0               # Maximum commanded forward speed (m/s)
MAX_ANGULAR = 2.0              # Maximum commanded angular speed (rad/s)
LINEAR_GAIN = 1.5              # Translation proportional gain
ANGULAR_GAIN = 2.0             # Rotation proportional gain


# ======================================================================
# Player kicking and shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kicking state below this ball distance
KICK_EXIT_M = 2.5              # Leave kicking state above this ball distance
CHASE_BEHIND_M = 0.35          # Distance behind the ball while chasing
CHASE_FIELD_MARGIN_M = 0.3     # Chase targets are clamped this far inside the field

# Danger zone: a ball this deep in our end must be cleared upfield, never
# staged for a shot, so no approach can push it into our own goal.
DEFENSE_DANGER_DEPTH_M = 3.5
DEFENSE_CLEAR_RANGE_M = 6.0    # Clearance aims this far upfield of the ball
DEFENSE_CLEAR_LATERAL_M = 2.0  # Clearance lateral bias toward the nearer flank


# ======================================================================
# Passing game
# ======================================================================

# Ball roll calibration: rough meters of roll per unit of kick power on this
# field's friction (derived from the kickoff pass sailing out of play at
# power 2.0). Every distance-based kick derives its power from this constant.
KICK_ROLL_M_PER_POWER = 3.8
# A pass rolls this much farther than the distance to its receiver, so it
# arrives with pace instead of dying short of the receiver.
PASS_ROLL_MARGIN = 1.1
# The best shot lane narrower than this (m) counts as blocked: the carrier
# looks for an open teammate instead of shooting into the blocker.
SHOT_MIN_CLEARANCE_M = 0.75
# Once passing, the shot lane must reopen by this extra margin (m) before
# shooting resumes, so the kick does not flip between shot and pass while an
# opponent hovers on the line.
PASS_EXIT_MARGIN_M = 0.25
# A pass lane must clear every opponent by this much (m) to be thrown.
PASS_MIN_LANE_CLEAR_M = 0.6
# The receiver must sit at least this far upfield of the ball (m): a blocked
# shot never turns into a pass backward or into our own danger zone.
PASS_BACKWARD_MARGIN_M = 0.5
# The current pass target is kept unless a candidate appears this much (m)
# closer, so the kick direction does not flap between teammates.
PASS_TARGET_SWITCH_M = 0.8


# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7
# When threatened, the keeper tracks the ball's y across the goal mouth plus
# this margin instead of sitting at the goal-area center.
GUARD_TRACK_MARGIN_M = 0.5
# A ball this close to the keeper or the goal-area center is swept: the keeper
# goes for it and clears it upfield instead of watching it roll in.
GUARD_SWEEP_M = 1.5
# Sweeps only apply to balls slower than this (m/s); a fast shot is met on its
# crossing line instead, since running into it can deflect the ball in.
GUARD_SWEEP_MAX_MPS = 2.0
# Sweeper-keeper (reduced): slow balls anywhere inside our goal area — this
# margin beyond its box — are the keeper's to clear, not just balls within
# GUARD_SWEEP_M of the keeper or the goal-area center. Fast balls are still
# only met on their crossing line.
GUARD_SWEEP_AREA_MARGIN_M = 0.3
# Shot prediction: ignore balls slower than this toward our goal, and look at
# most this far ahead (s) when computing the crossing point.
GUARD_PREDICT_MIN_VX_MPS = 0.3
GUARD_PREDICT_HORIZON_S = 1.2
# Frame-to-frame estimates faster than this come from the ball being placed at
# a restart, not from a real shot; treat them as standing still.
GUARD_VELOCITY_CAP_MPS = 15.0

# Stand-off distance from the ball. Waiting here means standing still, so keep
# above the 3 m inactivity line with margin (see INACTIVITY_CLEAR_M).
SUPPORT_DIST_M = 3.3

# With the ball in the opponent half, the support takes an advanced receiving
# spot this far from the ball on the ball-to-goal lane instead of standing
# behind the play, giving the carrier a forward pass option.
ATTACK_RECEIVE_DIST_M = 3.4
# Lateral offset of the receiving spot (m) toward the ball's side of the
# field, so the receiver never stands in our own shot lane.
ATTACK_RECEIVE_OFFSET_M = 1.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# A goal straight from kickoff does not count: the taker may only shoot once
# the ball has left this radius around the kickoff spot (referee line is
# 0.5 m; keep margin). Until then the first touch passes instead of shooting.
KICKOFF_BALL_MOVED_M = 0.6
KICKOFF_PASS_X_M = 2.5             # Shallow restart pass target (field x, m)
# Shallow lateral offset: even a long low-friction roll from the spot can then
# only leave play over the opponent end line, never the touchline.
KICKOFF_PASS_Y_M = 1.2

# A ball moving less than this between frames counts as stationary, which is
# what allows a repeated first-touch strike; a moving ball is never re-struck.
BALL_STILL_EPS_M = 0.05

# Indirect free kick: a goal needs a second touch, so the first touch passes
# this far forward of the spot toward this lateral offset instead of shooting.
# Spots deep in the opponent half pass toward the nearer flank, never backward.
SET_PLAY_PASS_AHEAD_M = 2.0
SET_PLAY_PASS_EDGE_MARGIN_M = 1.0

# Defensive throw-in: the power ramps with depth in our half. Near the halfway
# line the ball only needs to escape the press, so the touch stays soft; deep
# in our half it must travel, ramping up to the backfield clear power.
THROW_IN_POWER_MIN = 2.5
THROW_IN_RAMP_M = 3.5
# Aim this far upfield and this far toward midfield, so the ball comes off the
# touchline into open grass instead of hugging the side line.
THROW_IN_AHEAD_M = 3.0
THROW_IN_INWARD_M = 1.5

# Attacking restarts (our throw-in or corner in the opponent half) pass to a
# receiver instead of shooting from the restart. The deep receiver waits at
# the penalty spot; the short receiver sits this far along the ball-to-goal
# lane.
SET_PLAY_RECEIVE_DIST_M = 3.4

# Opponent set plays: field boundary margin for retreat targets (m).
OPP_SET_PLAY_FIELD_MARGIN_M = 0.3

# Opponent set-play wall: two blockers hold the ball-to-goal lane at these
# distances from the ball (both beyond the 3 m inactivity line and far above
# the 1.45 m clearance line), and a third spreads sideways from the near
# blocker when the goal itself is too close to guard.
OPP_SET_WALL_NEAR_M = 3.4
OPP_SET_WALL_FAR_M = 4.6
OPP_SET_WALL_SPREAD_M = 1.2

# No-activity rule: standing still for 10 s with the ball within 3 m draws a
# penalty. Waiting and standby points keep this distance, with margin.
INACTIVITY_CLEAR_M = 3.2


# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)
# Aim this far inside the near post when the goal center is covered.
GOAL_CORNER_INSET_M = 0.3

# ======================================================================
# Obstacle geometry
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball obstacle radius (m)
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth used for obstacle modeling (m)
POST_RADIUS = 0.18                      # Goal-post radius (m)
NET_RADIUS = 0.20                       # Goal-net sample radius (m)
NET_STEP = 0.35                         # Goal-net sampling interval (m)

START_IGNORE = 0.0                      # Ignore obstacles near the start (m)
TARGET_IGNORE = 0.0                     # Ignore obstacles near the target (m)

# ======================================================================
# Global path planner (A* grid planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Fall back to local VFH when disabled
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid cell size; smaller is finer (m)
GLOBAL_FIELD_MARGIN_M = 0.25            # Extra field margin for boundary paths (m)
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Additional obstacle inflation radius (m)
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # Lookahead distance along the path (m)

# ======================================================================
# Local path planner (VFH direction scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe length (m)
PLAN_CLEARANCE = 0.35                   # Minimum clearance for a candidate (m)
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad)
PLAN_MAX_OFFSET = math.radians(100)     # Maximum offset from target heading (rad)

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
