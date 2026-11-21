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
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7
# When threatened, the keeper tracks the ball's y across the goal mouth plus
# this margin instead of sitting at the goal-area center.
GUARD_TRACK_MARGIN_M = 0.5

# Stand-off distance from the ball. Waiting here means standing still, so keep
# above the 3 m inactivity line with margin (see INACTIVITY_CLEAR_M).
SUPPORT_DIST_M = 3.3

# The keeper treats the ball as in enemy possession when the nearest robot to
# the ball is an opponent by at least this margin (m). The margin adds
# hysteresis so the verdict cannot flip every frame in a 50/50 race.
GUARD_POSSESSION_MARGIN_M = 0.25

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

# A goal straight from kickoff does not count: the taker may only shoot once
# the ball has left this radius around the kickoff spot (referee line is
# 0.5 m; keep margin). Until then the first touch passes instead of shooting.
KICKOFF_BALL_MOVED_M = 0.6
KICKOFF_PASS_X_M = 2.5             # First-touch pass target (field x, m)
# Shallow lateral offset: even a long low-friction roll from the spot can then
# only leave play over the opponent end line, never the touchline.
KICKOFF_PASS_Y_M = 1.2

# Kickoff first touch: roll the ball this far in front of the opponent goal
# line, then chase it immediately and shoot. Power derives from the roll
# calibration constant, so the touch always stops short instead of sailing
# out of the field like a fixed-power kick does.
KICKOFF_ROLL_DEPTH_M = 2.0

# A ball moving less than this between frames counts as stationary, which is
# what allows a repeated first-touch strike; a moving ball is never re-struck.
BALL_STILL_EPS_M = 0.05

# Indirect free kick: a goal needs a second touch, so the first touch passes
# this far forward of the spot toward this lateral offset instead of shooting.
# Spots deep in the opponent half pass toward the nearer flank, never backward.
SET_PLAY_PASS_AHEAD_M = 2.0
SET_PLAY_PASS_EDGE_MARGIN_M = 1.0

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
# Ball motion estimates
# ======================================================================

# Exponential smoothing factor for the ball velocity estimate (0..1); higher
# values trust the newest frame more.
BALL_VEL_SMOOTHING = 0.45
# A ball velocity estimate spanning more than this much time is discarded (s);
# it would mix in the unknown motion during missed detections.
BALL_VEL_MAX_DT_S = 0.4


# ======================================================================
# Shot selection gate
# ======================================================================

# Only shoot when the ball is within this fraction of the field length from
# the opponent goal center; farther out, dribble or pass instead.
SHOOT_MAX_RANGE_FRAC = 0.7
# Only shoot when the ball-to-goal direction stays within this angle of the
# +x axis (radians); acute side angles produce wild shots.
SHOOT_MAX_ANGLE_RAD = 1.1
# Do not commit to a shot while the ball moves faster than this (m/s); a kick
# on a fast rolling ball goes anywhere.
SHOT_MAX_BALL_SPEED_MPS = 1.8
# A shot counts as open only when its line clears every opponent's center by
# this much (OPPONENT_RADIUS plus margin for kick-direction error).
SHOT_MIN_CLEARANCE_M = 0.6

# Keep the latched aim seam unless another candidate is clearly wider (m).
AIM_SWITCH_MARGIN_M = 0.3


# ======================================================================
# Kick power calibration
# ======================================================================

# Ball roll calibration: rough meters of roll per unit of kick power on this
# field's friction. Derived from the kickoff pass sailing out of play at
# power 2.0 (the end line sits 7.6 m away on that diagonal, so power 2.0
# rolls at least that far). Every distance-based kick derives its power from
# this constant; tune it after watching touch distances in Studio.
KICK_ROLL_M_PER_POWER = 3.8
# A shot rolls this much farther than the distance to its target, so it
# arrives with pace instead of dying short of the goal line.
SHOT_ROLL_MARGIN = 1.25
# A pass rolls this much farther than the distance to its receiver.
PASS_ROLL_MARGIN = 1.1
# A wide-open shot lane clears every opponent's center by this much (m);
# then the bot shoots as hard as it can instead of a calibrated roll.
SHOT_HARD_CLEARANCE_M = 1.2
# Power for the hardest shots. The firmware can fall on near-cap kicks, so
# this sits below the 10.0 cap; raise it toward 10.0 if the sim shows no
# falls after these shots.
SHOT_MAX_POWER = 8.0


# ======================================================================
# Dribbling
# ======================================================================

# Soft kick power for dribble touches; on this field even the minimum power
# rolls several meters, so dribbling is push-chase rather than close control.
KICK_POWER_DRIBBLE = 1.0
# A dribble direction must clear obstacles by this much over PLAN_LOOKAHEAD (m).
DRIBBLE_CLEARANCE_M = 0.45


# ======================================================================
# Tackling
# ======================================================================

# Within this distance of the ball an opponent counts as holding it (m).
OPP_BALL_HOLD_DIST_M = 0.9


# ======================================================================
# Normal-play defense positioning
# ======================================================================

# Enter defensive assignment when the ball crosses this far into our half (m);
# leave it again only past the exit line (hysteresis).
DEFEND_ENTER_X_M = -0.5
DEFEND_EXIT_X_M = 0.5
# Ball velocity toward our goal that counts as an attacking ball (m/s on x).
INTERCEPT_MIN_VX_MPS = 0.8
# Field defenders meet a ball rolling toward our goal this far ahead of it (m).
INTERCEPT_LEAD_M = 2.0
# The defending support stands on the ball-to-goal lane this far from the ball
# (m), beyond the 3 m inactivity line with margin.
DEFEND_WALL_DIST_M = 3.4
# The keeper predicts the ball path and sprints to its goal-line crossing
# point once the ball moves toward our goal at least this fast (m/s).
KEEPER_INTERCEPT_MIN_SPEED_MPS = 1.2
# When the ball comes this close, the keeper clears it upfield at once (m):
# the enter radius, and the wider exit radius with hysteresis so an
# in-flight clearance is not aborted while the ball is still nearby.
KEEPER_CLEAR_ENTER_M = 1.5
KEEPER_CLEAR_EXIT_M = 2.0
# Whether the keeper models the ball as an obstacle while blocking. False:
# getting the body onto the shot line is the job, so the keeper walks into
# the ball's path instead of steering around it.
KEEPER_BLOCK_AVOID_BALL = False
# When the predicted shot crosses the goal line within this time (s), the
# keeper lunges: a direct maximum-speed omnidirectional burst at the crossing
# point, with no turning or avoidance, to throw its body onto the shot line.
# A fall from the lunge is recovered automatically by the framework.
KEEPER_LUNGE_TIME_S = 1.0
# Optional SDK mode name for a real jump or dive save (for example "jump"),
# triggered when a shot is imminent. The SDK in this repository only
# documents "walk", so keep None unless the robot firmware confirms such a
# mode; the velocity lunge above works everywhere.
KEEPER_JUMP_MODE = None


# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)
# Aim this far inside the near post when the goal center is covered.
GOAL_CORNER_INSET_M = 0.3

# Aim points scanned across the goal mouth; more samples find narrower open
# seams between defenders than the old center-and-corners candidate set.
GOAL_AIM_SAMPLES = 7
# A shot counts as open only when its line clears every opponent's center by
# this much (OPPONENT_RADIUS plus margin for kick-direction error).
SHOT_MIN_CLEARANCE_M = 0.75
# Do not fire the kick while the planned shot direction points this far off
# the robot's heading (radians); reposition behind the ball facing the shot
# instead. A kick fired while facing away sends the ball backward.
SHOT_MAX_BODY_ERR_RAD = 0.6
# The bot must also stand behind the ball relative to the shot line: fire
# only when it sits at most this far ahead of the ball along the shot
# direction (m). Kicking from beside or past the ball sends it wherever the
# body points, not toward the goal.
SHOT_MAX_AHEAD_M = 0.05

# ======================================================================
# Blocked-lane passing
# ======================================================================

# Pass power at zero distance plus this much per meter of pass distance,
# capped at KICK_POWER_PASS_MAX. Full-power passes make robots fall.
KICK_POWER_PASS_BASE = 2.5
KICK_POWER_PASS_PER_M = 0.85
KICK_POWER_PASS_MAX = 8.0
# Only pass to teammates this far upfield of the ball, so a blocked shot
# never turns into a pass back toward our own goal.
PASS_BACKWARD_MARGIN_M = 0.5
# A pass lane must clear every opponent's center by this much (m).
PASS_MIN_LANE_CLEAR_M = 0.6
# Switch pass targets only when the new teammate is clearly closer (m), so
# the kick direction does not flap between teammates frame to frame.
PASS_TARGET_SWITCH_M = 0.8
# Once passing, keep passing until the shot lane reopens by this much above
# SHOT_MIN_CLEARANCE_M, preventing shot/pass oscillation at the threshold.
PASS_EXIT_MARGIN_M = 0.25

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
