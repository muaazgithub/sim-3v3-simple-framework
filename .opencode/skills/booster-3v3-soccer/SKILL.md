---
name: Booster 3v3 Soccer
description: Strategy project for the Booster Champion 3v3 simulated robot soccer tournament - verified design decisions, rule traps, tuning knobs, and the offline-check workflow for sim-3v3-simple-framework
---

# Booster 3v3 Soccer strategy

Working rules for editing this repository. Read AGENTS.md first; this skill
adds what we learned the hard way, with the match evidence behind it.

Baseline of record: git tag `baseline-10match-1loss` (10 matches, 1 loss,
player report 2026-11-22). Compare any experiment against it.

## Hard platform facts (verified)

- `play()` runs ~30 Hz; one high-level action per player per frame. No sleep,
  no heavy work, no IO.
- `kick()` is a **sustained task**: the robot walks to the ball and completes
  the kick until `release_kick()`. `walk_to`/`stop` release it first.
- The ball cannot be moved directly - only through `kick()`.
- Nothing here runs on the host; all behavioural verification happens in
  Booster Studio (virtual robot, activate-build-deploy-run). Offline checks
  (below) cover logic, never physics.

## Technical compliance (official rule II - can forfeit the match)

Only public interfaces are allowed: own sensor/odometry data, GameControl
broadcasts, the official SDK/ROS/agent API, and our own in-process
computation. Reading `context.game` (score, `secs_remaining`, penalised
flags) is explicitly permitted - score-aware play is legal. Conversely,
NEVER add to `src/`: sockets, HTTP, file or log reads, `/proc` scraping,
environment probing, or any simulation/GameControl control call. The
offline check scripts are host-side dev tools only; they never ship in
the agent container.

## Official tournament rules (authoritative source)

Player pasted the official Competition Rules on 2026-11-22; everything in
this section is from that document, not inference. AGENTS.md section 5
defers to this skill when they disagree.

### Parameters

| Item | Value |
|---|---|
| Field | 14.0 x 9.0 m, line width 0.05 m |
| Goal | 2.6 m between posts, 1.8 m high, 0.6 m deep |
| Center circle | radius 1.5 m |
| Penalty area / goal area | 3.0 x 6.0 m / 1.0 x 4.0 m, inward from the goal line |
| Ball radius | 0.11 m - the WHOLE ball must cross a line for a goal OR out of bounds |
| Match | 10 min, no halves (simulation time); knockout ties: golden-goal extra time in repeated 5 min blocks |
| Send-off penalty | unified 30 s |
| Goal kick spot | approx. x=+-6.0, y=+-2.0 |
| Corner spot | 0.05 m inward from the corner arc |

### Clocks

| Phase | Limit |
|---|---|
| Ready -> Set | early after 5 s stable, timeout 45 s |
| Set -> Playing | 5 s |
| Kick-off set play | 10 s until possession opens |
| Throw-in / goal kick / corner | 45 s until possession opens |
| Local deadlock (no touch in normal play) | 30 s -> drop ball at center |

When a set-play clock expires the ball goes `ball_free` and both teams may
play it. We never approach these limits: takers strike on the first frame
of the phase. If a DEFENDER touches before the kicking team (R-06), he is
sent off 30 s and the set play is RETAKEN - so never touch early.

### Rule lines and their code consequences

| Rule | Consequence in code |
|---|---|
| M-05: in playing, ball within 3 m + nearly motionless 10 s = sent off | All waiting points sit at `INACTIVITY_CLEAR_M` (3.2 m); the opp-set keeper gate also protects this (see below) |
| P-04/R-07: defender < 1.45 m from the ball at the moment the kicking team touches = sent off 30 s + retake | Opponent-set-play wall points 3.4/4.6 m |
| R-07 exemptions: moving away > 0.05 m/s within 500 ms, OR standing on own goal line within goal width | A keeper ON the goal line is exempt from the DISTANCE rule - but NOT from M-05; never park him within 3 m of a stationary set-play ball |
| G-03: direct kickoff goal needs >= 2 touches (or the 10 s expired) | Kickoff passes first; the receiver's shot is touch 2 - valid. `KICKOFF_BALL_MOVED_M` (0.6) is our margin above the 0.5 m line |
| F-08 indirect free kick: goal valid only with >= 2 robots touching, or a touch from > 0.5 m off the spot; the DEFENDING team may shoot directly | Indirect set plays pass first at 2.0 |
| G-04: own goal from OUR set play (<= 1 touch, not expired) = no goal, OPPONENT awarded a corner | Danger-zone staged clearance stays mandatory - a corner against us still allows a direct goal |
| Corners: a goal CAN be scored directly from a corner | Attacking corners pass to receivers (corner geometry kills the direct angle); defending corners get wall + keeper |
| P-02/P-03 kickoff positioning: kicking team own half or inside the circle (max 1 robot in it); non-kicking team own half and outside the circle | Taker alone at the spot; receiver holds his reset position; everyone else stays in our half |
| M-01/M-02: set path > 0.15 m per 1 s; stop path > 0.3 m per 2 s (after grace) | `STOPPED` phase stops everyone; covers SET, pause, missing referee state |
| M-04: fallen (upDot <= 0.45) for 10 s = sent off | Penalised/fallen robots are filtered out of role assignment |
| G-02: goal needs the whole ball over the front plane, else out of bounds | Near-miss shots that graze the line become goal-line outs; clearance targets are clamped 0.3 m inside the lines |
| R-08: 30 s no-touch in normal play -> drop ball at center, circle closed during its set | Chasing play makes deadlock rare; `STOPPED` covers the drop-ball set |
| No local timing for possession decisions | Ball motion judged by frame-to-frame position (`_ball_stationary`, `store.prev_ball`), never by clocks |

### Correction vs our earlier assumption

The official document restricts direct goals from KICKOFF (G-03) and
INDIRECT free kicks (F-08) only. A direct goal from a GOAL KICK is not
listed as invalid, and direct corner goals are explicitly valid. We still
pass the goal kick - the player's standing decision - but do not cite
"direct goal-kick goals do not count" as a rule.

## Verified design decisions (do not regress these casually)

1. **Never re-strike a rolling ball.** `kick()` was re-issued every frame
   while the ball stayed within `KICKOFF_BALL_MOVED_M` of the spot; the ball
   accumulated strikes and flew out. `_ball_stationary` + `store.prev_ball`
   gate every restart strike (labels `kickoff:striking`, `setplay:striking`,
   `throwin:striking`).
2. **Danger zone staging causes own goals.** Staging behind the ball relative
   to the aim puts the robot between the ball and OUR goal line; with the
   ball near our line that point fell inside our own goal. Ball deeper than
   `DEFENSE_DANGER_DEPTH_M` (3.5 m) is cleared upfield from an upfield
   approach (`_clear_plan`), and every chase point is clamped inside the
   field (`CHASE_FIELD_MARGIN_M`).
3. **Clearance bias direction matters.** Biasing toward the same side as the
   ball carried balls out over the touchline from our own corner (reverted
   once). If revisiting, bias away from the nearer touchline and clamp.
4. **Restart first touches pass to teammates, not at opponents.** The
   straight kickoff tap was intercepted every match, so the kickoff now
   passes to the teammate farthest from our goal (the wide flank option,
   power 2.0) and the goal kick passes to the most advanced teammate (5.0).
   An earlier backward kickoff pass read as odd - the difference here is
   aiming at a specific teammate who then holds position to meet the pass
   (kickoff:receive) instead of freezing near the spot.
5. **Restart powers are role-specific:** kickoff tap 2.0 (straight), throw-in
   ramp THROW_IN_POWER_MIN(2.5) -> backfield by depth, goal kick 5.0,
   attacking set plays pass to receivers at 2.0. Players report these feel
   right; change one knob at a time and re-test.
6. **Keeper sweeps only what it should:** within `GUARD_SWEEP_M` (1.5 m) AND
   slower than `GUARD_SWEEP_MAX_MPS`; fast shots are met via the predicted
   crossing point instead. Velocity estimates above `GUARD_VELOCITY_CAP_MPS`
   are restart teleports, not shots - ignore them.
7. **Beware float tie-breaks.** A one-ULP distance difference once flipped a
   pushed stand-off point to the backward side. Round capped keys before
   `max()`/`min()` comparisons on geometry.
8. **Sign every "inward" bias toward midfield, then clamp.** A positive
   inward offset on a ball already at the touchline aimed every defensive
   throw-in OUT of play on both sides of the field - and the test mirrored
   the same wrong formula, so only an in-field assertion caught it.
9. **Opponent set play = wall, not scatter:** two blockers on the
   ball-to-goal lane (3.4/4.6 m) plus keeper only when the goal is at least
   `OPP_SET_GUARD_GATE_M` (3.5 m) from the ball; otherwise a spread blocker
   replaces him. The gate is M-05 protection: with the ball closer, the
   keeper at the goal-area centre would sit ~2.7-3.0 m from a stationary
   ball, inside the 3 m/10 s trap while a slow taker stalls.
10. **Goal kicks and throw-ins hold the first touch AT the ball.** The
   hold counts from when the taker is standing at his settle spot -
   `SET_PLAY_SETTLE_M` (0.8 m) behind the ball on the line opposite the
   kick direction - not from phase entry. Standing metres away read as
   dead time and the long run-up once nudged touchline balls straight out
   of play. The at-ball hold is `SET_PLAY_HOLD_AT_BALL_S` (8.0 s), under
   the 10 s M-05 window with margin; a fallback from phase entry
   (`SET_PLAY_READY_MAX_S`, 15 s) strikes anyway if the approach is
   blocked. Corners and free kicks still strike on frame one.
11. **Late-game urgency bends only the support role.** Trailing inside
   URGENT_ATTACK_SECS (120 s) stands the supporter ahead of the ball on the
   ball-to-their-goal line; leading inside URGENT_DEFEND_SECS (90 s) parks
   him at most SUPPORT_DEFEND_DEPTH_M (2.8 m) from our goal. Presser and
   keeper never change; a missing broadcast or zeroed clock reads normal so
   a broken packet cannot flip the team permanently.
12. **Lock episode-long roles in store.** The set-play taker was re-ranked
   every frame; the hold-off walks him away from the ball, so the ranking
   flipped and the robots fought over the kick. The taker is locked in
   `store.setplay_taker_id` on entry and re-selected only when he is no
   longer playable - the AGENTS.md hysteresis rule.
13. **An unknown set-play kicker is the OPPONENTS' ball.** `get_phase`
   once fell through to NORMAL when `kicking_team` was not yet broadcast,
   sending the nearest robot to touch their dead ball (R-06 send-off plus
   retake). Any active set play with an unknown kicker now reads as
   OPP_SET_PLAY: defend, never touch.

## Offline checks (before any Studio run)

`scripts/check_*.py` encode the current behaviour with a stubbed
`booster_agent_framework` (no ROS needed):

```sh
cd <repo>
for f in .opencode/skills/booster-3v3-soccer/scripts/check_*.py; do
  PYTHONPATH=. python3 "$f"
done
python3 -m py_compile src/main.py src/player.py src/param.py
```

They stub `AgentBase`/`AgentFeatures` and fake players, then assert roles,
powers, directions and rule clearances. Update them **with** every behaviour
change - they are the executable spec. Never claim a change works from
reasoning alone: compile + offline checks + one full Studio match.

## Tuning knobs that map to player feel

- Hard shots: `KICK_POWER_DEFAULT` (8.0); deep clears: `KICK_POWER_BACKFIELD`
  (9.0). Near-cap power lofts the ball and can drop robots.
- Kickoff pass strength: `KICK_POWER_RESTART_PASS` (2.0, wide pass to the flank teammate).
- Throw-in softness near halfway: `THROW_IN_POWER_MIN` (2.5); ramp length
  `THROW_IN_RAMP_M` (3.5 m).
- Goal kick strength: `GOAL_KICK_POWER` (5.0).
- Setup hold-off for goal kicks and throw-ins: settle distance
  `SET_PLAY_SETTLE_M` (0.8 m), at-ball hold `SET_PLAY_HOLD_AT_BALL_S`
  (8.0 s), blocked-approach fallback `SET_PLAY_READY_MAX_S` (15 s).
- Late-game urgency: `URGENT_ATTACK_SECS` (120), `URGENT_DEFEND_SECS` (90),
  parked support depth `SUPPORT_DEFEND_DEPTH_M` (2.8 m).
- Keeper aggression: `GUARD_SWEEP_M`, `GUARD_SWEEP_MAX_MPS`.
- Support distance: `SUPPORT_DIST_M` (3.3, above the 3 m inactivity line).

Tune parameters before touching logic; one variable at a time.

## Known open weaknesses (as of the baseline)

- No strike-alignment gate: `attack()` kicks from the wrong side of the ball
  (reverted with 0f5ccd0; re-add carefully if shots spray again).
- Keeper role is not sticky: two robots with similar goal distance can swap
  the role every frame (reverted with 94940f5).
- Keeper will sweep a ball that is behind it (same reverted commit).
- When every shot lane is covered, the robot still shoots at the keeper
  instead of holding or passing.
- No search behaviour when the ball is out of view (everyone stops).

## Revert discipline

Prefer `git revert` (keeps history) over `git reset --hard` + force push.
When a multi-fix commit turns out bad, revert it as a block, then re-apply
the good parts one commit at a time - that is how 0f5ccd0/94940f5 were
unwound into 8ec131a. Tag known-good states so matches can be compared.
