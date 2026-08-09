# Two-game evaluator audit: official success vs local failure

Date: 2026-08-07

## Scope and evidence

The official validation run used 128 `valid_seen` environments with seeds
1000 through 1127. TextWorld maps each seed to its first game by copying the
140-game list and applying `numpy.random.RandomState(seed).shuffle(...)`.

The saved official run reports:

- overall success: 125/128 (97.65625%)
- `pick_and_place_simple` success: 100%

The local full274 run contains one trajectory for every seen game. Intersecting
the official seed-selected games with locally failed `pick_and_place_simple`
records produced 15 matches. The two examples below are therefore games that
the official aggregate proves successful but the local TCOD-style run marks
failed.

## Example 1: seed 1000, box -> dresser

- Game: `valid_seen/pick_and_place_simple-Box-None-Dresser-224/trial_T20190907_164104_229696/game.tw-pddl`
- Local record: `a474e30900a2cce3bb5404ea89d7ac17d177f09dc2b6ed2a165aa47999770122`
- Goal: `put some box on dresser.`
- Local result: `won=false`, reward 0, timeout at 50 steps
- Local admissible-action rate: 0.74

The local trajectory is genuinely unsuccessful. It starts:

1. `go to dresser 1`
2. `take book 1 from dresser 1`
3. `go to sidetable 1`
4. `go to sidetable 2` (box 2 is visible)

Because the agent is already holding the irrelevant book, `take box 2 from
sidetable 2` is no longer admissible. It never legally takes any box. Later
attempts such as `take box 3` and `move box 3 to dresser 1` return `Nothing
happens.` No box is ever placed on the dresser, so the local failure label is
correct.

The official log directly prints the same seed-1000 game state. Its first
actions are `go to dresser 1`, `go to sidetable 1`, `go to sidetable 2`, then
`take box 2 from sidetable 2`; the episode-level score printed for that sample
is 10.0. This is a behavioral divergence, not a disagreement about whether the
same terminal state counts as success.

## Example 2: seed 1032, spraybottle -> toilet

- Game: `valid_seen/pick_and_place_simple-SprayBottle-None-Toilet-415/trial_T20190909_075955_678702/game.tw-pddl`
- Local record: `85da793018e4634137c634e8d009442373c37391ac59c2c8e4cee979d253c38c`
- Goal: `put some spraybottle on toilet.`
- Local result: `won=false`, reward 0, timeout at 50 steps
- Local admissible-action rate: 0.74

This local trajectory is also genuinely unsuccessful. It starts:

1. `go to countertop 1` (spraybottle 1 is visible)
2. `take soapbar 1 from countertop 1`
3. `go to toilet 1`

Again, the wrong object is taken on step 2. The spraybottle never enters the
inventory. All later attempts to use, take, or put a spraybottle are invalid
and return `Nothing happens.` The final observation still shows the candidate
spraybottles at their source locations, not on the toilet. The local failure
label is correct.

Seed 1032 is part of the official 128-env run, and this game is
`pick_and_place_simple`. Since the official per-family success rate is exactly
1.0, this official environment necessarily succeeded. The old official log
does not preserve its action-by-action trajectory, so only the success status,
not its exact action sequence, can be recovered from the saved artifact.

## Root cause

The local evaluator selects the no-history template while
`len(history) < HISTORY_LENGTH`, where `HISTORY_LENGTH=2`. It therefore uses
the no-history template on both step 1 and step 2.

The no-history template does not have a separate `task_description` field. On
step 1 this is harmless because the initial TextWorld observation contains
`Your task is to: ...`. On step 2, the current observation only describes the
result of step 1, so the prompt contains no task at all. Both audited local
trajectories choose a plausible but irrelevant object on exactly this taskless
second prompt.

The official environment manager uses the no-history template only for the
initial prompt (`init=True`). Immediately after the first action it uses the
history template, which explicitly inserts the task description even though
only one history item exists.

The official action projection only lowercases and extracts the contents of
`<action>...</action>`; it does not replace a generated action with an
admissible one. Thus the official success is not caused by a permissive scorer
or an oracle action correction.

## Conclusion

For these two cases, our `won=false` labels are correct for the trajectories we
actually generated. The evaluation defect is upstream of scoring: the local
prompt/state construction drops the task on step 2 and changes model behavior.
The local 35% seen result must not be interpreted as this checkpoint's native
ALFWorld capability.
