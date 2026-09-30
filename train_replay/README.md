# Battlecode 2026 "Snake" — replay → imitation-learning data

This folder turns the binary `.replay` files (UNSW **Battlecode 2026 "Snake"**,
Python bots via `unswbc`) into per-dragon **state → action** training data that
you can use for imitation / behavioural cloning.

The replay format is **Cap'n Proto messages stored with packed compression**.
There is no public Python Cap'n Proto reader that copes with this schema (pycapnp
crashes on schemas this size), so the decoder is a small, self-contained,
pure-Python port of the official replay-viewer JS codegen.

## Files

| file | purpose |
|------|---------|
| `reader.py` | Pure-Python Cap'n Proto wire reader: unpackes packed bytes, splits segments, follows pointers, reads structs/lists (offsets & struct sizes taken from `unswbc`'s `replay-viewer.vsix` → `webview.js`). |
| `decode_replay.py` | High-level decoder: produces `meta` (map text, result) + a flat list of all decoded events (`dragonAction`, `dragonUpdate`, `dragonSplit`, `dragonDeath`, `sonarPing`, …). |
| `extract_training_data.py` | **Training-data generator.** Turns a folder of `.replay` files into per-dragon, per-round `{team, dragonId, round, head, facing, action}` records, with dragon→team labelling. |
| `replay.capnp` | Layout reference for the schema (not needed to run, only as documentation). |

## 1. Decode a replay (raw event stream)

```bash
python decode_replay.py ..\..\replays\battle-M647381-replays\M647381.replay
#   map name, botA, botB, total events, per-type breakdown
python decode_replay.py file.replay -o events.json
```

## 2. Build imitation-learning data (per dragon, per round)

```bash
# all replays in a folder:
python extract_training_data.py ..\..\replays\battle-M647381-replays --out training_data_all.json
# only one team (0 = A, 1 = B):
python extract_training_data.py ..\..\replays\battle-M647381-replays --team 1 --out team_1.json
```

Each record has the shape:

```json
{"replay":"M647381.replay","map":"Queen Of Spades","team":1,"dragonId":0,
 "round":3,"facing":"S","head":{"x":5,"y":14},
 "event":"move","steps":["S"],"tle":false}
```

`event` is one of `move` (has `steps`, one or more direction letters), `split_action`
(`childSegmentCount`), `split` (parent spawned `childId`), or `suicide`.

## 3. How dragon → team is resolved

* The map text carries the initial roster: `DRAGON <team> <len> <head_x> <head_y> ...`
  for every starting dragon. The first batch of `dragonUpdate` events (ids 0..N-1)
  are matched to those roster entries **by head position**, giving each starting id
  a team.
* Every `dragonSplit` event carries a `team` field, so each child id inherits the
  parent's team — the id→team map grows to cover the whole game.

Team `0` doubles as team **A**, team `1` as team **B** (the same numbering used in
`teamA`/`teamB` / `winner` in the game result).

## 4. Using the data for imitation learning

Each `(head, facing [, body])` is an observation; each `steps`/`split` is the
action the expert chose. A behavioural-cloning policy maps observation → action.
Typical feature encodings:

* Map/knowledge as a tensor: walls + your dragon body + enemies + pearls +
  sonar pings (all derivable by replaying `dragonUpdate`/`tileChange`/`sonarPing`).
* Action label: one of `N/E/S/W` (move) or `SPLIT`, aligned to the scheduled turn
  order (`turnStart id` → `dragonAction id`).
* Round-to-round: a dragon that tries to move into a wall/collision keeps the same
  state (its move is blocked) — cleanly flagged as the *inconsistent* pair, which
  you can either drop or keep as "legal vs illegal" supervision.

`instruction usage` (`count`, `exceeded`) is available on every `dragonAction` so
you can filter to within-budget decisions.

## Validation performed

On the 4 `battle-M647381-replays` files: **127,787 records** (120,606 moves,
7,116 splits, 65 suicides) across both teams, with 96%+ of consecutive-roundted
`move` records consistent with the resulting head displacement (the remainder are
blocked/illegal attempts, expected).

## 5. Behavioural-cloning policies (one per team) — the "Cache me outside" replays

The four replays in `..\..\replays\battle-M647381-replays\` are the two-team match
you asked to imitate. The team name **is not stored** in the replay files (both
`botA`/`botB` are empty), so we trained **one policy per team (A=0, B=1)** and you
pick by behaviour. A clear behavioural tell: **team A splits far more often**
(≈2,035 split actions vs ≈190 for team B in training).

`build_features.py` rebuilds each dragon's real 7×7 wrapped vision window
(7-channel tensor: own-body / ally-body / enemy-body / any-head / pearl / kelp /
empty; VISION_RADIUS=3, toroidal, exactly how a bot sees the board) at every
`dragonAction`, from the decoded events. Body reconstruction uses the roster +
head-path walking (matched to `dragonUpdate` head/tail) + authoritative
parent/child bodies from `dragonSplit`. Pearl growth honours `tileChange` /
countdown. Edge (kelp/portal) indexing is ported verbatim from the replay-viewer
JS (`hEdges=(H+1)W`, `vEdges=H(W+1)`, N/S/W/E lookups).

`train_bc.py` trains a small CPU MLP (343 features + 2 scalars → 6 outputs
`N/E/S/W/SPLIT/SUICIDE`) and reports train/val accuracy plus the per-class action
distribution. Use **the base Python** (`...Python311\python.exe`) — it has torch;
the project venv has no ML libraries.

```bash
# data + train + save for each team (20 epochs; ~1-2 min each on CPU)
python train_bc.py ..\..\replays\battle-M647381-replays --team 0 --out ...\policy_team_A --epochs 20
python train_bc.py ..\..\replays\battle-M647381-replays --team 1 --out ...\policy_team_B --epochs 20

# load a model and print a sample prediction
python train_bc.py --predict ...\policy_team_A-model.pt

# run a real policy over actual game states of a replay (agreement + per-class)
python demo_policy.py ...\policy_team_A-model.pt ...\M647381.replay
```

Saved artifacts (written next to the `--out` prefix):
`-model.pt` (state_dict + class counts + scaling), `-dataset.npz`,
`-obs.npy`, `-scalar.npy`, `-labels.npy`.

In your own bot, load the policy and act each round:

```python
from replay_analysis.train_bc import load_policy
from replay_analysis import build_features as bf   # build the same 7x7 obs online
predict, names, _ = load_policy("...\\policy_team_A-model.pt")
action, probs = predict(my_7x7_obs, np.array([round/500, my_length/500]))
# action: 0=N 1=E 2=S 3=W 4=SPLIT 5=SUICIDE
```

### Results (20 epochs, CPU, train/val 80/20 per team, all 4 matches merged)

| team | samples | class dist (train) | val acc | train acc | majority-vote baseline |
|------|--------|--------------------|---------|-----------|------------------------|
| A (0) | 59,695 | N11495 E12006 S10452 W11768 SPLIT2035 SUICIDE0 | **0.525** | 0.652 | 0.251 |
| B (1) | 57,432 | N10821 E11946 S10392 W11977 SPLIT754 SUICIDE56 | **0.482** | 0.614 | 0.260 |

Both models beat the "always move the most common way" baseline by **≈2×**. On
real game states (`demo_policy.py`, replay M647381) they agree with the expert
on **~64%** of Dragon-Actions with sensible per-direction accuracy and no
degenerate "always N" behaviour.

