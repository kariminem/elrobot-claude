---
name: elrobot-operator
description: Operate a real NormaCore ElRobot robot arm through the elrobot MCP tools (look, move_to, pick, put_ball_in_mug, go_home...). Use when asked to move the robot arm, pick up or place an object, go above something, or check what the robot's cameras see.
---

# Operating the ElRobot arm

You are moving a **real** arm next to people. Every motion tool returns both camera images (table camera +
gripper camera) and the arm's state. Read them before the next step.

## Ground rules

1. **Closed loop.** look → one move → look. Never chain several moves blind.
2. **Only report what the images show.** If the images do not clearly show success, say "unsure" and look again.
   Never explain away contradicting evidence (an object that "came back" means the previous step failed).
3. **Small steps.** move_to / move_by refuse moves over 15 cm or into the table: split them.
4. If anything looks wrong (a joint blocked, a refusal you don't understand), stop and tell the human.

## Coordinates

Centimetres in the arm's base frame: origin = centre of the base on the table, **+x = the arm's right**,
**+y = straight out, away from the base**, z = up from the table. The "fingertips" point is the centre between the
two jaw tips. `look` lists what the table camera recognises with (x, y) positions.

## Recipes

| Request | Do |
|---|---|
| "What do you see?" | `look`, then describe the objects and their positions |
| "Go above X" (no touching) | `look` → `move_to(x, y, z)` with z = object height + 8-10 cm (mugs are ~10 cm tall: z ≈ 20) |
| "Pick X" | `pick` (finds X with the table camera, then the gripper camera guides the grasp and closes) |
| "Put the ball in the white mug" | `put_ball_in_mug` (measures the mug with the gripper camera first, then picks, drops, checks) |
| "Put the ball on the cube" | `put_ball_on_cube` |
| "Go home" / "rest" | `go_home` (lifts straight up first when low) |
| "Show me what is happening" | `open_dashboard` and give the link (simulation=true if no robot) |
| Fine adjustment from the gripper view | `move_in_view(right_cm, up_cm, down_cm)` |

## How accurate things are

- **Table camera:** ±2-3 cm in the middle of the mat, up to 6-9 cm off at the edges and near the base.
- **Gripper camera:** precise up close; the composite tools use it to correct the table camera before a grasp or a drop.
- **The arm sags 1-3 cm** when stretched far out: correct with what the cameras show, not with the numbers alone.

## When something fails

- `REFUSED ... can't reach` → the target is too far or too low for that orientation: get closer or use `orient="free"`.
- `REFUSED ... max 15 cm` → split the move.
- "Station isn't reading the follower's motors (Motor ID mismatch)" → ask the human to restart NormaCore Station.
- A grasp `MISSED` → `look`, then try `pick` again; if the object is half outside the mat or under another object, say so.
