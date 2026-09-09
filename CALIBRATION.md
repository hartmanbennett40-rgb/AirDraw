# Calibration & Known Limitations

AirDraw draws with your fingertip by combining two independent tracking
systems every frame: MediaPipe hand-landmark detection (finds your hand and
whether you're pinching) and Lucas-Kanade optical flow on background
features (used only to estimate a tracking-confidence score, not to move
the drawing). Both are webcam-image techniques and both degrade under the
same real-world conditions. This doc covers how to set up your space, how
to tune the pinch gesture, and what will actually break.

## Recommended lighting / background

- **Even, frontal light.** Light the camera side of your face/hands, not
  behind them. Backlighting (window or lamp behind you) silhouettes your
  hand and MediaPipe's detector confidence drops or drops out entirely.
- **Avoid flicker.** Fluorescent/cheap LED lighting captured at some
  webcam shutter speeds produces visible banding/flicker between frames,
  which shows up as noise to optical flow. Natural light or a steady
  continuous-output lamp is best.
- **Textured, static background.** Optical flow needs corners/edges to
  lock onto (`cv2.goodFeaturesToTrack`). A blank wall, a solid-color
  backdrop, or an out-of-focus background gives it almost nothing to
  track — confidence will sit in yellow/red even when your hand tracking
  is fine. A bookshelf, textured wall, or cluttered desk behind you works
  much better than a plain wall.
- **Contrast between hand and background.** Skin-toned walls/desks near
  skin-toned hands reduce MediaPipe's segmentation confidence. Any
  reasonably contrasting background helps.
- **Camera distance:** keep your hand roughly 30-70cm from the camera,
  fully in frame. Too close blurs it (rolling shutter/focus), too far and
  the hand occupies too few pixels for reliable landmarks.

## Pinch threshold

Pinch detection is in `airdraw/hands.py` (`HandResult.pinch_distance` /
`is_pinching`) and the working threshold is `PINCH_THRESHOLD` in
`airdraw/app.py`, currently **0.45**.

The raw thumb-tip-to-index-tip pixel distance is **not** used directly,
because it scales with hand size and distance from the camera. Instead it's
normalized:

```
pinch_distance = distance(thumb_tip, index_tip) / distance(wrist, middle_finger_mcp)
is_pinching     = pinch_distance < PINCH_THRESHOLD
```

The denominator (wrist to middle-finger-MCP) is a stand-in for "hand size in
this frame," so the same threshold works whether your hand is close-up or
far away, adult-sized or a kid's hand.

**How to tune it per hand size:**

1. Run the app and pinch normally; note whether draw strokes start too
   early (thumb/index still visibly apart) or too late (need to squeeze
   hard before drawing starts).
2. Lower `PINCH_THRESHOLD` (e.g. `0.35`) if pinches are registering too
   easily / drawing triggers accidentally when your fingers are just
   resting near each other.
3. Raise it (e.g. `0.55`) if you have to pinch very tightly before drawing
   starts, which is common for people with larger hands/longer fingers
   where a "loose pinch" still measures as a larger relative distance.
4. Small hands (kids, or a hand far from camera) generally need **no**
   change — that's the point of the normalization — but if MediaPipe's
   landmark localization is noisy at that scale (small hand = fewer
   pixels = noisier landmark positions), the pinch distance itself will
   jitter more frame-to-frame, so consider adding light smoothing rather
   than changing the threshold if you see false pinch/unpinch flicker.

There is currently no runtime UI to tune this — pass a different value to
`App(pinch_threshold=...)` or edit the constant.

## Known failure modes

- **Fast head or laptop movement loses tracking.** Both systems assume the
  scene doesn't move much between frames. If you shift a laptop webcam or
  move your head quickly, MediaPipe's per-frame hand detection can miss a
  frame (hand moved out of its predicted crop) and optical-flow features
  from the previous frame won't be found near their old positions —
  `calcOpticalFlowPyrLK` match rate drops, so the confidence indicator
  should visibly go yellow/red right as this happens. Expect a stroke to
  skip or jump when this occurs; it will recover within a few frames once
  motion settles, but nothing currently interpolates across the gap.
- **Low-texture backgrounds break optical flow, not MediaPipe.** A plain
  wall or blurred background gives `goodFeaturesToTrack` few or no strong
  corners to seed, so the confidence indicator will sit at yellow/red
  persistently even though hand tracking may be working fine. This is a
  false alarm in the sense that drawing accuracy isn't actually degraded
  (the indicator only reflects background flow, it does not correct or
  gate drawing) — but it does mean the indicator is uninformative in that
  environment. Point the camera at a more textured background if you want
  the indicator to be meaningful.
- **Poor lighting breaks both.** Low light increases sensor noise, which
  both hurts MediaPipe's landmark confidence (missed/jittery detections)
  and optical flow's ability to match features frame-to-frame (noise looks
  like motion). This is the one condition where a red confidence reading
  correlates with genuinely unreliable hand tracking too — treat a
  persistent red indicator in low light as "add more light," not "ignore
  it."
- **Single-hand only.** The tracker is configured for one hand
  (`max_hands=1`); a second hand in frame is ignored, not confusing, but
  also not usable.
- **No occlusion recovery.** If your hand is briefly blocked (crosses in
  front of your face, leaves frame), the gesture state machine treats the
  next detected point as a fresh "move," not a continuation — you won't
  get a stray line across the gap, but you will lose the in-progress
  stroke.
