# Construction Safety Risk Analysis Technical Report

## Model

- Path: `models/best.pt`
- Task: YOLO object detection
- Classes: 10
- Embedded names match `data/data.yaml` exactly.

```text
0 Hardhat
1 Mask
2 NO-Hardhat
3 NO-Mask
4 NO-Safety Vest
5 Person
6 Safety Cone
7 Safety Vest
8 machinery
9 vehicle
```

## Dataset

`data/data.yaml` defines a YOLO object-detection dataset:

```yaml
train: ./train/images
val: ./valid/images
test: ./test/images
nc: 10
```

The dataset contains YOLO label files in the train, valid, and test splits. It
has no classes for footwear, falls, scaffolding, working at height, dangerous
edges, or missing fall protection.

## Supported Evidence

The model-supported hazards are `NO-Hardhat`, `NO-Mask`, and `NO-Safety Vest`.
Machinery and vehicles contribute only when their bounding boxes are near a
model-detected `Person`. Worker presence alone is not treated as a PPE
violation.

## Scoring Rules

- Each detected supported PPE violation contributes 25 raw points.
- Machinery and vehicles are reported objects only and do not contribute risk.
- Multiple PPE violation types add 10 points.
- Multiple affected workers add 5 points per additional worker.
- Multiple PPE violations associated with one worker add 10 points.
- The raw score is normalised to 0-100 using `MAX_RAW_SCORE = 100`.
- Risk levels remain Low 0-24, Medium 25-49, High 50-74, Critical 75-100.
- Video level uses the greater of average frame score and the midpoint of
  average and maximum frame score. This preserves peak evidence without
  assigning a label from visual judgment alone.

## Same Video Rerun

Video: `Screen Recording 2026-09-02 192725.mp4`

Settings: confidence `0.25`, every 10th frame, maximum 30 frames.

```text
Frames analysed:     30
Avg workers/frame:   0.43
Max workers/frame:   2
Avg hazards/frame:   0.17
Max hazards/frame:   1
Avg objects/frame:   1.03
Avg risk score:      2/100
Max risk score:      18/100
Peak-aware score:    10/100
Final risk level:    Low
```

Detected classes:

```text
Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest, machinery
```

Raw inference found no detections in frames 0 through 110 at the selected
sampling interval. Workers began appearing in later sampled frames. The
annotated diagnostic frames are saved in `outputs/video_debug/sample_1.jpg`,
`sample_2.jpg`, and `sample_3.jpg`.

## Limitations and Milestone 2

This model cannot directly identify working-at-height hazards, falls, unsafe
scaffolding, dangerous edges, unsafe climbing, or missing fall protection.
Those conditions must not be inferred from `Person`, `machinery`, or PPE classes.

Milestone 2 should use a labelled construction-hazard dataset and add classes
such as `working_at_height`, `fall`, `unsafe_scaffold`, `missing_fall_protection`,
`dangerous_edge`, and `unsafe_climbing`. A validated pose, geometry, or tracking
method could supplement the detector, but no such method is currently
implemented here.
