# Multi-Object Tracking & Analytics Architecture

## Overview

EdgeVision AI incorporates an edge-optimized, detector-agnostic **Multi-Object Tracking (MOT) and Analytics engine**. It operates downstream of the object detector, transforming per-frame detections into persistent spatiotemporal entities with trajectory history, motion analytics, and virtual tripwire event detection.

```
┌─────────────────────────────────┐
│     Object Detections           │
│  (BBoxes, Confidences, Labels)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Centroid Tracker           │
│  - Euclidean Distance Matrix    │
│  - Bipartite Linear Assignment  │
│  - ID Persistence & Lifecycle   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Spatiotemporal Tracks        │
│  - Track ID, Age, Missed Frames │
│  - Trajectory Breadcrumbs       │
│  - Bounding Box & Class         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Analytics Engine          │
│  - Motion Direction Estimation  │
│  - Virtual Tripwire (In / Out)  │
│  - Class & Count Aggregation    │
└─────────────────────────────────┘
```

---

## 1. Matching Strategy

The tracker uses spatial centroid Euclidean distance matching solved via optimal linear sum assignment (Hungarian algorithm):

1. **Centroid Extraction**: For each detection $D_j = (x_1, y_1, x_2, y_2)$, compute center:
   $$c_j = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$$
2. **Cost Matrix Construction**: Compute pairwise Euclidean distance between all active tracks $T_i$ and new detections $D_j$:
   $$C_{i, j} = \|c(T_i) - c(D_j)\|_2$$
   If `match_class=True`, pairs with mismatched class labels are penalized with an infinite cost ($10^6$).
3. **Bipartite Assignment**: Solve `scipy.optimize.linear_sum_assignment(C)` to find the globally optimal associations minimizing total distance.
4. **Gating Threshold**: Matches exceeding `max_distance` (default: 60.0 pixels) are rejected, preventing erroneous associations during fast motions or occlusions.

---

## 2. Track Lifecycle Management

Every track progresses through a deterministic state machine:

- **Creation**: An unmatched detection instantiates a new `Track` with an auto-incrementing persistent ID, `age = 1`, and `missed_frames = 0`.
- **Update**: When matched in a subsequent frame, the track updates its centroid, records displacement, appends the point to `trajectory`, increments `age` and `hits`, and resets `missed_frames = 0`.
- **Coast / Lost**: If an active track is not matched in a frame, `missed_frames` increments by 1. The track is marked `LOST` but preserved in memory.
- **Deletion / Deregistration**: If `missed_frames > max_missed_frames` (default: 12-15 frames), the target is permanently removed to prevent ghost tracks.

---

## 3. Trajectory & Direction Estimation

### Trajectory Breadcrumbs
Each track maintains a circular buffer of its last $K$ centroid coordinates (`max_history_length`, default: 25-30 frames). This enables rendering motion paths and understanding object trajectories.

### Direction Estimation
Direction is estimated from the displacement vector between the immediate preceding centroid $(x_{t-1}, y_{t-1})$ and current centroid $(x_t, y_t)$:
$$\Delta x = x_t - x_{t-1}, \quad \Delta y = y_t - y_{t-1}$$

Classification:
- **Stationary**: If $|\Delta x| < 2.0\text{ px}$ and $|\Delta y| < 2.0\text{ px}$.
- **Horizontal Dominant**: `right` ($\Delta x > 0$) or `left` ($\Delta x < 0$) when $|\Delta x| \ge |\Delta y|$.
- **Vertical Dominant**: `down` ($\Delta y > 0$) or `up` ($\Delta y < 0$) when $|\Delta y| > |\Delta x|$.

> [!WARNING]
> **Camera Calibration Notice:** Pixel displacement reflects 2D projected image plane motion only. **Pixel displacement is NOT equivalent to physical-world metric speed** (e.g., km/h or m/s) without intrinsic camera calibration, extrinsic camera pose, and ground-plane homography.

---

## 4. Virtual Tripwire & Entry/Exit Counting

The `Tripwire` component defines a virtual boundary line across the field of view:
- **Orientation**: Horizontal ($y = \text{pos}$) or Vertical ($x = \text{pos}$).
- **Crossing Detection**: A crossing event triggers when a track's previous centroid and current centroid straddle the virtual line in consecutive frames.
- **Bi-directional Classification**:
  - Horizontal: Top-to-bottom crossing = `in`, Bottom-to-top crossing = `out`.
  - Vertical: Left-to-right crossing = `in`, Right-to-left crossing = `out`.
- **Debounce & Deduplication**: Track IDs are registered in `crossed_tracks` upon crossing to prevent duplicate counts from jitter or lingering on the line.

---

## 5. Actual Performance Measurements

> [!NOTE]
> Measured on host CPU with 640x480 resolution frames across 30 timed iterations (5 warmups).

| Pipeline Component | Mean Latency | P50 (Median) Latency | P95 Latency | Overhead % |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection (SSDLite320)** | 54.23 ms | 54.19 ms | 61.05 ms | 99.98% |
| **Centroid Tracker + Analytics** | **0.012 ms** | **0.011 ms** | **0.018 ms** | **0.02%** |
| **Total Frame Processing** | **54.24 ms** | **54.21 ms** | **61.07 ms** | **100.0%** |
| **Pipeline Throughput** | **18.4 FPS** | **18.4 FPS** | — | — |

**Key Takeaway**: The centroid tracking and analytics module introduces negligible computational overhead (**~0.012 ms / 12 microseconds per frame**), preserving maximum throughput for real-time edge processing.

---

## 6. Known Limitations
1. **Centroid Occlusion Ambiguity**: Simple centroid tracking without visual appearance re-identification (ReID) embeddings may swap IDs if two objects of the same class cross each other closely.
2. **2D Image Coordinates**: Movement directions and tripwires operate in pixel coordinates; perspective distortions can make objects moving toward the camera appear slower than objects moving perpendicular to the optical axis.
