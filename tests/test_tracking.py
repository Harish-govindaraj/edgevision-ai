"""Unit tests for CentroidTracker, Track representation, and TrackingAnalytics."""

import numpy as np
import pytest
from cv_engine.detection.postprocessing import Detection
from cv_engine.tracking.analytics import (
    TrackingAnalytics,
    Tripwire,
    estimate_direction,
)
from cv_engine.tracking.tracker import CentroidTracker, Track, TrackState


def make_detection(x1: int, y1: int, x2: int, y2: int, class_id: int = 1, class_name: str = "person", conf: float = 0.9) -> Detection:
    """Helper to construct a test Detection."""
    return Detection(bbox=(x1, y1, x2, y2), class_id=class_id, class_name=class_name, confidence=conf)


def test_new_track_creation() -> None:
    """Tracker should create new tracks for unassociated detections."""
    tracker = CentroidTracker()
    dets = [
        make_detection(10, 10, 50, 50, class_id=1, class_name="person"),
        make_detection(100, 100, 150, 150, class_id=3, class_name="car"),
    ]

    tracks = tracker.update(dets)

    assert len(tracks) == 2
    assert tracker.total_unique_objects == 2
    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2
    assert tracks[0].centroid == (30, 30)
    assert tracks[1].centroid == (125, 125)
    assert len(tracks[0].trajectory) == 1
    assert tracks[0].age == 1


def test_matching_and_persistent_id_across_frames() -> None:
    """Tracks should persist their unique IDs across sequential frames."""
    tracker = CentroidTracker(max_distance=50.0)

    # Frame 1: Object at (30, 30)
    f1_dets = [make_detection(10, 10, 50, 50)]
    t1 = tracker.update(f1_dets)
    assert len(t1) == 1
    orig_id = t1[0].track_id

    # Frame 2: Object moves to (35, 35) (small displacement, < 50px)
    f2_dets = [make_detection(15, 15, 55, 55)]
    t2 = tracker.update(f2_dets)
    assert len(t2) == 1
    assert t2[0].track_id == orig_id
    assert t2[0].age == 2
    assert t2[0].hits == 2
    assert t2[0].previous_centroid == (30, 30)
    assert t2[0].centroid == (35, 35)
    assert t2[0].trajectory == [(30, 30), (35, 35)]


def test_multiple_objects_tracking() -> None:
    """Tracker should correctly maintain distinct identities for multiple moving objects."""
    tracker = CentroidTracker(max_distance=40.0)

    # Frame 1: Two objects
    f1 = [
        make_detection(0, 0, 20, 20),      # Center: (10, 10)
        make_detection(200, 200, 220, 220), # Center: (210, 210)
    ]
    tracks1 = tracker.update(f1)
    assert len(tracks1) == 2
    id_a = tracks1[0].track_id
    id_b = tracks1[1].track_id

    # Frame 2: Both objects move slightly
    f2 = [
        make_detection(5, 5, 25, 25),      # Center: (15, 15) -> matches A
        make_detection(205, 205, 225, 225), # Center: (215, 215) -> matches B
    ]
    tracks2 = tracker.update(f2)
    assert len(tracks2) == 2
    track_ids = {t.track_id for t in tracks2}
    assert track_ids == {id_a, id_b}


def test_object_disappearance_and_stale_removal() -> None:
    """Tracker should preserve track for max_missed_frames, then delete it."""
    tracker = CentroidTracker(max_missed_frames=2)

    # Frame 1: Object appears
    tracker.update([make_detection(10, 10, 50, 50)])
    assert len(tracker.tracks) == 1

    # Frame 2: Object disappears -> missed_frames = 1
    active_f2 = tracker.update([])
    assert len(active_f2) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[1].missed_frames == 1

    # Frame 3: Object missing again -> missed_frames = 2
    active_f3 = tracker.update([])
    assert len(active_f3) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[1].missed_frames == 2

    # Frame 4: Object missing again -> missed_frames = 3 > 2 -> deleted!
    active_f4 = tracker.update([])
    assert len(tracker.tracks) == 0


def test_trajectory_history_max_length() -> None:
    """Trajectory points should be capped at max_history_length."""
    tracker = CentroidTracker(max_history_length=5, max_distance=50.0)

    for i in range(10):
        tracker.update([make_detection(i * 5, i * 5, i * 5 + 20, i * 5 + 20)])

    assert len(tracker.tracks) == 1
    track = list(tracker.tracks.values())[0]
    assert len(track.trajectory) == 5


def test_estimate_direction() -> None:
    """Test direction classification from centroid displacement."""
    assert estimate_direction(None, (10, 10)) == "stationary"
    assert estimate_direction((10, 10), (11, 11), stationary_threshold=2.0) == "stationary"
    assert estimate_direction((10, 10), (25, 10)) == "right"
    assert estimate_direction((25, 10), (10, 10)) == "left"
    assert estimate_direction((10, 10), (10, 30)) == "down"
    assert estimate_direction((10, 30), (10, 10)) == "up"


def test_tripwire_crossing_horizontal() -> None:
    """Test tripwire entry/exit counting on horizontal line."""
    tripwire = Tripwire(orientation="horizontal", position=100, in_direction="down", out_direction="up")

    # Track 1: Moves downwards across y = 100 (from 80 to 120) -> 'in'
    track1 = Track(
        track_id=1,
        bbox=(50, 100, 90, 140),
        class_id=1,
        class_name="person",
        confidence=0.9,
        centroid=(70, 120),
        previous_centroid=(70, 80),
    )
    event1 = tripwire.check_crossing(track1)
    assert event1 == "in"
    assert tripwire.in_count == 1

    # Second check for same track -> should NOT duplicate count
    event1_repeat = tripwire.check_crossing(track1)
    assert event1_repeat is None
    assert tripwire.in_count == 1

    # Track 2: Moves upwards across y = 100 (from 110 to 90) -> 'out'
    track2 = Track(
        track_id=2,
        bbox=(50, 70, 90, 110),
        class_id=1,
        class_name="person",
        confidence=0.9,
        centroid=(70, 90),
        previous_centroid=(70, 110),
    )
    event2 = tripwire.check_crossing(track2)
    assert event2 == "out"
    assert tripwire.out_count == 1


def test_tracking_analytics_aggregation() -> None:
    """Test full TrackingAnalytics summary generation."""
    analytics = TrackingAnalytics()

    track_a = Track(track_id=1, bbox=(10, 10, 50, 50), class_id=1, class_name="person", confidence=0.9, centroid=(30, 30), previous_centroid=(25, 30))
    track_b = Track(track_id=2, bbox=(100, 100, 150, 150), class_id=3, class_name="car", confidence=0.85, centroid=(125, 135), previous_centroid=(125, 120))

    summary = analytics.update([track_a, track_b])

    assert summary["active_tracks_count"] == 2
    assert summary["total_unique_observed"] == 2
    assert summary["class_distribution"] == {"person": 1, "car": 1}
    assert summary["direction_distribution"]["right"] == 1  # track_a moved right
    assert summary["direction_distribution"]["down"] == 1   # track_b moved down
