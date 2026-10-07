"""Centroid-based multi-object tracker for real-time edge computer vision."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv_engine.detection.postprocessing import Detection


class TrackState(Enum):
    """Lifecycle state of a tracked target."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class Track:
    """
    Representation of an individual tracked object over time.

    Attributes:
        track_id: Unique persistent identifier.
        bbox: Current bounding box (x1, y1, x2, y2).
        class_id: Numerical class ID.
        class_name: Human-readable category label.
        confidence: Most recent detection confidence score.
        centroid: Current (x, y) center coordinate.
        previous_centroid: Center coordinate from immediately preceding frame.
        trajectory: Historical sequence of (x, y) center points.
        missed_frames: Consecutive frames this track was not matched.
        age: Total frames since this track was first created.
        hits: Total number of successful detection associations.
        state: Current TrackState.
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    centroid: Tuple[int, int]
    previous_centroid: Optional[Tuple[int, int]] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    missed_frames: int = 0
    age: int = 1
    hits: int = 1
    state: TrackState = TrackState.CONFIRMED

    @property
    def displacement(self) -> Tuple[int, int]:
        """Displacement vector (dx, dy) between previous and current centroid."""
        if self.previous_centroid is None:
            return (0, 0)
        return (
            self.centroid[0] - self.previous_centroid[0],
            self.centroid[1] - self.previous_centroid[1],
        )

    @property
    def velocity_magnitude(self) -> float:
        """Euclidean displacement per frame in pixels."""
        dx, dy = self.displacement
        return math.hypot(dx, dy)


class CentroidTracker:
    """
    Lightweight, deterministic Centroid Multi-Object Tracker.

    Matches detections to existing tracks using Euclidean distance between
    spatial centroids, assigns persistent IDs, maintains trajectory history,
    and handles target creation/deletion.
    """

    def __init__(
        self,
        max_distance: float = 60.0,
        max_missed_frames: int = 15,
        max_history_length: int = 30,
        match_class: bool = True,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            max_distance: Maximum Euclidean pixel distance allowed for a match.
            max_missed_frames: Frames an unmatched track is preserved before deletion.
            max_history_length: Maximum number of trajectory points retained.
            match_class: Require matching detections to have identical class ID.
        """
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.max_history_length = max_history_length
        self.match_class = match_class

        self._next_id: int = 1
        self.tracks: Dict[int, Track] = {}
        self.total_unique_objects: int = 0

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._next_id = 1
        self.tracks.clear()
        self.total_unique_objects = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of Detection objects from current frame.

        Returns:
            List of currently active Track objects.
        """
        # Case 1: No active tracks currently exist
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        # Case 2: No new detections in current frame
        if len(detections) == 0:
            tracks_to_delete = []
            for track_id, track in self.tracks.items():
                track.missed_frames += 1
                track.age += 1
                if track.missed_frames > self.max_missed_frames:
                    tracks_to_delete.append(track_id)
                else:
                    track.state = TrackState.LOST

            for tid in tracks_to_delete:
                del self.tracks[tid]

            return [t for t in self.tracks.values() if t.state != TrackState.LOST]

        # Case 3: Both existing tracks and incoming detections exist -> bipartite matching
        track_ids = list(self.tracks.keys())
        active_tracks = [self.tracks[tid] for tid in track_ids]

        # Construct cost matrix (N_tracks x M_detections)
        cost_matrix = np.full((len(active_tracks), len(detections)), 1e6, dtype=np.float32)

        for i, track in enumerate(active_tracks):
            tx, ty = track.centroid
            for j, det in enumerate(detections):
                # Class compatibility check
                if self.match_class and track.class_id != det.class_id:
                    continue

                dx = tx - det.center[0]
                dy = ty - det.center[1]
                dist = math.hypot(dx, dy)

                if dist <= self.max_distance:
                    cost_matrix[i, j] = dist

        # Optimal linear assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= self.max_distance:
                # Valid match
                track = active_tracks[r]
                det = detections[c]

                # Update track properties
                track.previous_centroid = track.centroid
                track.centroid = det.center
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.missed_frames = 0
                track.age += 1
                track.hits += 1
                track.state = TrackState.CONFIRMED

                # Update trajectory history
                track.trajectory.append(det.center)
                if len(track.trajectory) > self.max_history_length:
                    track.trajectory.pop(0)

                assigned_tracks.add(track.track_id)
                assigned_detections.add(c)

        # Unmatched existing tracks
        tracks_to_delete = []
        for tid, track in self.tracks.items():
            if tid not in assigned_tracks:
                track.missed_frames += 1
                track.age += 1
                if track.missed_frames > self.max_missed_frames:
                    tracks_to_delete.append(tid)
                else:
                    track.state = TrackState.LOST

        for tid in tracks_to_delete:
            del self.tracks[tid]

        # Unmatched detections -> spawn new tracks
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                self._create_track(det)

        # Return active tracks (confirmed or active in current frame)
        return [t for t in self.tracks.values() if t.missed_frames == 0]

    def _create_track(self, det: Detection) -> Track:
        """Instantiate and register a new Track."""
        tid = self._next_id
        self._next_id += 1
        self.total_unique_objects += 1

        new_track = Track(
            track_id=tid,
            bbox=det.bbox,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            centroid=det.center,
            previous_centroid=None,
            trajectory=[det.center],
            missed_frames=0,
            age=1,
            hits=1,
            state=TrackState.CONFIRMED,
        )
        self.tracks[tid] = new_track
        return new_track
