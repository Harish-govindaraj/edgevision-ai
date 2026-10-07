"""Real-time tracking analytics, direction estimation, and virtual tripwire entry/exit counting."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from cv_engine.tracking.tracker import Track


def estimate_direction(
    previous_centroid: Optional[Tuple[int, int]],
    current_centroid: Tuple[int, int],
    stationary_threshold: float = 2.0,
) -> str:
    """
    Estimate coarse 2D motion direction from centroid displacement.

    Args:
        previous_centroid: (x, y) coordinate from previous frame.
        current_centroid: (x, y) coordinate from current frame.
        stationary_threshold: Minimum pixel displacement to consider moving.

    Returns:
        One of 'stationary', 'left', 'right', 'up', 'down'.
    """
    if previous_centroid is None:
        return "stationary"

    dx = current_centroid[0] - previous_centroid[0]
    dy = current_centroid[1] - previous_centroid[1]

    if abs(dx) < stationary_threshold and abs(dy) < stationary_threshold:
        return "stationary"

    # Determine dominant axis
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


@dataclass
class Tripwire:
    """
    Virtual line tripwire for bi-directional entry/exit event detection.

    Attributes:
        orientation: 'horizontal' (line at y = position) or 'vertical' (line at x = position).
        position: Pixel coordinate along orientation axis.
        in_direction: Movement direction treated as 'entry' ('down' or 'right').
        out_direction: Movement direction treated as 'exit' ('up' or 'left').
    """

    orientation: str = "horizontal"
    position: int = 240
    in_direction: str = "down"
    out_direction: str = "up"

    in_count: int = 0
    out_count: int = 0
    crossed_tracks: Set[int] = field(default_factory=set)

    def check_crossing(self, track: Track) -> Optional[str]:
        """
        Check if the track crossed the tripwire between previous and current frames.

        Returns:
            'in', 'out', or None if no crossing occurred or already counted.
        """
        if track.previous_centroid is None or track.track_id in self.crossed_tracks:
            return None

        prev_x, prev_y = track.previous_centroid
        curr_x, curr_y = track.centroid

        direction_event: Optional[str] = None

        if self.orientation == "horizontal":
            # Crossed horizontal line y = position
            if prev_y < self.position <= curr_y:
                direction_event = "in" if self.in_direction == "down" else "out"
            elif prev_y >= self.position > curr_y:
                direction_event = "out" if self.out_direction == "up" else "in"
        else:  # vertical
            # Crossed vertical line x = position
            if prev_x < self.position <= curr_x:
                direction_event = "in" if self.in_direction == "right" else "out"
            elif prev_x >= self.position > curr_x:
                direction_event = "out" if self.out_direction == "left" else "in"

        if direction_event is not None:
            if direction_event == "in":
                self.in_count += 1
            else:
                self.out_count += 1
            self.crossed_tracks.add(track.track_id)
            return direction_event

        return None

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        """Draw virtual tripwire on frame."""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        if self.orientation == "horizontal":
            cv2.line(annotated, (0, self.position), (w, self.position), color, thickness)
            cv2.putText(
                annotated,
                f"TRIPWIRE (In: {self.in_count} | Out: {self.out_count})",
                (10, max(20, self.position - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.line(annotated, (self.position, 0), (self.position, h), color, thickness)
            cv2.putText(
                annotated,
                f"TRIPWIRE (In: {self.in_count} | Out: {self.out_count})",
                (max(10, self.position + 8), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated


class TrackingAnalytics:
    """Real-time analytics engine aggregating tracks, counts, and spatial events."""

    def __init__(self, tripwire: Optional[Tripwire] = None) -> None:
        self.tripwire = tripwire or Tripwire()
        self.total_unique_seen: int = 0
        self._seen_track_ids: Set[int] = set()

    def update(self, tracks: List[Track]) -> Dict[str, Any]:
        """
        Process active tracks and return real-time analytics summary.

        Args:
            tracks: List of currently active tracks.

        Returns:
            Dictionary containing active_tracks, total_unique, class_counts,
            direction_distribution, and tripwire counts.
        """
        class_counts: Dict[str, int] = {}
        direction_counts: Dict[str, int] = {"left": 0, "right": 0, "up": 0, "down": 0, "stationary": 0}

        for track in tracks:
            # Register unique track IDs seen over time
            if track.track_id not in self._seen_track_ids:
                self._seen_track_ids.add(track.track_id)
                self.total_unique_seen += 1

            # Count by class
            class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1

            # Estimate movement direction
            direction = estimate_direction(track.previous_centroid, track.centroid)
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

            # Tripwire check
            self.tripwire.check_crossing(track)

        return {
            "active_tracks_count": len(tracks),
            "total_unique_observed": self.total_unique_seen,
            "class_distribution": class_counts,
            "direction_distribution": direction_counts,
            "tripwire_in": self.tripwire.in_count,
            "tripwire_out": self.tripwire.out_count,
        }

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: List[Track],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_trajectory: bool = True,
    ) -> np.ndarray:
        """
        Render tracked targets with persistent ID tags, confidence, and trajectory breadcrumbs.

        Format: '{class_name} {confidence}% | ID: {track_id}'
        """
        annotated = frame.copy()

        # Draw trajectory breadcrumbs first so boxes sit on top
        if draw_trajectory:
            annotated = TrackingAnalytics.draw_trajectories(annotated, tracks)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{track.class_name} {int(track.confidence * 100)}% | ID: {track.track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            bg_y1 = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(
                annotated,
                (x1, bg_y1),
                (x1 + text_w + 4, bg_y1 + text_h + baseline + 4),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 2, bg_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def draw_trajectories(
        frame: np.ndarray,
        tracks: List[Track],
        color: Tuple[int, int, int] = (255, 140, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Render motion trajectory breadcrumb paths for active tracks.

        Args:
            frame: OpenCV BGR image array.
            tracks: Active tracks with historical trajectories.
            color: Polyline color (BGR).
            thickness: Line thickness.

        Returns:
            Frame with trajectory polylines rendered.
        """
        annotated = frame.copy()
        for track in tracks:
            if len(track.trajectory) > 1:
                pts = np.array(track.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=thickness)

            # Draw center point
            cv2.circle(annotated, track.centroid, 4, (0, 0, 255), -1)
        return annotated
