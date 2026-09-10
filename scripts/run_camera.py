"""Run the EdgeVision real-time camera pipeline."""

import argparse

import cv2

from cv_engine.pipeline import VisionPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run EdgeVision real-time OpenCV pipeline."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video file path.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Processing width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Processing height.",
    )

    return parser.parse_args()


def get_video_source(source: str) -> int | str:
    """Convert numeric camera sources to integers."""
    try:
        return int(source)
    except ValueError:
        return source


def main() -> None:
    """Start the real-time vision pipeline."""
    args = parse_arguments()

    source = get_video_source(args.source)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video source: {args.source}"
        )

    pipeline = VisionPipeline(
        width=args.width,
        height=args.height,
    )

    print("EdgeVision AI — Real-Time CV Pipeline")
    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("Unable to read frame.")
                break

            result = pipeline.process(frame)

            display = pipeline.create_display(result)

            cv2.imshow(
                "EdgeVision AI - MVP",
                display,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()