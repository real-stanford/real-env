#!/usr/bin/env python3
"""
Create a combined video from rollout episode cameras.

--side bottom (default):
  [         third_person_camera_0 (main)         ]   960x720
  [  left_wrist_camera_0  |  right_wrist_camera_0 ]  480x360 each -> 960x360
  Total: 960x1080

--side right:
  [ third_person_camera_0 | left_wrist_camera_0  ]   960x720 | 480x360
  [                       | right_wrist_camera_0 ]           | 480x360
  Total: 1440x720
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_video_path(rollout_dir: Path, camera: str, stream: str) -> Path:
    path = rollout_dir / f"{camera}.zarr" / f"{stream}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Create a combined view video from a rollout episode directory."
    )
    parser.add_argument("rollout_dir", type=Path, help="Episode rollout directory")
    parser.add_argument(
        "--side",
        choices=["bottom", "right"],
        default="bottom",
        help="Where to place the wrist cameras relative to third_person (default: bottom)",
    )
    parser.add_argument(
        "--top-stream",
        choices=["main", "ultrawide"],
        default="main",
        help="Stream type for third_person_camera_0 (default: main)",
    )
    parser.add_argument(
        "--left-stream",
        choices=["main", "ultrawide"],
        default="ultrawide",
        help="Stream type for left_wrist_camera_0 (default: ultrawide)",
    )
    parser.add_argument(
        "--right-stream",
        choices=["main", "ultrawide"],
        default="ultrawide",
        help="Stream type for right_wrist_camera_0 (default: ultrawide)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <rollout_dir>/combined.mp4)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality (lower = better, default: 18)",
    )
    args = parser.parse_args()

    rollout_dir = args.rollout_dir.resolve()
    if not rollout_dir.is_dir():
        print(f"Error: not a directory: {rollout_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or rollout_dir / "combined.mp4"

    try:
        top_video = get_video_path(rollout_dir, "third_person_camera_0", args.top_stream)
        left_video = get_video_path(rollout_dir, "left_wrist_camera_0", args.left_stream)
        right_video = get_video_path(rollout_dir, "right_wrist_camera_0", args.right_stream)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Third-person: {top_video.relative_to(rollout_dir)}")
    print(f"Left wrist:   {left_video.relative_to(rollout_dir)}")
    print(f"Right wrist:  {right_video.relative_to(rollout_dir)}")
    print(f"Layout:       --side {args.side}")
    print(f"Output:       {output}")

    if args.side == "bottom":
        # third_person at 960x720 on top; wrist cameras each at 480x360 side by side below.
        # Total: 960x1080
        filter_complex = (
            "[0:v]scale=960:720[top];"
            "[1:v]scale=480:360[left];"
            "[2:v]scale=480:360[right];"
            "[left][right]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[out]"
        )
    else:
        # third_person at 960x720 on the left; left wrist stacked on right wrist (480x360 each)
        # on the right. Total: 1440x720
        filter_complex = (
            "[0:v]scale=960:720[main];"
            "[1:v]scale=480:360[left];"
            "[2:v]scale=480:360[right];"
            "[left][right]vstack=inputs=2[wrists];"
            "[main][wrists]hstack=inputs=2[out]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(top_video),
        "-i", str(left_video),
        "-i", str(right_video),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Error: ffmpeg failed", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone: {output}")


if __name__ == "__main__":
    main()
