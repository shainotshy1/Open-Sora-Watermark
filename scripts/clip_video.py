#!/usr/bin/env python3
"""Clip video to specified number of frames (from the beginning)."""

import argparse
import subprocess
import json

def get_frame_count(video_path):
    """Get total frame count using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            return int(stream["nb_frames"])
    return None

def main(args):
    original_frames = get_frame_count(args.input)
    print(f"Original video: {original_frames} frames")
    print(f"Clipping to: {args.frames} frames")

    subprocess.run([
        "ffmpeg", "-y", "-i", args.input,
        "-vframes", str(args.frames),
        "-c:v", "libx264", "-c:a", "copy",
        args.output
    ], check=True)

    output_frames = get_frame_count(args.output)
    print(f"Output video: {output_frames} frames → {args.output}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("-f", "--frames", type=int, required=True, help="Number of frames to keep")
    args = parser.parse_args()

    main(args)