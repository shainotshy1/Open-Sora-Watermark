#!/usr/bin/env python3
"""Clip video to specified number of frames (from front or back)."""

import argparse
import subprocess
import json
import os

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

def clip_video(input_path, output_path, frames, from_back=False, total_frames=None):
    """Clip video to specified frames from front or back."""
    if from_back:
        start_frame = total_frames - frames
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"select='gte(n,{start_frame})',setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-c:a", "copy",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vframes", str(frames),
            "-c:v", "libx264", "-c:a", "copy",
            output_path
        ]
    subprocess.run(cmd, check=True)

def main(args):
    original_frames = get_frame_count(args.input)
    print(f"Original video: {original_frames} frames")
    
    base, ext = os.path.splitext(args.output)
    
    for frames in args.frames:
        if len(args.frames) > 1:
            output_path = f"{base}_{frames}f{ext}"
        else:
            output_path = args.output
        
        mode = "back" if args.back else "front"
        print(f"Clipping to: {frames} frames (from {mode})")
        
        clip_video(args.input, output_path, frames, args.back, original_frames)
        
        output_frames = get_frame_count(output_path)
        print(f"Output video: {output_frames} frames → {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("-f", "--frames", type=int, nargs='+', required=True, help="Number of frames to keep")
    parser.add_argument("--back", action="store_true", help="Clip from end instead of beginning")
    args = parser.parse_args()

    main(args)