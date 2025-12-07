#!/usr/bin/env bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 input.mp4 output.mp4 N"
  echo "  N = keep every Nth frame (e.g., N=2 keeps frames 0,2,4,...)"
  exit 1
fi

IN="$1"
OUT="$2"
N="$3"

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: N must be a positive integer"
  exit 1
fi

# Keep every Nth frame: frame indices where (n % N) == 0
FILTER="select='not(mod(n,$N))',setpts=PTS-STARTPTS"

echo "Input:        $IN"
echo "Output:       $OUT"
echo "Subsample:    every ${N}th frame"
echo "FFmpeg filter: $FILTER"
echo

ffmpeg -y -i "$IN" \
  -vf "$FILTER" -vsync vfr \
  -c:v libx264 -crf 0 -preset veryfast \
  -pix_fmt yuv420p \
  -an \
  "$OUT"

