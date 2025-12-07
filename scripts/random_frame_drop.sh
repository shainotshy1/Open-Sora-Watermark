#!/usr/bin/env bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 input.mp4 output.mp4 p"
  echo "  p = drop probability in [0,1]"
  exit 1
fi

IN="$1"
OUT="$2"
P="$3"

if ! [[ "$P" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]]; then
  echo "Error: p must be a number in [0,1]"
  exit 1
fi

KEEP_PROB=$(awk -v p="$P" 'BEGIN {print 1 - p}')

FILTER="select='lt(random(0),$KEEP_PROB)',setpts=PTS-STARTPTS"

echo "Input:        $IN"
echo "Output:       $OUT"
echo "Drop prob:    $P"
echo "Keep prob:    $KEEP_PROB"
echo "FFmpeg filter: $FILTER"
echo

ffmpeg -y -i "$IN" \
  -vf "$FILTER" -vsync vfr \
  -c:v libx264 -crf 0 -preset veryfast \
  -pix_fmt yuv420p \
  -an \
  "$OUT"
