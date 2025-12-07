#!/usr/bin/env bash
set -e

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 input.mp4 output.mp4 N front|back"
  echo "  N = number of frames to KEEP"
  echo "  'front' = keep first N frames"
  echo "  'back'  = keep last N frames"
  exit 1
fi

IN="$1"
OUT="$2"
N="$3"
MODE="$4"

# Get total frame count (video stream 0)
TOTAL_FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames \
  -of csv=p=0 "$IN")

if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" = "N/A" ]; then
  echo "Error: Could not read total frame count from file."
  exit 1
fi

if [ "$N" -gt "$TOTAL_FRAMES" ]; then
  echo "Error: N ($N) > total frames ($TOTAL_FRAMES). Nothing would remain."
  exit 1
fi

FILTER=""

case "$MODE" in
  front)
    # Keep first N frames → indices [0 .. N-1]
    FILTER="select='lt(n,$N)',setpts=PTS-STARTPTS"
    ;;

  back)
    # Keep last N frames → indices [START .. TOTAL_FRAMES-1]
    START=$((TOTAL_FRAMES - N))
    FILTER="select='gte(n,$START)',setpts=PTS-STARTPTS"
    ;;

  *)
    echo "Error: MODE must be 'front' or 'back'."
    exit 1
    ;;
esac

echo "Input:        $IN"
echo "Output:       $OUT"
echo "Total frames: $TOTAL_FRAMES"
echo "Keep N:     $N"
echo "Mode:         $MODE"
echo "FFmpeg filter: $FILTER"
echo

ffmpeg -y -i "$IN" \
  -vf "$FILTER" -vsync vfr \
  -c:v libx264 -crf 0 -preset veryfast \
  -pix_fmt yuv420p \
  -an \
  "$OUT"
