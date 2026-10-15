"""Convert uncompressed microscope AVI recordings to H.264 MP4.

The recipe is fixed and was verified byte-for-byte against files produced by
hand earlier, so the output of this script is identical to:

    ffmpeg -i X.avi -c:v libx264 -preset slow -crf 15 -pix_fmt yuv444p X.mp4

Requires ffmpeg and ffprobe on PATH.

Usage:

    python avi_to_mp4.py recording.avi
    python avi_to_mp4.py *.avi -o converted/
    python avi_to_mp4.py recording.avi --force

Output goes to <source dir>_mp4/ unless -o is given. Existing files are
skipped unless --force is passed.

Two things about these recordings are worth knowing:

* The AVI header declares a duration that was written when recording started,
  and it is usually longer than the file actually is. ffmpeg aligns timestamps
  to it, so a few frames end up displayed at double length and playback runs
  slow by 5-10%. Frame-by-frame analysis is unaffected, but anything computed
  per unit time (speed, displacement between frames) is off by the same amount.
  The real frame count is file size / (width * height * 3), not the header.

* The source is bgr24, and the command below pins yuv444p, which means no
  chroma subsampling. That keeps the colour data intact but produces a High
  4:4:4 Predictive stream, which many players and phones cannot open. If a
  more compatible file matters more than colour fidelity, use yuv420p instead.

Source files are only ever read.
"""

import argparse
import json
import os
import subprocess
import sys
import time

BYTES_PER_PIXEL = 3          # source is bgr24, three bytes per pixel


def probe(path):
    """Return (width, height, pix_fmt, frame rate, declared duration)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,pix_fmt,r_frame_rate",
           "-show_entries", "format=duration", "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ffprobe cannot read {path}: {r.stderr.strip()}")
    d = json.loads(r.stdout)
    s = d["streams"][0]
    return (int(s["width"]), int(s["height"]), s["pix_fmt"],
            s["r_frame_rate"], float(d["format"]["duration"]))


def n_frames(path):
    """Frame count of an MP4, read from the container index without decoding."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path]
    return int(subprocess.run(cmd, capture_output=True, text=True).stdout)


def main():
    ap = argparse.ArgumentParser(
        description="Uncompressed AVI to H.264 MP4 "
                    "(libx264 / preset slow / crf 15 / yuv444p)")
    ap.add_argument("inputs", nargs="+", help="input .avi files (read-only)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output directory; defaults to <source dir>_mp4/")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output (default: skip)")
    args = ap.parse_args()

    for src in args.inputs:
        outdir = args.outdir or os.path.dirname(os.path.abspath(src)) + "_mp4"
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + ".mp4")
        if os.path.exists(out) and not args.force:
            print(f"[skip] already exists: {out}")
            continue

        w, h, pf, fps, dur = probe(src)
        n = os.path.getsize(src) // (w * h * BYTES_PER_PIXEL)   # not the header value
        print(f"\n=== {os.path.basename(src)} -> {out} ===")
        print(f"  source: {w}x{h} {pf} {fps}, {n} frames, "
              f"header declares {dur:.2f}s")

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
               "-c:v", "libx264",
               "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv444p",
               # progress is written line by line to stdout, ffmpeg's own
               # messages go to stderr
               "-progress", "pipe:1", "-nostats", out]
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        # Redraw the progress line only on a terminal; when piped to a log it
        # would leave one line per update. stdout still has to be drained,
        # otherwise ffmpeg blocks once the pipe buffer fills up.
        live = sys.stdout.isatty()
        for line in proc.stdout:
            if line.startswith("frame="):
                k = int(line[6:])
                if live:
                    print(f"\r  [{100.0 * k / n:5.1f}%] {k}/{n} frames   ",
                          end="", flush=True)
            elif line.startswith("progress=end"):
                break
        proc.stdout.close()
        # Either failure below leaves a truncated MP4 behind, and the next run
        # would treat it as a finished file and skip it.
        if proc.wait() != 0:
            if os.path.exists(out):
                os.unlink(out)
            sys.exit(f"ffmpeg exited with {proc.returncode}; removed {out}")
        if live:
            print(f"\r  [100.0%] {n}/{n} frames   ")

        got = n_frames(out)
        if got != n:
            os.unlink(out)
            sys.exit(f"frame count {got} != {n} expected; removed {out}")

        print(f"  {os.path.getsize(out) / 1e6:.1f} MB  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
