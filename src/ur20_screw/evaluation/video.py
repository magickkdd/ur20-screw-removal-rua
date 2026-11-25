"""Encode paired robot camera frames into H.264 video."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    editor_bins = sorted(Path.home().glob(".trae-cn-server/bin/stable-*/bin/ffmpeg"))
    if editor_bins:
        return str(editor_bins[-1])
    raise RuntimeError("ffmpeg is required to encode rollout videos")


def encode_frames(frame_dir: Path, output: Path) -> int:
    frames = []
    for agent_path in sorted(frame_dir.glob("*_agentview_*.npy")):
        match = re.search(r"_(\d+)\.npy$", agent_path.name)
        if not match:
            continue
        wrist_path = agent_path.with_name(agent_path.name.replace("_agentview_", "_wristview_"))
        if not wrist_path.is_file():
            wrist_path = agent_path.with_name(agent_path.name.replace("_agentview_", "_closeup_"))
        if wrist_path.is_file():
            frames.append(np.concatenate((np.load(agent_path), np.load(wrist_path)), axis=1))
    if not frames:
        raise ValueError(f"No paired agentview/wristview frames found in {frame_dir}")
    height, width = frames[0].shape[:2]
    command = [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", "20", "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(output)]
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=stderr_file)
        try:
            assert process.stdin is not None
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            process.stdin.close()
            status = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-2000:]}")
    print(f"encoded {len(frames)} frames to {output}")
    return len(frames)


def encode_jpeg_frames(frame_dir: Path, output: Path, fps: int = 20) -> int:
    paths = sorted(frame_dir.glob("ur20_video_*.jpg"))
    if not paths:
        raise ValueError(f"No UR20 video frames found in {frame_dir}")
    first = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.uint8)
    height, width = first.shape[:2]
    command = [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(output)]
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=stderr_file)
        try:
            assert process.stdin is not None
            for path in paths:
                image = first if path == paths[0] else np.asarray(
                    Image.open(path).convert("RGB"), dtype=np.uint8
                )
                process.stdin.write(np.ascontiguousarray(image).tobytes())
            process.stdin.close()
            status = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-2000:]}")
    print(f"encoded {len(paths)} real-time UR20 frames to {output} at {fps} FPS")
    return len(paths)


def encode_jpeg_frames_with_waits(
    frame_dir: Path, output: Path, wait_events: list[dict], fps: int = 20
) -> int:
    """Encode simulation frames while preserving measured policy wait intervals.

    Each event identifies how many frames existed before an API response. The
    last available frame is held for the measured wait duration, so the output
    timeline includes the real API wall-clock delay rather than silently
    compressing it away.
    """
    paths = sorted(frame_dir.glob("ur20_video_*.jpg"))
    if not paths:
        raise ValueError(f"No UR20 video frames found in {frame_dir}")
    first = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.uint8)
    height, width = first.shape[:2]
    waits: dict[int, int] = {}
    for event in wait_events:
        frame_count = max(1, int(event.get("frame_count_before_response", 1)))
        extra = max(0, round(float(event.get("wait_s", 0.0)) * fps))
        waits[frame_count] = waits.get(frame_count, 0) + extra
    command = [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(output)]
    written = 0
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=stderr_file)
        try:
            assert process.stdin is not None
            previous = first
            for index, path in enumerate(paths, start=1):
                for _ in range(waits.get(index, 0)):
                    process.stdin.write(np.ascontiguousarray(previous).tobytes())
                    written += 1
                previous = first if index == 1 else np.asarray(
                    Image.open(path).convert("RGB"), dtype=np.uint8
                )
                process.stdin.write(np.ascontiguousarray(previous).tobytes())
                written += 1
            process.stdin.close()
            status = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-2000:]}")
    print(f"encoded {written} timeline frames to {output} at {fps} FPS")
    return written


def main() -> None:
    encode_frames(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
