import argparse
import asyncio
from pathlib import Path
import signal
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd

# 允许作为脚本直接运行：把项目根目录加入 sys.path（src 的上一级）
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

from src.audio_codecs.audio_codec import AudioCodec
from src.audio_processing.wake_word_detect import WakeWordDetector
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _print_devices() -> None:
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Failed to query devices: {e}", file=sys.stderr)
        return

    for i, d in enumerate(devices):
        try:
            name = d.get("name")
            in_ch = int(d.get("max_input_channels", 0))
            out_ch = int(d.get("max_output_channels", 0))
            sr = d.get("default_samplerate")
        except Exception:
            continue
        if in_ch > 0 or out_ch > 0:
            print(f"{i} {name} | in {in_ch} out {out_ch} | sr {sr}")


def _apply_device_override(device_index: int, *, input_only: bool = False) -> None:
    cfg = ConfigManager.get_instance()

    # Avoid persisting changes to disk: patch in-memory config only.
    try:
        audio_cfg = cfg._config.setdefault("AUDIO_DEVICES", {})  # noqa: SLF001
        audio_cfg["input_device_id"] = int(device_index)
        audio_cfg["input_device_name"] = None
        if not input_only:
            audio_cfg["output_device_id"] = int(device_index)
            audio_cfg["output_device_name"] = None
    except Exception as e:
        raise RuntimeError(f"Failed to apply device override: {e}") from e


class _LevelMeter:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._last = 0.0
        self._acc = []
        self.frames_total = 0

    def on_audio_data(self, audio_data: np.ndarray) -> None:
        now = time.time()
        if audio_data is None or len(audio_data) == 0:
            return
        self.frames_total += 1
        try:
            x = audio_data.astype(np.float32)
            rms = float(np.sqrt(np.mean((x / 32768.0) ** 2)))
        except Exception:
            return

        self._acc.append(rms)
        if (now - self._last) >= self.interval_s:
            self._last = now
            if self._acc:
                v = float(np.mean(self._acc))
                self._acc.clear()
                logger.info("mic_rms=%.4f", v)


async def _stats_loop(stop_event: asyncio.Event, meter: Optional[_LevelMeter]) -> None:
    last_total = 0
    while not stop_event.is_set():
        await asyncio.sleep(1.0)
        if meter is None:
            logger.info("heartbeat")
            continue
        total = int(getattr(meter, "frames_total", 0))
        delta = total - last_total
        last_total = total
        logger.info("frames_received=%d (+%d/s)", total, delta)


async def _run_direct(
    args: argparse.Namespace,
    detector: WakeWordDetector,
    stop_event: asyncio.Event,
) -> int:
    if not detector.enabled:
        logger.error(
            "Wake word is disabled by config (WAKE_WORD_OPTIONS.USE_WAKE_WORD=false)"
        )
        return 2
    if not detector.keyword_spotter:
        logger.error("KeywordSpotter is not initialized")
        return 3

    sample_rate = int(args.direct_sample_rate)
    blocksize = int(sample_rate * (args.direct_block_ms / 1000.0))
    loop = asyncio.get_running_loop()

    stream = detector.keyword_spotter.create_stream()
    last_detection_time = 0.0

    def _notify_detected(result: str) -> None:
        logger.warning("WAKE_WORD_DETECTED: %r", result)
        if args.exit_on_detect:
            stop_event.set()

    def _callback(indata, frames, time_info, status):
        nonlocal last_detection_time
        try:
            if status:
                logger.debug("input_stream_status=%s", status)
            if indata is None or len(indata) == 0:
                return

            # indata dtype int16, shape (frames, channels)
            x = indata[:, 0].astype(np.float32)
            rms = float(np.sqrt(np.mean((x / 32768.0) ** 2)))
            # 直接在回调里做轻量日志（按 1s 节流）
            now = time.time()
            if not hasattr(_callback, "_last_log"):
                _callback._last_log = 0.0
            if (now - _callback._last_log) >= 1.0:
                _callback._last_log = now
                logger.info("direct_mic_rms=%.4f", rms)

            samples = x / 32768.0
            stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
            if detector.keyword_spotter.is_ready(stream):
                detector.keyword_spotter.decode_stream(stream)
                result = detector.keyword_spotter.get_result(stream)
                if result:
                    # 防抖：避免短时间重复触发
                    if now - last_detection_time >= 1.0:
                        last_detection_time = now
                        loop.call_soon_threadsafe(_notify_detected, str(result))
                    detector.keyword_spotter.reset_stream(stream)
        except Exception as e:
            # 回调线程里不要抛异常
            try:
                loop.call_soon_threadsafe(
                    lambda: logger.error("direct_callback_error: %s", e, exc_info=True)
                )
            except Exception:
                pass

    logger.info(
        "Direct mode starting: device=%s sr=%s block_ms=%.1f blocksize=%s",
        args.device_index,
        sample_rate,
        float(args.direct_block_ms),
        blocksize,
    )

    with sd.InputStream(
        device=args.device_index,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
        callback=_callback,
    ):
        logger.info("Direct wake word test started. Say the wake word, or Ctrl+C to exit.")
        await stop_event.wait()
    return 0


async def _run(args: argparse.Namespace) -> int:
    setup_logging()

    if args.list_devices:
        _print_devices()
        return 0

    if args.device_index is not None:
        _apply_device_override(args.device_index, input_only=args.input_only)

    detector = WakeWordDetector()

    stop_event = asyncio.Event()

    def _request_stop(*_):
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, _request_stop)
            except NotImplementedError:
                pass

        if args.direct:
            return await _run_direct(args, detector, stop_event)

        audio_codec = AudioCodec()
        await audio_codec.initialize()

        meter: Optional[_LevelMeter] = None
        if args.meter:
            meter = _LevelMeter(interval_s=args.meter_interval)
            audio_codec.add_audio_listener(meter)

        stats_task = asyncio.create_task(_stats_loop(stop_event, meter))

        if not detector.enabled:
            logger.error(
                "Wake word is disabled by config (WAKE_WORD_OPTIONS.USE_WAKE_WORD=false)"
            )
            return 2

        def _on_detected(wake_word, full_text):
            logger.warning("WAKE_WORD_DETECTED: %r | %r", wake_word, full_text)
            if args.exit_on_detect:
                stop_event.set()

        detector.on_detected(_on_detected)

        ok = await detector.start(audio_codec)
        if not ok:
            logger.error("Failed to start WakeWordDetector")
            return 3

        logger.info("Wake word test started. Say the wake word, or press Ctrl+C to exit.")
        await stop_event.wait()
        return 0

    finally:
        try:
            stats_task.cancel()  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            await detector.stop()
        except Exception:
            pass
        try:
            if "audio_codec" in locals():
                await audio_codec.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="wake_word_test")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--exit-on-detect", action="store_true")
    parser.add_argument("--meter", action="store_true")
    parser.add_argument("--meter-interval", type=float, default=1.0)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass AudioCodec; use sounddevice.InputStream(16k/mono) to drive KWS directly",
    )
    parser.add_argument("--direct-sample-rate", type=int, default=16000)
    parser.add_argument("--direct-block-ms", type=float, default=60.0)
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
