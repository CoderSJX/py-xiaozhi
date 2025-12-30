import argparse
import asyncio
import signal
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd

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

    def on_audio_data(self, audio_data: np.ndarray) -> None:
        now = time.time()
        if audio_data is None or len(audio_data) == 0:
            return
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


async def _run(args: argparse.Namespace) -> int:
    setup_logging()

    if args.list_devices:
        _print_devices()
        return 0

    if args.device_index is not None:
        _apply_device_override(args.device_index, input_only=args.input_only)

    audio_codec = AudioCodec()
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

        await audio_codec.initialize()

        meter: Optional[_LevelMeter] = None
        if args.meter:
            meter = _LevelMeter(interval_s=args.meter_interval)
            audio_codec.add_audio_listener(meter)

        if not detector.enabled:
            logger.error("Wake word is disabled by config (WAKE_WORD_OPTIONS.USE_WAKE_WORD=false)")
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
            await detector.stop()
        except Exception:
            pass
        try:
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
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
