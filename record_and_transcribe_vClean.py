import argparse
import json
import math
import os
import re
import site
import sys
import tempfile
import time
import urllib.error
from collections import Counter, defaultdict, deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_NVIDIA_DLL_HANDLES: list[object] = []


def add_nvidia_dll_paths() -> None:
    discovered_bin_dirs: list[str] = []

    if not hasattr(os, "add_dll_directory"):
        for site_dir in site.getsitepackages():
            nvidia_dir = Path(site_dir) / "nvidia"
            if not nvidia_dir.exists():
                continue
            for dll_dir in list(nvidia_dir.glob("*/*/bin")) + list(nvidia_dir.glob("*/bin")):
                if dll_dir.exists():
                    discovered_bin_dirs.append(str(dll_dir))
        if discovered_bin_dirs:
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(discovered_bin_dirs + [current_path])
        return

    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.exists():
            continue

        for dll_dir in nvidia_dir.glob("*/*/bin"):
            if dll_dir.exists():
                discovered_bin_dirs.append(str(dll_dir))
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))

        for dll_dir in nvidia_dir.glob("*/bin"):
            if dll_dir.exists():
                discovered_bin_dirs.append(str(dll_dir))
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))

    if discovered_bin_dirs:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(discovered_bin_dirs + [current_path])


add_nvidia_dll_paths()

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from faster_whisper import WhisperModel


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return cleaned[:64] or "device"


def resolve_sidecar_path(base_dir: Path, target: Path) -> Path:
    return target if target.is_absolute() else base_dir / target.name


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.size / float(source_rate)
    target_size = max(1, int(duration * target_rate))
    x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def clamp_text_size(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[... gekürzt wegen Kontextlänge ...]\n\n" + tail


# -----------------------------------------------------------------------------
# Audio device handling
# -----------------------------------------------------------------------------


def list_devices() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Available audio devices:\n")
    for idx, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        print(
            f"[{idx}] {dev['name']} | hostapi={hostapi_name} | "
            f"in={dev['max_input_channels']} out={dev['max_output_channels']}"
        )


def resolve_recording_config(
    device_name: str | None,
    mic: bool,
    device_index: int | None,
) -> tuple[int, int, int, sd.WasapiSettings | None, str]:
    devices = sd.query_devices()

    if mic:
        if device_index is not None:
            if device_index < 0 or device_index >= len(devices):
                raise RuntimeError(f"Device index {device_index} is out of range.")
            dev = devices[device_index]
            if int(dev["max_input_channels"]) < 1:
                raise RuntimeError(f"Device index {device_index} has no input channels: '{dev['name']}'.")
            device_idx = device_index
        else:
            default_input = sd.default.device[0]
            if default_input is None or default_input < 0:
                raise RuntimeError("No default microphone input device found.")
            device_idx = int(default_input)
            dev = devices[device_idx]

        channels = max(1, min(2, int(dev["max_input_channels"])))
        sample_rate = int(dev["default_samplerate"])
        extra_settings = None
        source_name = f"microphone: {dev['name']}"
        return device_idx, channels, sample_rate, extra_settings, source_name

    if hasattr(sd, "WasapiSettings"):
        try:
            params = sd.WasapiSettings.__init__.__code__.co_varnames
            if "loopback" in params:
                loopback_settings = sd.WasapiSettings(loopback=True)
            else:
                loopback_settings = sd.WasapiSettings()
        except Exception:
            loopback_settings = None
    else:
        loopback_settings = None

    if loopback_settings is not None:
        if device_index is not None:
            if device_index < 0 or device_index >= len(devices):
                raise RuntimeError(f"Device index {device_index} is out of range.")
            dev = devices[device_index]
            if int(dev["max_output_channels"]) < 1:
                raise RuntimeError(f"Device index {device_index} is not an output device: '{dev['name']}'.")
            device_idx = device_index
        elif device_name:
            wanted = device_name.lower()
            device_idx = None
            for idx, dev in enumerate(devices):
                if int(dev["max_output_channels"]) < 1:
                    continue
                if wanted in dev["name"].lower():
                    device_idx = idx
                    break
            if device_idx is None:
                raise RuntimeError(f"Device '{device_name}' not found.")
            dev = devices[device_idx]
        else:
            default_output = sd.default.device[1]
            if default_output is None or default_output < 0:
                raise RuntimeError("No default output device found for loopback.")
            device_idx = int(default_output)
            dev = devices[device_idx]

        channels = max(1, min(2, int(dev["max_output_channels"])))
        sample_rate = int(dev["default_samplerate"])
        source_name = f"system audio loopback (WASAPI): {dev['name']}"
        return device_idx, channels, sample_rate, loopback_settings, source_name

    raise RuntimeError("No WASAPI loopback support found. Install a newer sounddevice build or use --mic.")


def capture_output_loopback_stream() -> Any:
    try:
        import pyaudiowpatch as pyaudio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Output-loopback backend not installed. Run: pip install pyaudiowpatch") from exc
    return pyaudio


def resolve_pyaudio_loopback_device(p: Any, pyaudio: Any, device_name: str | None, device_index: int | None) -> dict:
    if device_index is not None:
        try:
            candidate = p.get_device_info_by_index(device_index)
            if candidate.get("isLoopbackDevice", False):
                return candidate
            if int(candidate.get("maxOutputChannels", 0)) >= 1:
                output_name = str(candidate.get("name", ""))
                for loopback in p.get_loopback_device_info_generator():
                    if output_name in str(loopback.get("name", "")):
                        return loopback
        except Exception:
            pass

    if device_name:
        wanted = device_name.lower()
        for idx in range(p.get_device_count()):
            info = p.get_device_info_by_index(idx)
            if int(info.get("maxOutputChannels", 0)) < 1:
                continue
            if wanted in str(info.get("name", "")).lower():
                if info.get("isLoopbackDevice", False):
                    return info
                output_name = str(info.get("name", ""))
                for loopback in p.get_loopback_device_info_generator():
                    if output_name in str(loopback.get("name", "")):
                        return loopback

    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_output = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    output_name = str(default_output.get("name", ""))
    for loopback in p.get_loopback_device_info_generator():
        if output_name in str(loopback.get("name", "")):
            return loopback
    return default_output


# -----------------------------------------------------------------------------
# Whisper transcription
# -----------------------------------------------------------------------------


def enforce_large_model(model_size: str) -> str:
    enforced = "large-v3"
    if model_size != enforced:
        print(f"Requested model '{model_size}' overridden to '{enforced}' for transcription quality.", file=sys.stderr)
    return enforced


def create_whisper_model(model_size: str) -> WhisperModel:
    try:
        print(f"Loading Whisper model: {model_size} (CUDA)")
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception as exc:  # noqa: BLE001
        print("CUDA Whisper unavailable, using CPU int8. " f"Reason: {exc}", file=sys.stderr)
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_with_model(model: WhisperModel, audio_path: Path) -> str:
    segments, _info = model.transcribe(str(audio_path), language="de", vad_filter=True)
    raw = " ".join(segment.text.strip() for segment in segments if segment.text)
    return normalize_spaces(raw)


def transcribe_with_runtime_fallback(model: WhisperModel, model_size: str, audio_path: Path) -> tuple[str, WhisperModel]:
    try:
        return transcribe_with_model(model, audio_path), model
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc).lower()
        is_cuda_error = any(token in error_text for token in ("cublas", "cuda", "cudnn"))
        if not is_cuda_error:
            raise
        print("CUDA inference failed, retrying on CPU (int8). " f"Reason: {exc}", file=sys.stderr)
        cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return transcribe_with_model(cpu_model, audio_path), cpu_model


def transcribe_segment_audio(model: WhisperModel, model_size: str, segment_audio: np.ndarray, sample_rate: int) -> tuple[str, WhisperModel]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        sf.write(str(temp_path), np.clip(segment_audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
        text, model = transcribe_with_runtime_fallback(model, model_size, temp_path)
        return text.strip(), model
    finally:
        if temp_path.exists():
            temp_path.unlink()


# -----------------------------------------------------------------------------
# Speaker embedding / clustering
# -----------------------------------------------------------------------------


def load_speechbrain_backend(prefer_gpu: bool = True) -> tuple[Any | None, Any | None, str]:
    try:
        import torch as local_torch
        from speechbrain.inference.speaker import EncoderClassifier

        use_cuda = bool(prefer_gpu and local_torch.cuda.is_available())
        device = "cuda:0" if use_cuda else "cpu"
        print(f"Speaker backend: speechbrain ({device})")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        return local_torch, classifier, device
    except Exception as exc:  # noqa: BLE001
        print("SpeechBrain backend unavailable, falling back to local spectral embedding. " f"Reason: {exc}", file=sys.stderr)
        return None, None, "cpu"


def estimate_pitch_autocorr(frame: np.ndarray, sample_rate: int) -> float:
    min_f0 = 75.0
    max_f0 = 340.0
    min_lag = int(sample_rate / max_f0)
    max_lag = int(sample_rate / min_f0)
    if max_lag <= min_lag or frame.size <= max_lag + 2:
        return float("nan")
    centered = frame - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy < 1e-8:
        return float("nan")
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    window = corr[min_lag:max_lag]
    if window.size == 0:
        return float("nan")
    peak_idx = int(np.argmax(window)) + min_lag
    peak_val = float(corr[peak_idx])
    if peak_val <= 0:
        return float("nan")
    normalized_peak = peak_val / max(float(corr[0]), 1e-8)
    if normalized_peak < 0.15:
        return float("nan")
    return float(sample_rate / peak_idx)


def spectral_fallback_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = max(int(0.04 * sample_rate), 512)
    hop = max(int(0.02 * sample_rate), 256)
    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))

    window = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    rows: list[np.ndarray] = []

    for start in range(0, max(1, audio.size - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if frame.size < frame_size:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if rms < 1e-3:
            continue
        win_frame = frame * window
        spec = np.abs(np.fft.rfft(win_frame))
        spec_sum = float(np.sum(spec))
        if spec_sum < 1e-8:
            continue
        centroid = float(np.sum(freqs * spec) / spec_sum)
        rolloff_idx = int(np.searchsorted(np.cumsum(spec), 0.85 * np.sum(spec)))
        rolloff_idx = min(rolloff_idx, freqs.size - 1)
        rolloff = float(freqs[rolloff_idx])
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame))).astype(np.float32)))
        pitch = estimate_pitch_autocorr(frame, sample_rate)
        pitch_value = 0.0 if math.isnan(pitch) else pitch
        rows.append(np.array([pitch_value, centroid, rolloff, zcr, rms], dtype=np.float32))

    if not rows:
        return np.zeros(10, dtype=np.float32)
    mat = np.asarray(rows, dtype=np.float32)
    return np.concatenate([np.mean(mat, axis=0), np.std(mat, axis=0)]).astype(np.float32)


class SpeakerBackbone:
    def __init__(self, prefer_gpu: bool = True, backend: str = "speechbrain") -> None:
        self.backend_name = backend
        self.torch = None
        self.classifier = None
        self.device = "cpu"
        if backend == "speechbrain":
            self.torch, self.classifier, self.device = load_speechbrain_backend(prefer_gpu=prefer_gpu)
        else:
            print("Speaker backend: fallback spectral mode")

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        mono = np.asarray(mono, dtype=np.float32)
        if self.classifier is not None and self.torch is not None:
            mono_16k = resample_linear(mono, sample_rate, 16000)
            wav = self.torch.from_numpy(mono_16k).float().unsqueeze(0)
            if str(self.device).startswith("cuda"):
                wav = wav.to(self.device)
            with self.torch.no_grad():
                emb = self.classifier.encode_batch(wav)
            return emb.squeeze().detach().cpu().numpy().astype(np.float32)
        return spectral_fallback_embedding(mono, sample_rate)


class OnlineSpeakerClustering:
    def __init__(self, max_speakers: int, distance_threshold: float, known_profiles: dict[str, np.ndarray] | None = None) -> None:
        self.max_speakers = max(1, max_speakers)
        self.distance_threshold = float(distance_threshold)
        self.known_profiles = known_profiles or {}
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}

    def _nearest(self, embedding: np.ndarray, candidates: dict[str, np.ndarray]) -> tuple[str, float] | None:
        if not candidates:
            return None
        best_name = ""
        best_dist = float("inf")
        for name, vec in candidates.items():
            if vec.size != embedding.size:
                continue
            dist = cosine_distance(embedding, vec)
            if dist < best_dist:
                best_name = name
                best_dist = dist
        if not best_name:
            return None
        return best_name, best_dist

    def assign(self, embedding: np.ndarray) -> tuple[str, float]:
        known_hit = self._nearest(embedding, self.known_profiles)
        if known_hit is not None:
            name, dist = known_hit
            if dist <= self.distance_threshold:
                return name, max(0.0, min(1.0, 1.0 - dist))

        nearest = self._nearest(embedding, self.centroids)
        if nearest is None:
            speaker = "Sprecher_1"
            self.centroids[speaker] = embedding.copy()
            self.counts[speaker] = 1
            return speaker, 1.0

        best_speaker, best_dist = nearest
        if best_dist > self.distance_threshold and len(self.centroids) < self.max_speakers:
            speaker = f"Sprecher_{len(self.centroids) + 1}"
            self.centroids[speaker] = embedding.copy()
            self.counts[speaker] = 1
            return speaker, 0.95

        n = self.counts.get(best_speaker, 1)
        self.centroids[best_speaker] = (self.centroids[best_speaker] * n + embedding) / (n + 1)
        self.counts[best_speaker] = n + 1
        return best_speaker, max(0.0, min(1.0, 1.0 - best_dist))

    def export_profiles(self) -> dict[str, np.ndarray]:
        merged = {name: vec.copy() for name, vec in self.known_profiles.items()}
        for name, vec in self.centroids.items():
            if name in merged and merged[name].size == vec.size:
                merged[name] = 0.75 * merged[name] + 0.25 * vec
            else:
                merged[name] = vec.copy()
        return merged


def load_profiles(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    if not path.exists():
        raise RuntimeError(f"Speaker profile file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    if isinstance(raw, dict):
        for name, obj in raw.items():
            vec = obj.get("embedding", obj.get("signature", [])) if isinstance(obj, dict) else obj
            arr = np.asarray(vec, dtype=np.float32)
            if arr.size:
                out[str(name)] = arr
    return out


def save_profiles(path: Path, profiles: dict[str, np.ndarray]) -> None:
    payload = {name: {"embedding": np.asarray(vec, dtype=np.float32).tolist()} for name, vec in profiles.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_raw_session_payload(raw_json_path: Path) -> tuple[dict, list[dict]]:
    if not raw_json_path.exists():
        raise RuntimeError(f"Raw session file not found: {raw_json_path}")
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Raw session file is not a JSON object: {raw_json_path}")
    metadata = payload.get("metadata", {})
    entries = payload.get("entries", [])
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Raw session metadata is invalid in: {raw_json_path}")
    if not isinstance(entries, list):
        raise RuntimeError(f"Raw session entries are invalid in: {raw_json_path}")
    return metadata, entries


# -----------------------------------------------------------------------------
# Segmentation and processing
# -----------------------------------------------------------------------------


class EnergySegmenter:
    def __init__(
        self,
        sample_rate: int,
        min_speech_seconds: float = 0.55,
        min_silence_seconds: float = 0.35,
        max_segment_seconds: float = 18.0,
        preroll_seconds: float = 0.25,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_speech_seconds = min_speech_seconds
        self.min_silence_seconds = min_silence_seconds
        self.max_segment_seconds = max_segment_seconds
        self.noise_floor_db = -58.0
        self.in_speech = False
        self.silence_seconds = 0.0
        self.segment_start_seconds = 0.0
        self.current_frames: list[np.ndarray] = []
        self.preroll: deque[tuple[float, np.ndarray]] = deque()
        self.preroll_limit = max(1, int((preroll_seconds * sample_rate) / 2048) + 2)

    @staticmethod
    def _dbfs(frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        return 20.0 * math.log10(max(rms, 1e-9))

    def _append_preroll(self, start_seconds: float, frame: np.ndarray) -> None:
        self.preroll.append((start_seconds, frame.copy()))
        while len(self.preroll) > self.preroll_limit:
            self.preroll.popleft()

    def _start_threshold(self) -> float:
        return self.noise_floor_db + 11.0

    def _stop_threshold(self) -> float:
        return self.noise_floor_db + 7.0

    def _finalize(self, end_seconds: float) -> dict | None:
        if not self.current_frames:
            self.in_speech = False
            self.silence_seconds = 0.0
            return None
        audio = np.concatenate(self.current_frames, axis=0)
        if audio.shape[0] / float(self.sample_rate) < self.min_speech_seconds:
            self.current_frames = []
            self.in_speech = False
            self.silence_seconds = 0.0
            return None
        segment = {"start_seconds": self.segment_start_seconds, "end_seconds": end_seconds, "audio": audio}
        self.current_frames = []
        self.in_speech = False
        self.silence_seconds = 0.0
        return segment

    def feed(self, frame: np.ndarray, frame_start_seconds: float) -> list[dict]:
        outputs: list[dict] = []
        frame_duration = frame.shape[0] / float(self.sample_rate)
        frame_db = self._dbfs(frame)

        if not self.in_speech:
            self.noise_floor_db = 0.98 * self.noise_floor_db + 0.02 * frame_db
        self._append_preroll(frame_start_seconds, frame)

        if not self.in_speech and frame_db >= self._start_threshold():
            self.in_speech = True
            self.silence_seconds = 0.0
            self.segment_start_seconds = self.preroll[0][0] if self.preroll else frame_start_seconds
            self.current_frames = [chunk.copy() for _, chunk in self.preroll] if self.preroll else [frame.copy()]
            return outputs

        if self.in_speech:
            self.current_frames.append(frame.copy())
            if frame_db < self._stop_threshold():
                self.silence_seconds += frame_duration
            else:
                self.silence_seconds = 0.0
            segment_duration = frame_start_seconds + frame_duration - self.segment_start_seconds
            if self.silence_seconds >= self.min_silence_seconds or segment_duration >= self.max_segment_seconds:
                segment = self._finalize(frame_start_seconds + frame_duration - self.silence_seconds)
                if segment is not None:
                    outputs.append(segment)

        return outputs

    def flush(self, end_seconds: float) -> dict | None:
        if self.in_speech and self.current_frames:
            return self._finalize(end_seconds)
        return None


def append_jsonl(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class SegmentProcessor:
    def __init__(
        self,
        model: WhisperModel,
        model_size: str,
        sample_rate: int,
        embedder: SpeakerBackbone,
        clustering: OnlineSpeakerClustering,
        txt_path: Path,
        log_path: Path,
        speaker_log_path: Path,
        live_window: Any,
    ) -> None:
        self.model = model
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.embedder = embedder
        self.clustering = clustering
        self.txt_path = txt_path
        self.log_path = log_path
        self.speaker_log_path = speaker_log_path
        self.live_window = live_window
        self.entries: list[dict] = []
        self._queue: Queue[dict | None] = Queue()
        self._thread: Thread | None = None
        self._segment_index = 0

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, segment: dict) -> None:
        self._queue.put(segment)

    def close(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while True:
            segment = self._queue.get()
            if segment is None:
                break
            self._process(segment)

    def _process(self, segment: dict) -> None:
        audio = segment["audio"]
        text, self.model = transcribe_segment_audio(self.model, self.model_size, audio, self.sample_rate)
        if not text:
            return

        embedding = self.embedder.embed(audio, self.sample_rate)
        speaker, confidence = self.clustering.assign(embedding)
        self._segment_index += 1
        row = {
            "segment": self._segment_index,
            "start_seconds": float(segment["start_seconds"]),
            "end_seconds": float(segment["end_seconds"]),
            "duration_seconds": round(float(segment["end_seconds"] - segment["start_seconds"]), 2),
            "speaker": speaker,
            "speaker_confidence": round(float(confidence), 3),
            "text": text,
        }
        self.entries.append(row)
        line = f"[{format_mmss(row['start_seconds'])}-{format_mmss(row['end_seconds'])}] {speaker}: {text}"
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        append_jsonl(self.log_path, row)
        append_jsonl(self.speaker_log_path, row)
        print(line)
        if self.live_window:
            self.live_window.add_chunk(self._segment_index, f"{speaker}: {text}")


# -----------------------------------------------------------------------------
# Heuristic reports
# -----------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def build_topic_signature(text: str) -> list[str]:
    stopwords = {
        "der", "die", "das", "und", "oder", "aber", "dass", "ist", "sind", "war", "waren",
        "ein", "eine", "einer", "einem", "einen", "mit", "von", "für", "auf", "im", "in",
        "zu", "zum", "zur", "den", "dem", "des", "nicht", "wir", "ihr", "sie", "ich", "du",
        "ja", "nein", "auch", "noch", "schon", "nur", "weil", "wenn", "wie", "halt", "dann",
    }
    words = re.findall(r"[a-zA-ZäöüÄÖÜß0-9]{4,}", text.lower())
    filtered = [w for w in words if w not in stopwords]
    counts = Counter(filtered)
    return [word for word, _count in counts.most_common(12)]


def detect_action_items(sentences: list[str]) -> list[str]:
    action_keywords = (
        "todo", "to do", "action", "next step", "next steps", "must", "need to", "should",
        "deadline", "until", "follow up", "owner", "abklären", "klären", "prüfen", "checken",
        "entscheiden", "fertig", "liefern", "abschließen", "finalisieren", "müssen", "soll",
        "sollen", "muss", "nachfragen", "vorbereiten", "schicken", "senden",
    )
    items = [s for s in sentences if any(k in s.lower() for k in action_keywords)]
    return items[:14]


def detect_decisions(sentences: list[str]) -> list[str]:
    decision_keywords = (
        "wir machen", "wir nehmen", "wir nutzen", "entscheiden", "beschließen", "ist fix", "wird", "bleibt",
        "einigen uns", "vereinbaren", "dann machen wir", "das passt", "wir gehen", "wir setzen", "haben wir entschieden",
    )
    items = [s for s in sentences if any(k in s.lower() for k in decision_keywords)]
    return items[:10]


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "No summary available."
    return " ".join(sentences[:max_sentences])

def smooth_segment_text(text: str) -> str:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return cleaned
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned

def build_near_raw_flow(entries: list[dict], max_chars_per_block: int = 900) -> list[str]:
    """Build a near-verbatim flow text from all segments with only light smoothing."""
    blocks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for row in entries:
        start = format_mmss(float(row["start_seconds"]))
        end = format_mmss(float(row["end_seconds"]))
        speaker = row.get("speaker", "Unknown")
        text = smooth_segment_text(str(row.get("text", "")))
        if not text:
            continue

        line = f"[{start}-{end}] {speaker}: {text}"
        line_len = len(line) + 1
        if current_lines and current_len + line_len > max_chars_per_block:
            blocks.append(" ".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        blocks.append(" ".join(current_lines))

    return blocks


def build_dialog_report(entries: list[dict], title: str) -> str:
    lines = [f"# {title}", "", "## Dialog / Multilog"]
    if not entries:
        lines.append("- Keine Segmente erkannt.")
        return "\n".join(lines) + "\n"

    for row in entries:
        start = format_mmss(float(row["start_seconds"]))
        end = format_mmss(float(row["end_seconds"]))
        speaker = row.get("speaker", "Unknown")
        conf = float(row.get("speaker_confidence", 0.0))
        lines.append(f"- [{start}-{end}] {speaker} ({conf:.2f}): {row['text']}")

    lines.extend(["", "## Sprecherweise Ansicht"])
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in entries:
        by_speaker[str(row.get("speaker", "Unknown"))].append(row)
    for speaker in sorted(by_speaker.keys()):
        lines.append(f"### {speaker}")
        for row in by_speaker[speaker]:
            start = format_mmss(float(row["start_seconds"]))
            end = format_mmss(float(row["end_seconds"]))
            lines.append(f"- [{start}-{end}] {row['text']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_full_report(entries: list[dict], metadata: dict) -> str:
    all_text = normalize_spaces(" ".join(row["text"] for row in entries))
    flow_blocks = build_near_raw_flow(entries)
    lines = [
        f"# {metadata['title']} - vollständiges Inhaltsprotokoll",
        "",
        "## Hinweis",
        "Dies ist eine segmentnahe Inhaltsrekonstruktion. Sie bleibt absichtlich nahe am Rohtranskript und glättet nur Sprache/Logik.",
        "Sprecherlabels können fehlerhaft sein.",
        "",
        "## Segmentnahe Rekonstruktion (vollständig, chronologisch)",
    ]
    if not flow_blocks:
        lines.append("- Keine Segmente erkannt.")
    else:
        for idx, block in enumerate(flow_blocks, start=1):
            lines.append(f"### Verlauf {idx}")
            lines.append(block)
            lines.append("")

    lines.extend([
        "",
        "## Vollständige Timeline (Rohstruktur)",
    ])
    if not entries:
        lines.append("- Keine Segmente erkannt.")
    for row in entries:
        start = format_mmss(float(row["start_seconds"]))
        end = format_mmss(float(row["end_seconds"]))
        lines.append(f"- [{start}-{end}] {row.get('speaker', 'Unknown')}: {row['text']}")

    lines.extend(["", "## Gesamter zusammenhängender Inhalt", all_text or "Kein Transkripttext vorhanden."])
    return "\n".join(lines).strip() + "\n"


def build_long_report(entries: list[dict], metadata: dict) -> str:
    all_text = normalize_spaces(" ".join(row["text"] for row in entries))
    sentences = split_sentences(all_text)
    summary = summarize_text(all_text, max_sentences=6)
    topics = build_topic_signature(all_text)
    action_items = detect_action_items(sentences)
    decisions = detect_decisions(sentences)

    lines = [
        f"# {metadata['title']} - Langprotokoll",
        "",
        "## Zusammenfassung",
        summary or "Keine Zusammenfassung verfügbar.",
        "",
        "## Kernthemen",
    ]
    lines.extend(f"- {topic}" for topic in topics) if topics else lines.append("- Keine klaren Themen erkannt.")
    lines.extend(["", "## Entscheidungen"])
    lines.extend(f"- {item}" for item in decisions) if decisions else lines.append("- Keine Meeting-Entscheidungen erkannt.")
    lines.extend(["", "## Offene Punkte / To-dos"])
    lines.extend(f"- {item}" for item in action_items) if action_items else lines.append("- Keine Meeting-To-dos erkannt.")
    lines.extend(["", "## Nachvollziehbare Timeline"])
    for row in entries:
        start = format_mmss(float(row["start_seconds"]))
        end = format_mmss(float(row["end_seconds"]))
        lines.append(f"- [{start}-{end}] {row.get('speaker', 'Unknown')}: {row['text']}")
    return "\n".join(lines).strip() + "\n"


def build_short_protocol(entries: list[dict], metadata: dict) -> str:
    all_text = normalize_spaces(" ".join(row["text"] for row in entries))
    sentences = split_sentences(all_text)
    decisions = detect_decisions(sentences)
    action_items = detect_action_items(sentences)
    topics = build_topic_signature(all_text)[:5]

    lines = [f"# {metadata['title']} - Kurzprotokoll", ""]
    if topics:
        lines.append("## Wichtigste Punkte")
        lines.extend(f"- {topic}" for topic in topics)
    else:
        lines.append("- Keine klaren Kernaussagen erkannt.")
    lines.append("")
    lines.append("## Entscheidungen / To-dos")
    if decisions:
        lines.extend(f"- Entscheidung: {item}" for item in decisions[:4])
    else:
        lines.append("- Keine Meeting-Entscheidungen erkannt.")
    if action_items:
        lines.extend(f"- To-do: {item}" for item in action_items[:4])
    else:
        lines.append("- Keine Meeting-To-dos erkannt.")
    return "\n".join(lines).strip() + "\n"


# -----------------------------------------------------------------------------
# LLM finalization: OpenAI-compatible, works with Ollama /v1
# -----------------------------------------------------------------------------


def call_openai_compatible_chat(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    timeout_seconds: int,
    use_json_response_format: bool,
) -> str:
    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if use_json_response_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
            verify=False,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM request failed (network): {exc}") from exc
        
    if response.status_code != 200:
        snippet = response.text[:1000].replace("\n", " ").strip()
        raise RuntimeError(f"LLM request failed: {response.status_code} {snippet}")

    body = response.text
    if not body.strip():
        raise RuntimeError("LLM response body was empty.")

    try:
        decoded = response.json()
    except ValueError as exc:
        snippet = body[:1000].replace("\n", " ").strip()
        raise RuntimeError(f"LLM response was not valid JSON: {snippet}") from exc

    choices = decoded.get("choices", [])
    if not choices:
        raise RuntimeError("LLM response contains no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        content = "".join(chunks)
    if not content:
        raise RuntimeError("LLM response contains no content.")
    return str(content)


def parse_llm_json_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    # Try to decode the first valid JSON object/array found in the text using
    # JSONDecoder.raw_decode which returns the parsed object and the index
    # where parsing stopped. This allows ignoring trailing non-JSON content
    # without failing on "Extra data" errors.
    decoder = json.JSONDecoder()
    # Skip any leading characters until we hit a JSON opening bracket
    idx = 0
    while idx < len(stripped) and stripped[idx] not in "{[":
        idx += 1
    if idx:
        stripped = stripped[idx:]
    try:
        obj, end = decoder.raw_decode(stripped)
        return obj
    except json.JSONDecodeError:
        # Fallback to previous heuristic: extract the first {...} or [...] slice
        if not stripped.startswith(("{", "[")):
            first_obj = stripped.find("{")
            last_obj = stripped.rfind("}")
            first_arr = stripped.find("[")
            last_arr = stripped.rfind("]")
            candidates: list[tuple[int, int]] = []
            if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
                candidates.append((first_obj, last_obj + 1))
            if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
                candidates.append((first_arr, last_arr + 1))
            if candidates:
                start, end = min(candidates, key=lambda item: item[0])
                stripped = stripped[start:end]
    return json.loads(stripped)


def call_llm_json_with_retry(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    timeout_seconds: int,
    required_keys: tuple[str, ...] | None,
) -> dict:
    last_exc: Exception | None = None
    is_hosted = "diz.uk-erlangen" in api_base
    
    for use_json_response_format in (True, False) if not is_hosted else (False,):
        try:
            content = call_openai_compatible_chat(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout_seconds=timeout_seconds,
                use_json_response_format=use_json_response_format,
            )
            decoded = parse_llm_json_response(content)
            if required_keys is not None and not all(key in decoded for key in required_keys):
                raise RuntimeError(f"LLM JSON is missing required keys: {required_keys}")
            return decoded
        except (RuntimeError, urllib.error.URLError, requests.RequestException, json.JSONDecodeError) as exc:
            last_exc = exc
            if use_json_response_format:
                print(
                    "LLM json_object response_format failed, retrying once without response_format. "
                    f"Reason: {exc}",
                    file=sys.stderr,
                )
                continue
            break
    raise RuntimeError(f"LLM call failed: {last_exc}")


def build_report_payload(metadata: dict, entries: list[dict], dialog_text: str) -> dict:
    return {"metadata": metadata, "entries": entries, "dialog_seed": dialog_text}


def build_final_report_messages_from_payload(payload_text: str, custom_system_prompt: str | None) -> list[dict]:
    system_prompt = custom_system_prompt or (
        "Du bist ein deutscher Protokoll-Assistent für Meeting-Transkripte. "
        "Deine Hauptaufgabe ist die vollständige und korrekte Erfassung des Meeting-Inhalts, nicht die Sprecheranalyse. "
        "Sprecherlabels stammen aus automatischer Erkennung und können falsch sein. Nutze sie nur vorsichtig als Orientierung. "
        "Priorität haben Inhalt, Kontext, Themen, Entscheidungen, offene Punkte, To-dos, Verantwortlichkeiten und nächste Schritte. "
        "Du musst ausschließlich auf Deutsch antworten. "
        "Du musst ausschließlich valides JSON ausgeben. Keine Markdown-Codefences. Kein Text vor oder nach dem JSON. "
        "Das JSON muss exakt diese Keys enthalten: full_report, long_report, short_protocol. Alle Werte sind Strings im Markdown-Format. "
        "full_report = segmentnahes, nahezu vollständiges Inhaltsprotokoll. Pflicht: chronologische Rekonstruktion mit sehr hoher Abdeckung "
        "(nahe am Rohtranskript), inklusive Details, Argumentationslinien, offenen Punkten, Entscheidungen, To-dos und Verantwortlichkeiten. "
        "Du darfst sprachlich glätten, aber nicht stark verdichten. Keine inhaltlichen Luecken durch aggressive Zusammenfassung. "
        "long_report = verdichtetes Langprotokoll nach Sinnabschnitten und Themen. Es soll das Meeting gut nachvollziehbar halten. "
        "short_protocol = ultra-kurzes Protokoll mit maximal 8 Bulletpoints, nur die wichtigsten Punkte, keine Metadaten und keine Sprecheranalyse. "
        "Erfinde keine Fakten, Namen, Rollen, Daten oder Orte. Keine Platzhalter. "
        "Wenn Information fehlt, schreibe 'nicht genannt' oder lasse den Punkt weg. Wenn etwas unklar ist, schreibe 'unklar'. "
        "Wenn keine belastbaren Entscheidungen enthalten sind, schreibe: 'Keine Meeting-Entscheidungen erkannt'. "
        "Wenn keine belastbaren To-dos enthalten sind, schreibe: 'Keine Meeting-To-dos erkannt'."
    )

    user_prompt = (
        "Erstelle aus den folgenden Rohdaten ausschließlich deutsche Protokollausgaben.\n"
        "Antworte ausschließlich als valides JSON mit exakt diesen Schlüsseln:\n"
        "- full_report: string, Markdown, vollständiges Inhaltsprotokoll\n"
        "- long_report: string, Markdown, verdichtetes Langprotokoll\n"
        "- short_protocol: string, Markdown, ultra-kurzes Kurzprotokoll\n\n"
        "1. full_report als nahezu vollstaendige, segmentnahe Rekonstruktion (nur sprachlich/logisch geglaettet)\n"
        "2. Meeting-Inhalt korrekt und lueckenarm erfassen\n"
        "3. Kontext und Sinnzusammenhänge erhalten\n"
        "4. Entscheidungen, offene Punkte und To-dos sauber extrahieren\n"
        "5. Sprecherlabels nur vorsichtig verwenden, weil sie fehlerhaft sein können\n\n"
        "Regeln:\n"
        "- full_report muss chronologisch und detailreich sein; er darf nur leicht geglättet werden und soll nahe am Rohinhalt bleiben.\n"
        "- Keine starke Verdichtung im full_report.\n"
        "- long_report thematisch/sinnvoll zusammenfassen.\n"
        "- short_protocol maximal 8 Bulletpoints.\n"
        "- Keine erfundenen Fakten.\n"
        "- Alles auf Deutsch.\n\n"
        "Rohdaten:\n"
        f"{payload_text}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_full_report_messages_from_payload(payload_text: str, custom_system_prompt: str | None) -> list[dict]:
    system_prompt = custom_system_prompt or (
        "Du bist ein deutscher Protokoll-Assistent fuer Meeting-Transkripte. "
        "Deine Hauptaufgabe ist die vollstaendige und korrekte Erfassung des Meeting-Inhalts, nicht die Sprecheranalyse. "
        "Sprecherlabels stammen aus automatischer Erkennung und koennen falsch sein. Nutze sie nur vorsichtig als Orientierung. "
        "Prioritaet haben Inhalt, Kontext, Themen, Entscheidungen, offene Punkte, To-dos, Verantwortlichkeiten und naechste Schritte. "
        "Du musst ausschliesslich auf Deutsch antworten. "
        "Du musst ausschliesslich ein Markdown-Protokoll ausgeben. Keine Codefences. Kein JSON. Kein Text vor oder nach dem Protokoll. "
        "Schreibe ein vollstaendiges, detailliertes Inhaltsprotokoll mit klaren Ueberschriften und nachvollziehbarer Chronologie. "
        "Erfinde keine Fakten, Namen, Rollen, Daten oder Orte. Wenn Information fehlt, schreibe 'nicht genannt' oder lasse den Punkt weg. "
        "Wenn etwas unklar ist, schreibe 'unklar'."
    )

    user_prompt = (
        "Erstelle aus den folgenden Rohdaten ein vollstaendiges Meeting-Protokoll im Markdown-Format.\n"
        "Das Ergebnis soll nah am Rohtranskript bleiben, aber sprachlich geglättet und gut lesbar sein.\n"
        "Nutze sinnvolle Ueberschriften, Abschnitte und eine chronologische Struktur.\n"
        "Fokussiere auf den gesamten inhaltlichen Verlauf, nicht nur auf eine Kurzfassung.\n"
        "Keine JSON-Ausgabe. Keine Bulletpoint-Sammlung ohne Zusammenhang.\n\n"
        f"{payload_text}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_long_report_messages_from_payload(dialog_text: str, custom_system_prompt: str | None) -> list[dict]:
    system_prompt = custom_system_prompt or (
        "Du bist ein deutscher Protokoll-Assistent. Erzeuge ein Langprotokoll im Markdown-Format. "
        "Nutze den Verlauf als inhaltliche Grundlage und schreibe ein gut lesbares, sachliches Protokoll. "
        "Keine Stichwortliste, keine Themen-Wortwolke, keine Ein-Wort-Bullets. "
        "Das Ergebnis soll deutlich ausführlicher sein als ein Kurzprotokoll und die wichtigsten Inhalte, Entscheidungen, "
        "offenen Punkte, To-dos und Zusammenhänge in ganzen Sätzen wiedergeben. "
        "Antworte ausschließlich auf Deutsch und ohne Codefences."
    )

    user_prompt = (
        "Erstelle aus dem folgenden Gesprächsverlauf ein Langprotokoll in Markdown. "
        "Strukturvorschlag: Zusammenfassung, Kernthemen, Entscheidungen, offene Punkte / To-dos, nachvollziehbare Timeline. "
        "Wichtig: Nicht nur Schlagworte nennen, sondern die inhaltlichen Zusammenhänge in ganzen Sätzen darstellen. "
        "Wenn es keine belastbaren Entscheidungen oder To-dos gibt, schreibe das explizit.\n\n"
        f"{dialog_text}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_short_protocol_messages_from_payload(dialog_text: str, custom_system_prompt: str | None) -> list[dict]:
    system_prompt = custom_system_prompt or (
        "Du bist ein deutscher Protokoll-Assistent. Erzeuge ein sehr kurzes, aber inhaltlich präzises Kurzprotokoll im Markdown-Format. "
        "Maximal 8 Bulletpoints. Jeder Bulletpoint muss eine konkrete Aussage zu Inhalt, Entscheidung oder To-do enthalten. "
        "Keine Stichwortwolke, keine generischen Füllwörter, keine Ein-Wort-Punkte. "
        "Antworte ausschließlich auf Deutsch und ohne Codefences."
    )

    user_prompt = (
        "Erstelle aus dem folgenden Gesprächsverlauf ein Kurzprotokoll mit maximal 8 Bulletpoints. "
        "Fokussiere auf die wichtigsten Entscheidungen, nächsten Schritte und die zentralen Inhalte. "
        "Wenn keine belastbaren Entscheidungen vorliegen, schreibe das explizit. "
        "Wenn keine klaren To-dos vorliegen, schreibe das explizit.\n\n"
        f"{dialog_text}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_chunk_summary_messages(chunk_payload_text: str) -> list[dict]:
    system_prompt = (
        "Du bist ein deutscher Protokoll-Assistent. Fasse diesen Ausschnitt eines Meeting-Transkripts inhaltlich zusammen. "
        "Sprecherlabels können falsch sein. Fokus auf Inhalt, Kontext, Entscheidungen, offene Punkte, To-dos und Zeitmarken. "
        "Antworte ausschließlich als valides JSON ohne Codefences mit exakt den Keys: "
        "chunk_summary, chunk_protocol_detailed, decisions, todos, open_points, key_quotes. "
        "chunk_protocol_detailed muss eine chronologische, detailreiche und segmentnahe Rekonstruktion dieses Chunks sein "
        "(nur sprachlich geglättet, nicht stark verdichtet). "
        "Alle Werte sind Strings oder Listen von Strings auf Deutsch. Erfinde keine Fakten."
    )
    user_prompt = "Fasse diesen Meeting-Ausschnitt für eine spätere Gesamtauswertung zusammen:\n" + chunk_payload_text
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def split_entries_by_json_chars(entries: list[dict], max_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for row in entries:
        row_text = json.dumps(row, ensure_ascii=False)
        row_len = len(row_text) + 2
        if current and current_len + row_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(row)
        current_len += row_len
    if current:
        chunks.append(current)
    return chunks


def finalize_reports_with_llm(
    *,
    metadata: dict,
    entries: list[dict],
    dialog_text: str,
    args: argparse.Namespace,
) -> dict | None:
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DIZ_AI_KEY", "")
    local_hosts = ("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1")
    is_local_endpoint = str(args.llm_api_base).rstrip("/").lower().startswith(local_hosts)

    if args.llm_provider == "openai-compatible" and not api_key and not is_local_endpoint:
        print("LLM finalize requested, but no API key provided. Set --llm-api-key, OPENAI_API_KEY, or DIZ_AI_KEY.", file=sys.stderr)
        return None

    try:
        payload = build_report_payload(metadata, entries, dialog_text)
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        heuristic_full_report = build_full_report(entries, metadata)
        heuristic_long_report = build_long_report(entries, metadata)
        heuristic_short_protocol = build_short_protocol(entries, metadata)
        detailed_dialog_text = clamp_text_size(dialog_text, args.llm_max_input_chars)

        direct_timeout_seconds = max(args.llm_timeout_seconds, 3600)
        chunk_timeout_seconds = max(args.llm_timeout_seconds, 3600)

        # Direct path for short/medium meetings.
        if len(payload_text) <= args.llm_max_input_chars:
            full_report_messages = build_full_report_messages_from_payload(payload_text, args.llm_system_prompt)
            full_report_text = call_openai_compatible_chat(
                api_base=args.llm_api_base,
                api_key=api_key,
                model=args.llm_model,
                messages=full_report_messages,
                timeout_seconds=direct_timeout_seconds,
                use_json_response_format=False,
            )
            long_messages = build_long_report_messages_from_payload(detailed_dialog_text, args.llm_system_prompt)
            short_messages = build_short_protocol_messages_from_payload(detailed_dialog_text, args.llm_system_prompt)
            long_report_text = call_openai_compatible_chat(
                api_base=args.llm_api_base,
                api_key=api_key,
                model=args.llm_model,
                messages=long_messages,
                timeout_seconds=direct_timeout_seconds,
                use_json_response_format=False,
            )
            short_protocol_text = call_openai_compatible_chat(
                api_base=args.llm_api_base,
                api_key=api_key,
                model=args.llm_model,
                messages=short_messages,
                timeout_seconds=direct_timeout_seconds,
                use_json_response_format=False,
            )
            return {
                "full_report": str(full_report_text or heuristic_full_report),
                "long_report": str(long_report_text or heuristic_long_report),
                "short_protocol": str(short_protocol_text or heuristic_short_protocol),
            }

        # Chunked path for long meetings: summarize blocks, then create final reports.
        print("LLM input is large. Using two-stage chunked summarization before final report generation.")
        chunk_size = max(20_000, args.llm_chunk_input_chars)
        entry_chunks = split_entries_by_json_chars(entries, chunk_size)
        chunk_summaries: list[dict] = []

        for idx, chunk_entries in enumerate(entry_chunks, start=1):
            chunk_payload = {
                "chunk_index": idx,
                "chunk_count": len(entry_chunks),
                "metadata": metadata,
                "entries": chunk_entries,
            }
            chunk_payload_text = json.dumps(chunk_payload, ensure_ascii=False, indent=2)
            messages = build_chunk_summary_messages(chunk_payload_text)
            decoded = call_llm_json_with_retry(
                api_base=args.llm_api_base,
                api_key=api_key,
                model=args.llm_model,
                messages=messages,
                timeout_seconds=chunk_timeout_seconds,
                required_keys=None,
            )
            chunk_summary = str(decoded.get("chunk_summary") or decoded.get("summary") or "")
            chunk_protocol_detailed = str(decoded.get("chunk_protocol_detailed") or decoded.get("chunk_protocol") or chunk_summary)
            chunk_summaries.append(
                {
                    "chunk_index": idx,
                    "chunk_summary": chunk_summary,
                    "chunk_protocol_detailed": chunk_protocol_detailed,
                    "decisions": decoded.get("decisions", []),
                    "todos": decoded.get("todos", []),
                    "open_points": decoded.get("open_points", []),
                    "key_quotes": decoded.get("key_quotes", []),
                }
            )
            print(f"LLM chunk summary {idx}/{len(entry_chunks)} finished.")

        final_payload = {
            "metadata": metadata,
            "chunk_summaries": chunk_summaries,
            "raw_entries_head_tail": {
                "head": entries[: min(8, len(entries))],
                "tail": entries[-min(8, len(entries)) :] if entries else [],
            },
            "instruction": (
                "Nutze insbesondere chunk_protocol_detailed pro Chunk als primäre Quelle fuer den full_report. "
                "Die raw_entries_head_tail dienen nur zur Orientierung. "
                "Erzeuge die finalen drei Reports vollständig auf Deutsch."
            ),
        }
        final_payload_text = json.dumps(final_payload, ensure_ascii=False, indent=2)
        full_report_messages = build_full_report_messages_from_payload(clamp_text_size(final_payload_text, args.llm_max_input_chars), args.llm_system_prompt)
        full_report_text = call_openai_compatible_chat(
            api_base=args.llm_api_base,
            api_key=api_key,
            model=args.llm_model,
            messages=full_report_messages,
            timeout_seconds=direct_timeout_seconds,
            use_json_response_format=False,
        )
        long_messages = build_long_report_messages_from_payload(detailed_dialog_text, args.llm_system_prompt)
        short_messages = build_short_protocol_messages_from_payload(detailed_dialog_text, args.llm_system_prompt)
        long_report_text = call_openai_compatible_chat(
            api_base=args.llm_api_base,
            api_key=api_key,
            model=args.llm_model,
            messages=long_messages,
            timeout_seconds=direct_timeout_seconds,
            use_json_response_format=False,
        )
        short_protocol_text = call_openai_compatible_chat(
            api_base=args.llm_api_base,
            api_key=api_key,
            model=args.llm_model,
            messages=short_messages,
            timeout_seconds=direct_timeout_seconds,
            use_json_response_format=False,
        )
        return {
            "full_report": str(full_report_text or heuristic_full_report),
            "long_report": str(long_report_text or heuristic_long_report),
            "short_protocol": str(short_protocol_text or heuristic_short_protocol),
        }

    except Exception as exc:  # noqa: BLE001
        print(f"LLM finalization failed, keeping heuristic reports. Reason: {exc}", file=sys.stderr)
        return None


def regenerate_reports_from_raw_json(args: argparse.Namespace) -> int:
    raw_json_path = Path(args.raw_json)
    if not raw_json_path.is_absolute():
        if args.meeting_folder:
            raw_json_path = Path(args.meeting_folder) / raw_json_path.name
        else:
            raw_json_path = Path.cwd() / raw_json_path

    metadata, entries = load_raw_session_payload(raw_json_path)
    session_dir = raw_json_path.parent
    metadata = dict(metadata)
    metadata.setdefault("session_dir", str(session_dir))
    metadata.setdefault("sample_rate", 16000)  # Ensure sample_rate is not None for LLM serialization

    txt_path = resolve_sidecar_path(session_dir, args.txt)
    log_path = resolve_sidecar_path(session_dir, args.log)
    speaker_log_path = resolve_sidecar_path(session_dir, args.speaker_log)
    dialog_path = resolve_sidecar_path(session_dir, args.dialog)
    full_report_path = resolve_sidecar_path(session_dir, args.full_report)
    long_report_path = resolve_sidecar_path(session_dir, args.long_report)
    short_report_path = resolve_sidecar_path(session_dir, args.short_report)

    transcript_text = "\n".join(
        f"[{format_mmss(row['start_seconds'])}-{format_mmss(row['end_seconds'])}] {row.get('speaker', 'Unknown')}: {row['text']}"
        for row in entries
    )
    txt_path.write_text(transcript_text + ("\n" if transcript_text else ""), encoding="utf-8")

    if entries:
        speaker_log_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + "\n",
            encoding="utf-8",
        )
        log_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + "\n",
            encoding="utf-8",
        )
    else:
        speaker_log_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")

    dialog_text = build_dialog_report(entries, str(metadata.get("title", "Meeting Report")))
    full_report = build_full_report(entries, metadata)
    long_report = build_long_report(entries, metadata)
    short_protocol = build_short_protocol(entries, metadata)

    if args.llm_finalize:
        llm_result = finalize_reports_with_llm(metadata=metadata, entries=entries, dialog_text=dialog_text, args=args)
        if llm_result is not None:
            full_report = llm_result["full_report"]
            long_report = llm_result["long_report"]
            short_protocol = llm_result["short_protocol"]
            print("LLM finalization applied to regenerated full/long/short reports.")

    dialog_path.write_text(dialog_text, encoding="utf-8")
    full_report_path.write_text(full_report, encoding="utf-8")
    long_report_path.write_text(long_report, encoding="utf-8")
    short_report_path.write_text(short_protocol, encoding="utf-8")

    print(f"Loaded raw data from: {raw_json_path}")
    print(f"Saved transcript to: {txt_path}")
    print(f"Saved dialog report to: {dialog_path}")
    print(f"Saved full report to: {full_report_path}")
    print(f"Saved long report to: {long_report_path}")
    print(f"Saved short protocol to: {short_report_path}")
    print("Report regeneration finished.")
    return 0


# -----------------------------------------------------------------------------
# Runtime orchestration
# -----------------------------------------------------------------------------


def create_session_dir(args: argparse.Namespace) -> Path:
    if args.meeting_folder:
        session_dir = Path(args.meeting_folder)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path(args.storage_root) / f"{args.session_prefix}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


class LiveChunkWindow:
    """Fullscreen-friendly Tkinter window that shows live transcription chunks.

    Useful as a "TV screen" next to a played video so the running transcript is
    visible in real time.
    """

    def __init__(self, title: str = "Live Transcription", font_size: int = 28) -> None:
        try:
            import tkinter as tk
            from tkinter import scrolledtext
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Tkinter is not available for TV window mode.") from exc

        self._tk = tk
        self._closed = False
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry("1280x760")
        self._root.configure(bg="#0A0D14")

        header = tk.Label(
            self._root,
            text="LIVE TRANSCRIPTION",
            font=("Segoe UI", max(16, font_size - 8), "bold"),
            bg="#0A0D14",
            fg="#9AA7C2",
            anchor="w",
            padx=20,
            pady=12,
        )
        header.pack(fill="x")

        self._text = scrolledtext.ScrolledText(
            self._root,
            wrap="word",
            font=("Segoe UI", font_size, "bold"),
            bg="#0F1526",
            fg="#F4F7FF",
            insertbackground="#F4F7FF",
            padx=20,
            pady=20,
            spacing1=8,
            spacing3=14,
        )
        self._text.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self._text.configure(state="disabled")

        def on_close() -> None:
            self._closed = True
            self._root.destroy()

        self._root.protocol("WM_DELETE_WINDOW", on_close)
        self.pump()

    def _append(self, line: str) -> None:
        if self._closed:
            return
        self._text.configure(state="normal")
        self._text.insert("end", line + "\n\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def add_chunk(self, chunk_index: int, text: str) -> None:
        if self._closed:
            return
        content = text if text else "[kein erkannter Text]"
        self._append(f"[{chunk_index}] {content}")
        self.pump()

    def add_status(self, text: str) -> None:
        if self._closed:
            return
        self._append(f"[status] {text}")
        self.pump()

    def pump(self) -> None:
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True


def create_live_window(args: argparse.Namespace) -> Any:
    if not args.tv_window:
        return None
    try:
        window = LiveChunkWindow(title=args.tv_title, font_size=args.tv_font_size)
    except RuntimeError as exc:
        print(f"TV window requested, but unavailable: {exc}", file=sys.stderr)
        return None
    window.add_status("Transcription started")
    return window


def run_vclean(args: argparse.Namespace) -> int:
    if args.regenerate_reports:
        return regenerate_reports_from_raw_json(args)

    if args.list_devices:
        list_devices()
        return 0

    if args.output_loopback and args.mic:
        raise RuntimeError("--output-loopback and --mic cannot be used together.")

    session_dir = create_session_dir(args)
    out_path = resolve_sidecar_path(session_dir, args.out)
    txt_path = resolve_sidecar_path(session_dir, args.txt)
    log_path = resolve_sidecar_path(session_dir, args.log)
    raw_path = resolve_sidecar_path(session_dir, args.raw_json)
    speaker_log_path = resolve_sidecar_path(session_dir, args.speaker_log)
    profiles_out_path = resolve_sidecar_path(session_dir, args.speaker_profiles_out)
    dialog_path = resolve_sidecar_path(session_dir, args.dialog)
    full_report_path = resolve_sidecar_path(session_dir, args.full_report)
    long_report_path = resolve_sidecar_path(session_dir, args.long_report)
    short_report_path = resolve_sidecar_path(session_dir, args.short_report)

    for path in [txt_path, log_path, raw_path, speaker_log_path, dialog_path, full_report_path, long_report_path, short_report_path]:
        path.write_text("", encoding="utf-8")

    metadata = {
        "title": f"Meeting Report {time.strftime('%Y-%m-%d %H:%M')}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_dir": str(session_dir),
        "audio_path": str(out_path),
        "txt_path": str(txt_path),
        "log_path": str(log_path),
        "duration_seconds": float(args.duration),
        "model": args.model,
        "speaker_backend": args.speaker_backend,
        "output_loopback": bool(args.output_loopback),
        "device": args.device,
        "device_index": args.device_index,
        "sample_rate": 16000,  # Default fallback value; updated dynamically below
    }

    model_size = enforce_large_model(args.model)
    whisper_model = create_whisper_model(model_size)
    embedder = SpeakerBackbone(prefer_gpu=not args.cpu_only, backend=args.speaker_backend)
    clustering = OnlineSpeakerClustering(
        max_speakers=args.max_speakers,
        distance_threshold=args.speaker_distance_threshold,
        known_profiles=load_profiles(args.speaker_profiles_in),
    )
    live_window = create_live_window(args)

    full_audio_parts: list[np.ndarray] = []
    entries: list[dict] = []
    interrupted = False

    if args.output_loopback:
        pyaudio = capture_output_loopback_stream()
        frames_per_buffer = 1024

        with pyaudio.PyAudio() as p:
            loopback_info = resolve_pyaudio_loopback_device(p, pyaudio, args.device, args.device_index)
            sample_rate = int(loopback_info["defaultSampleRate"])
            channels = max(1, min(2, int(loopback_info.get("maxInputChannels", 2))))
            metadata["sample_rate"] = sample_rate
            print(f"Dynamic source: system audio output loopback: {loopback_info.get('name', 'unknown')}")

            processor = SegmentProcessor(
                model=whisper_model,
                model_size=model_size,
                sample_rate=sample_rate,
                embedder=embedder,
                clustering=clustering,
                txt_path=txt_path,
                log_path=log_path,
                speaker_log_path=speaker_log_path,
                live_window=live_window,
            )
            processor.start()

            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer,
                input_device_index=int(loopback_info["index"]),
            )

            segmenter = EnergySegmenter(
                sample_rate=sample_rate,
                min_speech_seconds=args.min_speech_seconds,
                min_silence_seconds=args.min_silence_seconds,
                max_segment_seconds=args.max_segment_seconds,
            )

            start_time = time.time()
            captured_frames = 0
            try:
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= args.duration:
                        break
                    raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
                    arr_f32 = arr.astype(np.float32) / 32768.0
                    full_audio_parts.append(arr_f32)
                    mono = np.mean(arr_f32, axis=1) if arr_f32.ndim == 2 else arr_f32
                    frame_start = captured_frames / sample_rate
                    captured_frames += arr_f32.shape[0]
                    for segment in segmenter.feed(np.asarray(mono, dtype=np.float32), frame_start):
                        processor.submit(segment)
            except KeyboardInterrupt:
                interrupted = True
                print("Manual stop received, finalizing quickly...")
            finally:
                stream.stop_stream()
                stream.close()

            tail = segmenter.flush(captured_frames / sample_rate)
            if tail is not None:
                processor.submit(tail)
            processor.close()
            entries = processor.entries

    else:
        device_idx, channels, sample_rate, extra_settings, source_name = resolve_recording_config(args.device, args.mic, args.device_index)
        metadata["sample_rate"] = sample_rate
        print(f"Dynamic source: {source_name}")

        queue: Queue[np.ndarray] = Queue()
        stop_event = Event()
        segmenter = EnergySegmenter(
            sample_rate=sample_rate,
            min_speech_seconds=args.min_speech_seconds,
            min_silence_seconds=args.min_silence_seconds,
            max_segment_seconds=args.max_segment_seconds,
        )
        processor = SegmentProcessor(
            model=whisper_model,
            model_size=model_size,
            sample_rate=sample_rate,
            embedder=embedder,
            clustering=clustering,
            txt_path=txt_path,
            log_path=log_path,
            speaker_log_path=speaker_log_path,
            live_window=live_window,
        )
        processor.start()

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"Audio stream status: {status}", file=sys.stderr)
            if not stop_event.is_set():
                queue.put(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device_idx,
            extra_settings=extra_settings,
            callback=callback,
        )

        start_time = time.time()
        captured_frames = 0
        try:
            with stream:
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= args.duration:
                        break
                    try:
                        raw = queue.get(timeout=0.2)
                    except Empty:
                        continue
                    full_audio_parts.append(raw)
                    mono = np.mean(raw, axis=1) if raw.ndim == 2 else raw
                    frame_start = captured_frames / sample_rate
                    captured_frames += raw.shape[0]
                    for segment in segmenter.feed(np.asarray(mono, dtype=np.float32), frame_start):
                        processor.submit(segment)
        except KeyboardInterrupt:
            interrupted = True
            print("Manual stop received, finalizing quickly...")
        finally:
            stop_event.set()

        tail = segmenter.flush(captured_frames / sample_rate)
        if tail is not None:
            processor.submit(tail)
        processor.close()
        entries = processor.entries

    if full_audio_parts and not args.no_save_audio:
        full_audio = np.concatenate(full_audio_parts, axis=0)
        sf.write(str(out_path), np.clip(full_audio, -1.0, 1.0), int(metadata["sample_rate"]), subtype="PCM_16")

    speaker_profiles = clustering.export_profiles()
    save_profiles(profiles_out_path, speaker_profiles)

    raw_payload = {
        "metadata": metadata,
        "entries": entries,
        "speaker_profiles": {name: vec.tolist() for name, vec in speaker_profiles.items()},
        "interrupted": interrupted,
    }
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    speaker_log_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )

    transcript_text = "\n".join(
        f"[{format_mmss(row['start_seconds'])}-{format_mmss(row['end_seconds'])}] {row.get('speaker', 'Unknown')}: {row['text']}"
        for row in entries
    )
    txt_path.write_text(transcript_text + ("\n" if transcript_text else ""), encoding="utf-8")

    dialog_text = build_dialog_report(entries, metadata["title"])
    full_report = build_full_report(entries, metadata)
    long_report = build_long_report(entries, metadata)
    short_protocol = build_short_protocol(entries, metadata)

    if args.llm_finalize:
        llm_result = finalize_reports_with_llm(metadata=metadata, entries=entries, dialog_text=dialog_text, args=args)
        if llm_result is not None:
            full_report = llm_result["full_report"]
            long_report = llm_result["long_report"]
            short_protocol = llm_result["short_protocol"]
            print("LLM finalization applied to full/long/short reports.")

    dialog_path.write_text(dialog_text, encoding="utf-8")
    full_report_path.write_text(full_report, encoding="utf-8")
    long_report_path.write_text(long_report, encoding="utf-8")
    short_report_path.write_text(short_protocol, encoding="utf-8")

    print(f"Saved raw data to: {raw_path}")
    print(f"Saved transcript to: {txt_path}")
    print(f"Saved dialog report to: {dialog_path}")
    print(f"Saved full report to: {full_report_path}")
    print(f"Saved long report to: {long_report_path}")
    print(f"Saved short protocol to: {short_report_path}")
    print(f"Saved speaker log to: {speaker_log_path}")
    print(f"Saved speaker profiles to: {profiles_out_path}")
    if not args.no_save_audio:
        print(f"Saved audio to: {out_path}")

    print("Manual stop received at top-level. Exiting cleanly." if interrupted else "Meeting processing finished.")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vClean: dynamic meeting recorder with Whisper, optional SpeechBrain, and Ollama/OpenAI-compatible report finalization."
    )
    parser.add_argument("--duration", type=int, default=600, help="Recording length in seconds")
    parser.add_argument("--out", type=Path, default=Path("meeting.wav"), help="Output wav file")
    parser.add_argument("--txt", type=Path, default=Path("meeting.txt"), help="Transcript text file")
    parser.add_argument("--log", type=Path, default=Path("meeting_raw_segments.jsonl"), help="Raw segment log jsonl")
    parser.add_argument("--raw-json", type=Path, default=Path("meeting_raw.json"), help="Raw structured data json")
    parser.add_argument("--dialog", type=Path, default=Path("meeting_dialog.md"), help="Dialog / speaker detail markdown")
    parser.add_argument("--full-report", type=Path, default=Path("meeting_report_full.md"), help="Full content report markdown")
    parser.add_argument("--long-report", type=Path, default=Path("meeting_report_long.md"), help="Condensed long report markdown")
    parser.add_argument("--short-report", type=Path, default=Path("meeting_protocol_short.md"), help="Ultra short protocol markdown")
    parser.add_argument("--speaker-log", type=Path, default=Path("meeting_speakers.jsonl"), help="Speaker log jsonl")
    parser.add_argument("--speaker-profiles-out", type=Path, default=Path("speaker_profiles_auto.json"), help="Speaker profile output json")
    parser.add_argument("--speaker-profiles-in", type=Path, default=None, help="Input speaker profiles json")
    parser.add_argument("--speaker-backend", choices=["speechbrain", "fallback"], default="speechbrain")
    parser.add_argument("--max-speakers", type=int, default=8, help="Max unknown speaker clusters")
    parser.add_argument("--speaker-distance-threshold", type=float, default=0.28, help="Speaker merge/new threshold")
    parser.add_argument("--min-speech-seconds", type=float, default=0.55)
    parser.add_argument("--min-silence-seconds", type=float, default=0.35)
    parser.add_argument("--max-segment-seconds", type=float, default=18.0)
    parser.add_argument("--model", default="large-v3", help="Whisper model size (enforced large-v3)")
    parser.add_argument("--device", default=None, help="Optional substring for output device name")
    parser.add_argument("--device-index", type=int, default=None, help="Exact device index from --list-devices")
    parser.add_argument("--mic", action="store_true", help="Record microphone instead of system audio")
    parser.add_argument("--output-loopback", action="store_true", help="Use WASAPI output loopback backend via pyaudiowpatch")
    parser.add_argument("--no-save-audio", action="store_true", help="Do not write final wav")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU use for SpeechBrain")
    parser.add_argument("--list-devices", action="store_true", help="Print devices and exit")
    parser.add_argument("--tv-window", action="store_true", help="Show live chunks in a large TV-style window")
    parser.add_argument("--tv-font-size", type=int, default=28, help="Font size for --tv-window")
    parser.add_argument("--tv-title", default="Live Transcription", help="Window title for --tv-window")
    parser.add_argument("--storage-root", default="meetings", help="Base folder for timestamped meeting sessions")
    parser.add_argument("--session-prefix", default="meeting", help="Prefix for auto-created session folder")
    parser.add_argument("--meeting-folder", default=None, help="Explicit folder for this meeting session")
    parser.add_argument("--regenerate-reports", action="store_true", help="Regenerate full/long/short reports from an existing meeting_raw.json")

    parser.add_argument(
        "--llm-finalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LLM at the end to generate final full/long/short reports",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai-compatible"],
        default="openai-compatible",
        help="LLM backend type",
    )
    parser.add_argument(
        "--llm-api-base",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL; for Ollama use http://localhost:11434/v1",
    )
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="Model name for report finalization")
    parser.add_argument("--llm-api-key", default=None, help="API key; not needed for local Ollama endpoint")
    parser.add_argument("--llm-timeout-seconds", type=int, default=6000, help="Timeout for each LLM request")
    parser.add_argument("--llm-max-input-chars", type=int, default=220000, help="Max chars for direct final LLM pass")
    parser.add_argument("--llm-chunk-input-chars", type=int, default=80000, help="Max chars per chunk in two-stage LLM mode")
    parser.add_argument("--llm-system-prompt", default=None, help="Optional custom system prompt for final LLM reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_vclean(args)
    except KeyboardInterrupt:
        print("Manual stop received at top-level. Exiting cleanly.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
