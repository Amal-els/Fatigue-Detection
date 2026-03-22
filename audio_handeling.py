import wave

import av
import numpy as np


class AudioFrameHandler:
    """Play a custom alarm sound over the WebRTC audio stream."""

    def __init__(self, sound_file_path: str = ""):
        self.custom_audio, self.custom_audio_rate = self._load_wav(sound_file_path)

        self.audio_segment_shape: tuple = ()
        self.play_state_tracker: dict = {"curr_segment": -1}
        self.audio_segments_created: bool = False
        self.audio_segments: list[np.ndarray] = []

    def _load_wav(self, sound_file_path: str) -> tuple[np.ndarray, int]:
        with wave.open(sound_file_path, "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_audio = wav_file.readframes(frame_count)

        dtype_map = {
            1: np.uint8,
            2: np.int16,
            4: np.int32,
        }
        if sample_width not in dtype_map:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        audio = np.frombuffer(raw_audio, dtype=dtype_map[sample_width]).copy()
        audio = audio.reshape(-1, channels)

        if sample_width == 1:
            # 8-bit PCM WAV is unsigned, while AV frames use signed-style math.
            audio = audio.astype(np.int16) - 128

        return audio, sample_rate

    def _resample_audio(self, audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or len(audio) == 0:
            return audio

        duration = len(audio) / source_rate
        target_length = max(int(round(duration * target_rate)), 1)

        source_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)

        resampled_channels = [
            np.interp(target_positions, source_positions, audio[:, channel_idx])
            for channel_idx in range(audio.shape[1])
        ]

        return np.stack(resampled_channels, axis=1)

    def _match_output_format(self, frame: av.AudioFrame) -> np.ndarray:
        audio = self.custom_audio.astype(np.float32)
        target_channels = len(frame.layout.channels)

        if audio.shape[1] != target_channels:
            if audio.shape[1] == 1:
                audio = np.repeat(audio, target_channels, axis=1)
            else:
                audio = audio[:, :target_channels]

        audio = self._resample_audio(audio, self.custom_audio_rate, frame.sample_rate)

        dtype = frame.to_ndarray().dtype
        info = np.iinfo(dtype)
        return np.clip(np.round(audio), info.min, info.max).astype(dtype)

    def _frame_samples_to_shape(self, samples: np.ndarray, reference_shape: tuple) -> np.ndarray:
        if len(reference_shape) == 1:
            return samples.reshape(reference_shape)

        if reference_shape[0] == samples.shape[1]:
            return samples.T.reshape(reference_shape)

        return samples.reshape(reference_shape)

    def prepare_audio(self, frame: av.AudioFrame):
        raw_samples = frame.to_ndarray()
        self.audio_segment_shape = raw_samples.shape

        matched_audio = self._match_output_format(frame)
        samples_per_segment = raw_samples.shape[-1]

        self.audio_segments = []
        for start_idx in range(0, len(matched_audio), samples_per_segment):
            segment = matched_audio[start_idx : start_idx + samples_per_segment]
            if len(segment) < samples_per_segment:
                padding = np.zeros(
                    (samples_per_segment - len(segment), matched_audio.shape[1]),
                    dtype=matched_audio.dtype,
                )
                segment = np.vstack([segment, padding])
            self.audio_segments.append(segment)

        if not self.audio_segments:
            silent_segment = np.zeros((samples_per_segment, len(frame.layout.channels)), dtype=raw_samples.dtype)
            self.audio_segments.append(silent_segment)

        self.total_segments = len(self.audio_segments) - 1
        self.audio_segments_created = True

    def process(self, frame: av.AudioFrame, play_sound: bool = False):
        """
        Return either the next alarm segment or a silent audio frame.
        """

        if not self.audio_segments_created:
            self.prepare_audio(frame)

        raw_samples = frame.to_ndarray()
        _curr_segment = self.play_state_tracker["curr_segment"]

        if play_sound:
            if _curr_segment < self.total_segments:
                _curr_segment += 1
            else:
                _curr_segment = 0

            segment = self.audio_segments[_curr_segment]
        else:
            if -1 < _curr_segment < self.total_segments:
                _curr_segment += 1
                segment = self.audio_segments[_curr_segment]
            else:
                _curr_segment = -1
                segment = np.zeros((raw_samples.shape[-1], len(frame.layout.channels)), dtype=raw_samples.dtype)

        self.play_state_tracker["curr_segment"] = _curr_segment

        new_samples = self._frame_samples_to_shape(segment, self.audio_segment_shape)
        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate

        return new_frame
