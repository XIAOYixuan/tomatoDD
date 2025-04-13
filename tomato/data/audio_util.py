# Adpated from: https://github.com/piotrkawa/deepfake-whisper-features
import torch
import numpy as np
import librosa
import soundfile as sf
import logging

logging.basicConfig(level=logging.WARNING)

# Set the logging level for Numba
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

SAMPLING_RATE = 16_000

def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = librosa.resample(waveform, sample_rate, target_sample_rate)
    return waveform, sample_rate

def trim_silence(audio: np.ndarray, 
                 threshold: float = 30, 
                 win_length: int = 1024, 
                 shift_length: int = 256) -> np.ndarray:
    """
    Trims the leading and trailing silence of the given audio.

    :param audio: A numpy.ndarray representing the audio signal.
    :param threshold: The threshold in decibels used for silence detection. Default is 30 dB.
    :param win_length: The window length for analysis in points. Default is 1024.
    :param shift_length: The shift length for the analysis window in points. Default is 256.
    :return: A numpy.ndarray of the trimmed audio.
    """
    # Use librosa's trim function to remove leading and trailing silence based on the specified parameters.
    trimmed_audio, _ = librosa.effects.trim(
        y=audio,
        top_db=threshold,
        frame_length=win_length,
        hop_length=shift_length
    )
    
    return trimmed_audio

def get_audio(
    audio_path,
    to_mono=True,
    norm=True,
    trim_sil=True,
    target_rms=-20,
    frame_offset=0,
    num_frames=-1
):
    """
    Load an audio file using soundfile, optionally resample with librosa, 
    optionally convert to mono, and optionally RMS-normalize to target_rms dB.
    
    :param audio_path: Path to the audio file.
    :param to_mono: If True, keep only the first channel (as in original code).
    :param norm: If True, apply RMS-based normalization to target_rms dB.
    :param trim_sil: If True, trim leading and trailing silence.
    :param target_rms: Target RMS level in dB (default -20 dB).
    :param frame_offset: Number of frames to skip from the beginning.
    :param num_frames: How many frames to read. None or negative => read all.
    :return: A tuple (waveform, sample_rate), 
             where waveform is a NumPy array of shape (num_samples,) or (channels, num_samples).
    """
    with sf.SoundFile(audio_path, 'r') as f:
        if frame_offset > 0:
            f.seek(frame_offset)
        data = f.read(frames=num_frames, dtype='float32', always_2d=False)
        sample_rate = f.samplerate

    if data.ndim != 1:
        data = data.T

    if sample_rate != SAMPLING_RATE:
        if data.ndim == 1:
            data = librosa.resample(y=data, orig_sr=sample_rate, target_sr=SAMPLING_RATE)
        else:
            raise NotImplementedError

    if to_mono:
        if data.ndim > 1 and data.shape[0] > 1:
            data = data[0]

    if trim_sil:
        data = trim_silence(data)

    if norm:
        rms = np.sqrt(np.mean(np.square(data)))
        target_rms_linear = 10 ** (target_rms / 20.0)
        scaling_factor = target_rms_linear / rms
        data = data * scaling_factor
        data = np.clip(data, -1.0, 1.0)
    
    data = data.reshape(1, -1)
    data_tensor = torch.from_numpy(data)
    return data_tensor, sample_rate


if __name__ == "__main__":
    # NOTE: need to set random seed for testing 
    def read_audio(audio_path, max_len):
        feats, sample_rate = get_audio(audio_path, to_mono=True, trim_sil=True)
        max_len = 4 * sample_rate
        if feats.shape[1] == max_len:
            return feats
        elif feats.shape[1] < max_len: # max len: 4 secs, 64000
            num_repeats = int(max_len / feats.shape[1]) + 1
            feats = feats.repeat(1, num_repeats)
        stt = np.random.randint(feats.shape[1] - max_len)
        feats = feats[:, stt:stt+max_len]
        return feats

    # read 4 seconds
    audio_path = "wav/audio.flac"
    #audio = read_audio(audio_path, 4)
    audio, sample_rate = get_audio(audio_path, to_mono=True, trim_sil=True)
    audio_np = audio.numpy()
    print(audio_np.shape)
    # Convert float32 to int16 for FLAC
    audio_np = (audio_np * 32767).astype(np.int16)
    sf.write("wav/audio_trimmed.flac", audio_np.T, SAMPLING_RATE, format='FLAC')
