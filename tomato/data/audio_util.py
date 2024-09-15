# Adpated from: https://github.com/piotrkawa/deepfake-whisper-features
import torchaudio
import numpy as np

SAMPLING_RATE = 16_000
SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def get_audio(audio_path, to_mono=True, trim_sil=False, **kwargs):
    waveform, sample_rate = torchaudio.load(audio_path, **kwargs)
    if sample_rate != SAMPLING_RATE:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    if to_mono:
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform[:1, ...]

    if trim_sil:
    # trim too long utterances
        waveform, sample_rate = apply_trim(waveform, sample_rate)
    return waveform, sample_rate

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
    audio_path = "wav/audio.wav"
    audio = read_audio(audio_path, 4)
    torchaudio.save("wav/audio_trimmed.wav", audio, 16000)