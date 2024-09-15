# based on ESPNet2 add_noise and convolve_rir
from pathlib import Path
import torch
import torchaudio
import scipy
import numpy as np

from tomato.utils import logger
from . import audio_util

class AudioAugmentation:

    def __init__(self):
        # TODO: move these to config
        noise_path = Path("/resources/speech/corpora/Noise/musan/noise")
        self.noises = self._get_all_noise_paths(noise_path)
        rir_path = Path("/resources/speech/corpora/Noise/RIRS_NOISES")
        self.rir = self._get_all_noise_paths(rir_path)
        self.noise_db_low, self.noise_db_high = 5, 20 # TODO: set by config
        self.add_noise_prob = 0.5
        self.convolve_rir_prob = 0.5
        # alway do single channel

    def pink_call(self, audio):
        noise_prob = np.random.rand()
        if noise_prob > self.add_noise_prob:
            audio, noise = self.add_pink_noise(audio)
        return audio

    def __call__(self, audio):
        self.audio_power = float((audio**2).mean())
        noise_prob = np.random.rand()
        rir_prob = np.random.rand()
        #logger.info(f"noise_prob: {noise_prob}, rir_prob: {rir_prob}")
        if rir_prob > self.convolve_rir_prob:
            audio, rir = self.convolve_rir(audio)
        if noise_prob > self.add_noise_prob:
            audio, noise = self.add_noises(audio)
        return audio

    def _get_all_noise_paths(self, path):
        noises = []
        # recursively search for files that endswith .wav 
        for noise in path.rglob("*.wav"):
            noises.append(noise)
        return noises
    
    def add_noises(self, audio):
        noise_path = np.random.choice(self.noises)
        noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
        noise, sample_rate = audio_util.get_audio(noise_path, to_mono=True, trim_sil=False)
        
        audio_nsamples = audio.size(1)
        noise_nsamples = noise.size(1)
        # align noise and audio such that they have the same length
        if audio_nsamples == noise_nsamples:
            pass
        elif audio_nsamples > noise_nsamples:
            offset = np.random.randint(0, audio_nsamples - noise_nsamples)
            noise = np.pad(
                noise,
                [(0, 0), (offset, audio_nsamples - noise_nsamples - offset)],
                mode = "wrap"
            )
        else:
            offset = np.random.randint(0, noise_nsamples - audio_nsamples)
            noise = noise[:, offset:offset + audio_nsamples]
        noise_power = (noise**2).mean()
        scale = (
            10 ** (-noise_db / 20) 
            * np.sqrt(self.audio_power)
            / np.sqrt(max(noise_power, 1e-10))
        )
        audio = audio + scale * noise
        return audio, noise
        
    def convolve_rir(self, audio):
        rir_path = np.random.choice(self.rir)
        rir, sample_rate = audio_util.get_audio(rir_path, to_mono=True, trim_sil=False)
        
        augmented = scipy.signal.convolve(audio.numpy(), rir.numpy(), 
                                          mode="full")[:, :audio.size(1)]
        # reverse mean power
        augment_power = (augmented**2).mean()
        scale = float(np.sqrt(self.audio_power / max(augment_power, 1e-10)))
        augmented = scale * augmented 
        return torch.from_numpy(augmented), rir.squeeze(1)

    def add_pink_noise(self, audio, noise_std=0.1):
        num_rows = 16
        length = audio.size(1)
        array = torch.randn(num_rows, length // num_rows + 1)
        reshaped_array = torch.cumsum(array, dim=1)
        reshaped_array = reshaped_array.reshape(-1)
        reshaped_array = reshaped_array[:length]
        # Normalize
        pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
        pink_noise = pink_noise.unsqueeze(0)
        audio = audio + pink_noise * noise_std
        return audio, pink_noise

    def add_music(self, audio):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    audio_path = os.environ.get("AUDIO_PATH")
    out_dir = os.environ.get("OUT_DIR")
    audio, sample_rate = torchaudio.load(audio_path, normalize=True)
    target_sample_rate = 16_000
    audio, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        audio, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    print(f"sample rate: {sample_rate}")

    audio_augmentation = AudioAugmentation()
    audio, noise = audio_augmentation.add_pink_noise(audio)
    out_audio_path = Path(out_dir) / "audio.wav"
    out_noise_path = Path(out_dir) / "noise.wav"
    
    # save audio
    torchaudio.save(str(out_audio_path), audio, sample_rate)
    torchaudio.save(str(out_noise_path), noise, sample_rate)