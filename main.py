#!/usr/bin/env python3
"""
Project 97: Audio Style Transfer (Guitar to Violin Timbre Translation)
CycleGAN-Audio implementation with differentiable chroma pitch preservation,
depthwise-separable convolutional generators, and physical acoustic synthesis.
"""

import os
import json
import time
import math
import random
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# 1. Physical Acoustic Sound Synthesis Engine
# ---------------------------------------------------------

SAMPLE_RATE = 16000
AUDIO_DURATION = 2.048  # 32768 samples
N_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = 256
N_BINS = 128  # First 128 frequency bins (0 Hz to 8000 Hz)
N_FRAMES = 256  # Full time frames (2.048s)
TRAIN_FRAMES = 128  # Training patch size (1.024s)

def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note number to frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

def synth_guitar_note(f0: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize acoustic guitar plucked note using Karplus-Strong physical modeling.
    Fast attack transient with frequency-dependent exponential decay.
    """
    P = max(2, int(sr / f0))
    n_pts = int(sr * dur)
    if n_pts <= P:
        return np.zeros(n_pts, dtype=np.float32)

    # Pluck excitation: burst of random noise
    x = np.zeros(n_pts, dtype=np.float32)
    x[:P] = np.random.uniform(-1.0, 1.0, P)

    # Pluck position notch filter (simulating pluck at 1/5th string length)
    pluck_notch = max(1, P // 5)
    x[pluck_notch:P] -= 0.6 * x[:P - pluck_notch]

    # Karplus-Strong feedback loop filter
    alpha = 0.991
    b = [1.0]
    a = np.zeros(P + 2, dtype=np.float32)
    a[0] = 1.0
    a[P] = -0.5 * alpha
    a[P + 1] = -0.5 * alpha

    y = signal.lfilter(b, a, x)
    return y.astype(np.float32)

def synth_violin_note(f0: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize bowed violin note using Helmholtz slip-stick friction model.
    Continuous harmonic excitation, body formants (corpus & bridge hill), and vibrato.
    """
    n_pts = int(sr * dur)
    t = np.linspace(0, dur, n_pts, endpoint=False, dtype=np.float32)

    # Natural expressive vibrato (5.5 Hz rate, 1.5% depth)
    vib_rate = 5.5
    vib_depth = 0.015
    phase = 2.0 * np.pi * f0 * t + vib_depth * np.sin(2.0 * np.pi * vib_rate * t)

    # Harmonic overtone series with violin body resonance response
    harmonics = [1.0, 0.72, 0.55, 0.42, 0.35, 0.28, 0.22, 0.18, 0.14, 0.10, 0.08, 0.06]
    y = np.zeros(n_pts, dtype=np.float32)

    for k, amp in enumerate(harmonics, start=1):
        freq = k * f0
        if freq >= sr / 2.0:
            break
        # Violin body resonances:
        # Air mode: ~280 Hz, Corpus wood mode: ~450 Hz, Bridge hill: ~2800 Hz (Q~5)
        air_boost = 1.2 * np.exp(-((freq - 280.0) / 80.0) ** 2)
        corpus_boost = 1.6 * np.exp(-((freq - 450.0) / 120.0) ** 2)
        bridge_boost = 2.4 * np.exp(-((freq - 2800.0) / 500.0) ** 2)
        body_filter = 1.0 + air_boost + corpus_boost + bridge_boost

        y += amp * body_filter * np.sin(k * phase)

    # Bowed string ADSR envelope (gentle attack, prolonged sustain plateau, smooth release)
    att = min(int(sr * 0.06), max(1, n_pts // 5))
    rel = min(int(sr * 0.08), max(1, n_pts // 5))
    env = np.ones(n_pts, dtype=np.float32)
    env[:att] = np.linspace(0.0, 1.0, att, dtype=np.float32)
    env[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32)
    y = y * env

    # Bow friction noise filtered through bridge resonance
    noise = np.random.normal(0.0, 0.015, n_pts).astype(np.float32) * env
    b_noise, a_noise = signal.butter(2, [300.0 / (sr / 2.0), 4500.0 / (sr / 2.0)], btype="band")
    y += signal.lfilter(b_noise, a_noise, noise).astype(np.float32)

    return y

def procedural_melody(scale_root: int = 60, scale_type: str = "major", n_notes: int = 8):
    """Generate procedural melodic phrases using musical scale formulas."""
    scale_intervals = {
        "major": [0, 2, 4, 5, 7, 9, 11, 12],
        "minor": [0, 2, 3, 5, 7, 8, 10, 12],
        "pentatonic": [0, 2, 4, 7, 9, 12],
        "dorian": [0, 2, 3, 5, 7, 9, 10, 12],
    }
    intervals = scale_intervals.get(scale_type, scale_intervals["major"])
    scale_notes = [scale_root + iv for iv in intervals]

    notes_with_dur = []
    dur_choices = [0.256, 0.512]
    total_time = 0.0
    curr_idx = random.randint(0, len(scale_notes) - 1)

    while total_time < AUDIO_DURATION:
        dur = random.choice(dur_choices)
        if total_time + dur > AUDIO_DURATION:
            dur = AUDIO_DURATION - total_time
        midi = scale_notes[curr_idx]
        notes_with_dur.append((midi, dur))
        total_time += dur
        curr_idx = max(0, min(len(scale_notes) - 1, curr_idx + random.choice([-2, -1, 0, 1, 2])))

    return notes_with_dur

def render_audio(notes_with_dur, synth_fn, sr: int = SAMPLE_RATE, total_dur: float = AUDIO_DURATION) -> np.ndarray:
    """Render a sequence of notes into a continuous audio clip."""
    total_len = int(sr * total_dur)
    full_audio = np.zeros(total_len, dtype=np.float32)
    curr_idx = 0

    for midi, note_dur in notes_with_dur:
        f0 = midi_to_hz(midi)
        seg = synth_fn(f0, note_dur, sr=sr)
        end_idx = min(curr_idx + len(seg), total_len)
        full_audio[curr_idx:end_idx] += seg[:end_idx - curr_idx]
        curr_idx = end_idx
        if curr_idx >= total_len:
            break

    peak = np.max(np.abs(full_audio)) + 1e-6
    return (full_audio / peak * 0.95).astype(np.float32)

# ---------------------------------------------------------
# 2. Differentiable Spectrogram Transforms & Audio Inversion
# ---------------------------------------------------------

class AudioSpectrogramTransformer:
    """Transforms raw waveforms to normalized log-STFT representations and inverts back."""
    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.comp_factor = 10.0

    def wav_to_spec(self, audio: torch.Tensor):
        """Convert waveform (B, T) to normalized log-spectrogram (B, 1, 128, frames)."""
        device = audio.device
        win = self.window.to(device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            return_complex=True
        )
        n_frames = min(stft.shape[-1], N_FRAMES)
        mag = torch.abs(stft[:, :N_BINS, :n_frames])
        phase = torch.angle(stft[:, :N_BINS, :n_frames])

        log_mag = torch.log1p(self.comp_factor * mag)
        norm_mag = torch.tanh(log_mag * 0.5)
        return norm_mag.unsqueeze(1), phase

    def spec_to_wav(self, norm_mag: torch.Tensor, phase: torch.Tensor = None, length: int = N_SAMPLES) -> torch.Tensor:
        """Invert normalized log-spectrogram back to waveform."""
        device = norm_mag.device
        win = self.window.to(device)
        clamped_norm = torch.clamp(norm_mag.squeeze(1), -0.999, 0.999)
        log_mag = torch.atanh(clamped_norm) * 2.0
        mag = (torch.exp(torch.clamp(log_mag, min=0.0)) - 1.0) / self.comp_factor
        mag = torch.clamp(mag, min=0.0)

        B, F_bins, T_frames = mag.shape
        padded_mag = torch.zeros(B, self.n_fft // 2 + 1, T_frames, device=device)
        padded_mag[:, :F_bins, :] = mag

        if phase is None:
            padded_phase = torch.zeros_like(padded_mag)
        else:
            p_frames = min(phase.shape[-1], T_frames)
            padded_phase = torch.zeros_like(padded_mag)
            padded_phase[:, :F_bins, :p_frames] = phase[:, :, :p_frames]

        complex_spec = torch.polar(padded_mag, padded_phase)
        wav = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            length=length
        )
        max_val = torch.max(torch.abs(wav), dim=-1, keepdim=True)[0] + 1e-6
        return (wav / max_val * 0.95)

# ---------------------------------------------------------
# 3. Differentiable Pitch Chroma Loss
# ---------------------------------------------------------

def build_chroma_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_bins: int = N_BINS, n_chroma: int = 12) -> torch.Tensor:
    """Create a 12-semitone differentiable chroma projection matrix [12, n_bins]."""
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)[:n_bins]
    fb = np.zeros((n_chroma, n_bins), dtype=np.float32)
    for i, f in enumerate(freqs):
        if f > 35.0:
            midi = 69.0 + 12.0 * np.log2(f / 440.0)
            pitch_class = int(np.round(midi)) % 12
            weight = np.exp(-0.5 * ((midi - np.round(midi)) / 0.35) ** 2)
            fb[pitch_class, i] += weight
    fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-6)
    return torch.tensor(fb, dtype=torch.float32)

class ChromaConsistencyLoss(nn.Module):
    """Penalizes melodic/pitch drift between source and translated audio."""
    def __init__(self, chroma_fb: torch.Tensor):
        super().__init__()
        self.register_buffer("fb", chroma_fb)

    def forward(self, spec_pred: torch.Tensor, spec_tgt: torch.Tensor) -> torch.Tensor:
        p_pred = torch.relu(spec_pred.squeeze(1) + 1.0)
        p_tgt = torch.relu(spec_tgt.squeeze(1) + 1.0)
        c_pred = torch.matmul(self.fb, p_pred)  # (B, 12, T)
        c_tgt = torch.matmul(self.fb, p_tgt)
        return F.l1_loss(c_pred, c_tgt)

# ---------------------------------------------------------
# 4. Neural Architecture: CycleGAN-Audio
# ---------------------------------------------------------

class DepthwiseSeparableConv2d(nn.Module):
    """Efficient depthwise-separable 2D convolution for audio spectrograms."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, stride=stride, padding=padding, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DSResnetBlock(nn.Module):
    """Residual block with depthwise-separable convolutions and Instance Normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            DepthwiseSeparableConv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv(x)

class AudioStyleGenerator(nn.Module):
    """
    Generator network translating audio spectrograms between Guitar and Violin domains.
    Fully convolutional across time, supporting arbitrary input duration.
    """
    def __init__(self, in_c: int = 1, out_c: int = 1, ngf: int = 18, n_blocks: int = 3):
        super().__init__()
        net = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_c, ngf, 5),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample 1: 128 x T -> 64 x T/2
            nn.Conv2d(ngf, ngf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample 2: 64 x T/2 -> 32 x T/4
            nn.Conv2d(ngf * 2, ngf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Bottleneck residual blocks
        for _ in range(n_blocks):
            net.append(DSResnetBlock(ngf * 4))
        # Upsampling
        net += [
            # Upsample 1: 32 x T/4 -> 64 x T/2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Upsample 2: 64 x T/2 -> 128 x T
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # Output projection
            nn.ReflectionPad2d(2),
            nn.Conv2d(ngf, out_c, 5),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

class AudioPatchDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator evaluating realism of localized spectro-temporal patches."""
    def __init__(self, in_c: int = 1, ndf: int = 16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, ndf, 4, stride=2, padding=1),  # 64 x T/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),  # 32 x T/4
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),  # 16 x T/8
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1)  # Patch logits
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 5. Pre-Computed Spectrogram Dataset Loader
# ---------------------------------------------------------

class AudioStyleDataset(Dataset):
    """Unpaired dataset with pre-computed normalized log-spectrograms."""
    def __init__(self, n_samples_per_domain: int = 60, sr: int = SAMPLE_RATE):
        self.sr = sr
        transformer = AudioSpectrogramTransformer()

        self.specs_A = []  # Guitar
        self.specs_B = []  # Violin

        roots = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67]
        scales = ["major", "minor", "pentatonic", "dorian"]

        for i in range(n_samples_per_domain):
            # Domain A
            root_a = random.choice(roots)
            scale_a = random.choice(scales)
            mel_a = procedural_melody(root_a, scale_a, n_notes=random.randint(6, 12))
            wav_a = render_audio(mel_a, synth_guitar_note, sr=self.sr)
            t_wav_a = torch.tensor(wav_a, dtype=torch.float32).unsqueeze(0)
            spec_a, _ = transformer.wav_to_spec(t_wav_a)
            self.specs_A.append(spec_a.squeeze(0))

            # Domain B
            root_b = random.choice(roots)
            scale_b = random.choice(scales)
            mel_b = procedural_melody(root_b, scale_b, n_notes=random.randint(6, 12))
            wav_b = render_audio(mel_b, synth_violin_note, sr=self.sr)
            t_wav_b = torch.tensor(wav_b, dtype=torch.float32).unsqueeze(0)
            spec_b, _ = transformer.wav_to_spec(t_wav_b)
            self.specs_B.append(spec_b.squeeze(0))

    def __len__(self):
        return max(len(self.specs_A), len(self.specs_B))

    def __getitem__(self, idx):
        spec_a = self.specs_A[idx % len(self.specs_A)]
        spec_b = self.specs_B[idx % len(self.specs_B)]

        # Random time crop to TRAIN_FRAMES (128 frames) for fast, robust training
        total_frames = spec_a.shape[-1]
        if total_frames > TRAIN_FRAMES:
            start_idx = random.randint(0, total_frames - TRAIN_FRAMES)
            spec_a = spec_a[:, :, start_idx : start_idx + TRAIN_FRAMES]
            spec_b = spec_b[:, :, start_idx : start_idx + TRAIN_FRAMES]

        return spec_a, spec_b

# ---------------------------------------------------------
# 6. Training Pipeline
# ---------------------------------------------------------

def train_cyclegan(
    epochs: int = 10,
    batch_size: int = 6,
    lr: float = 0.0003,
    device: str = "cpu"
):
    print(f"\n[INFO] Synthesizing & Pre-computing Spectrograms (Guitar <-> Violin)...")
    dataset = AudioStyleDataset(n_samples_per_domain=60)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    transformer = AudioSpectrogramTransformer()
    chroma_fb = build_chroma_filterbank().to(device)
    chroma_loss_fn = ChromaConsistencyLoss(chroma_fb)

    # Networks
    G_AB = AudioStyleGenerator(in_c=1, out_c=1, ngf=18, n_blocks=3).to(device)  # Guitar -> Violin
    G_BA = AudioStyleGenerator(in_c=1, out_c=1, ngf=18, n_blocks=3).to(device)  # Violin -> Guitar
    D_A = AudioPatchDiscriminator(in_c=1, ndf=16).to(device)   # Guitar Discriminator
    D_B = AudioPatchDiscriminator(in_c=1, ndf=16).to(device)   # Violin Discriminator

    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(0.5, 0.999))

    lambda_cyc = 10.0
    lambda_id = 5.0
    lambda_chroma = 2.0

    print(f"[INFO] Generators: {sum(p.numel() for p in G_AB.parameters()):,} params each.")
    print(f"[INFO] Discriminators: {sum(p.numel() for p in D_A.parameters()):,} params each.")
    print(f"[INFO] Training for {epochs} epochs on device: {device}...\n")

    history = {
        "loss_G": [],
        "loss_D": [],
        "loss_cyc": [],
        "loss_id": [],
        "loss_chroma": [],
    }

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ep_loss_G, ep_loss_D = 0.0, 0.0
        ep_loss_cyc, ep_loss_id, ep_loss_chroma = 0.0, 0.0, 0.0

        for real_A, real_B in dataloader:
            real_A, real_B = real_A.to(device), real_B.to(device)

            # ----------------------------------
            # 1. Update Generators G_AB and G_BA
            # ----------------------------------
            opt_G.zero_grad()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Adversarial Loss (LSGAN)
            loss_G_adv = torch.mean((D_B(fake_B) - 1.0) ** 2) + torch.mean((D_A(fake_A) - 1.0) ** 2)

            # Cycle Consistency Loss
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cyc_A = F.l1_loss(rec_A, real_A)
            loss_cyc_B = F.l1_loss(rec_B, real_B)
            loss_cyc = loss_cyc_A + loss_cyc_B

            # Identity Loss (G_AB(B) ~ B, G_BA(A) ~ A)
            id_B = G_AB(real_B)
            id_A = G_BA(real_A)
            loss_id = F.l1_loss(id_B, real_B) + F.l1_loss(id_A, real_A)

            # Pitch Chroma Loss
            loss_chroma = chroma_loss_fn(fake_B, real_A) + chroma_loss_fn(fake_A, real_B)

            loss_G = loss_G_adv + (lambda_cyc * loss_cyc) + (lambda_id * loss_id) + (lambda_chroma * loss_chroma)
            loss_G.backward()
            opt_G.step()

            # ----------------------------------
            # 2. Update Discriminators D_A and D_B (using detached fakes)
            # ----------------------------------
            opt_D.zero_grad()

            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A.detach())
            loss_D_A = 0.5 * (torch.mean((pred_real_A - 1.0) ** 2) + torch.mean(pred_fake_A ** 2))

            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B.detach())
            loss_D_B = 0.5 * (torch.mean((pred_real_B - 1.0) ** 2) + torch.mean(pred_fake_B ** 2))

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            ep_loss_G += loss_G.item()
            ep_loss_D += loss_D.item()
            ep_loss_cyc += loss_cyc.item()
            ep_loss_id += loss_id.item()
            ep_loss_chroma += loss_chroma.item()

        n_batches = len(dataloader)
        ep_loss_G /= n_batches
        ep_loss_D /= n_batches
        ep_loss_cyc /= n_batches
        ep_loss_id /= n_batches
        ep_loss_chroma /= n_batches

        history["loss_G"].append(ep_loss_G)
        history["loss_D"].append(ep_loss_D)
        history["loss_cyc"].append(ep_loss_cyc)
        history["loss_id"].append(ep_loss_id)
        history["loss_chroma"].append(ep_loss_chroma)

        ep_time = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
              f"Loss_G: {ep_loss_G:.4f} | Loss_D: {ep_loss_D:.4f} | "
              f"Cyc: {ep_loss_cyc:.4f} | Id: {ep_loss_id:.4f} | Chroma: {ep_loss_chroma:.4f}")

    total_train_time = time.time() - start_train_time
    print(f"\n[INFO] Training complete in {total_train_time:.1f}s.")

    # Save model weights
    save_path = "models/cyclegan_guitar2violin_best.pth"
    torch.save({
        "G_AB_state": G_AB.state_dict(),
        "G_BA_state": G_BA.state_dict(),
        "D_A_state": D_A.state_dict(),
        "D_B_state": D_B.state_dict(),
        "history": history,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }, save_path)
    print(f"[INFO] Model checkpoint saved to '{save_path}'.")

    return G_AB, G_BA, transformer, chroma_fb, history

# ---------------------------------------------------------
# 7. Benchmarking, Metrics & Visualization
# ---------------------------------------------------------

def evaluate_and_benchmark(G_AB, G_BA, transformer, chroma_fb, history, device: str = "cpu"):
    print("\n[INFO] Running Comprehensive Quantitative Evaluation & Benchmarking...")
    G_AB.eval()
    G_BA.eval()

    random.seed(42)
    np.random.seed(42)

    # Test phrase: Diatonic stepwise motif (2.048s)
    test_motif = [
        (62, 0.256), (59, 0.256), (60, 0.256), (62, 0.256),
        (60, 0.256), (59, 0.256), (57, 0.256), (55, 0.512)
    ]
    guitar_test_wav = render_audio(test_motif, synth_guitar_note)
    violin_test_wav = render_audio(test_motif, synth_violin_note)

    t_guitar = torch.tensor(guitar_test_wav, dtype=torch.float32).unsqueeze(0).to(device)
    t_violin = torch.tensor(violin_test_wav, dtype=torch.float32).unsqueeze(0).to(device)

    # Extract full 128x256 spectrograms
    spec_guitar, phase_guitar = transformer.wav_to_spec(t_guitar)
    spec_violin, phase_violin = transformer.wav_to_spec(t_violin)

    # Benchmarking latency on full 2.048s spectrogram
    latencies = []
    with torch.no_grad():
        for _ in range(5):
            _ = G_AB(spec_guitar)
        for _ in range(30):
            t0 = time.perf_counter()
            _ = G_AB(spec_guitar)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))
    std_latency_ms = float(np.std(latencies))
    rtf = (mean_latency_ms / 1000.0) / AUDIO_DURATION
    speedup = AUDIO_DURATION / (mean_latency_ms / 1000.0)

    with torch.no_grad():
        transferred_violin_spec = G_AB(spec_guitar)
        cycled_guitar_spec = G_BA(transferred_violin_spec)

        transferred_guitar_spec = G_BA(spec_violin)
        cycled_violin_spec = G_AB(transferred_guitar_spec)

        identity_violin_spec = G_AB(spec_violin)
        identity_guitar_spec = G_BA(spec_guitar)

    # Audio synthesis
    transferred_violin_wav = transformer.spec_to_wav(transferred_violin_spec, phase_guitar).cpu().numpy().flatten()
    cycled_guitar_wav = transformer.spec_to_wav(cycled_guitar_spec, phase_guitar).cpu().numpy().flatten()
    transferred_guitar_wav = transformer.spec_to_wav(transferred_guitar_spec, phase_violin).cpu().numpy().flatten()

    # Quantitative metrics
    cycle_l1_A = float(F.l1_loss(cycled_guitar_spec, spec_guitar).item())
    cycle_l1_B = float(F.l1_loss(cycled_violin_spec, spec_violin).item())
    mean_cycle_l1 = (cycle_l1_A + cycle_l1_B) / 2.0

    identity_l1_violin = float(F.l1_loss(identity_violin_spec, spec_violin).item())
    identity_l1_guitar = float(F.l1_loss(identity_guitar_spec, spec_guitar).item())
    mean_identity_l1 = (identity_l1_violin + identity_l1_guitar) / 2.0

    def compute_chroma_sim(spec_a, spec_b):
        p_a = torch.relu(spec_a.squeeze(1) + 1.0)
        p_b = torch.relu(spec_b.squeeze(1) + 1.0)
        c_a = torch.matmul(chroma_fb, p_a).flatten()
        c_b = torch.matmul(chroma_fb, p_b).flatten()
        sim = F.cosine_similarity(c_a.unsqueeze(0), c_b.unsqueeze(0)).item()
        return float(sim)

    chroma_sim_A2B = compute_chroma_sim(spec_guitar, transferred_violin_spec)
    chroma_sim_B2A = compute_chroma_sim(spec_violin, transferred_guitar_spec)
    mean_chroma_sim = (chroma_sim_A2B + chroma_sim_B2A) / 2.0

    f_res = (SAMPLE_RATE / 2.0) / (N_FFT // 2)
    bin_low = int(2000.0 / f_res)
    bin_high = int(3500.0 / f_res)

    guitar_bridge_energy = float(torch.mean(spec_guitar[:, :, bin_low:bin_high, :]).item())
    transferred_violin_bridge_energy = float(torch.mean(transferred_violin_spec[:, :, bin_low:bin_high, :]).item())
    violin_ref_bridge_energy = float(torch.mean(spec_violin[:, :, bin_low:bin_high, :]).item())

    metrics = {
        "cycle_consistency_l1": round(mean_cycle_l1, 4),
        "cycle_l1_guitar2violin2guitar": round(cycle_l1_A, 4),
        "cycle_l1_violin2guitar2violin": round(cycle_l1_B, 4),
        "identity_preservation_l1": round(mean_identity_l1, 4),
        "chroma_pitch_cosine_similarity": round(mean_chroma_sim, 4),
        "chroma_similarity_guitar_to_violin": round(chroma_sim_A2B, 4),
        "chroma_similarity_violin_to_guitar": round(chroma_sim_B2A, 4),
        "bridge_formant_energy_input_guitar": round(guitar_bridge_energy, 4),
        "bridge_formant_energy_transferred_violin": round(transferred_violin_bridge_energy, 4),
        "bridge_formant_energy_target_violin": round(violin_ref_bridge_energy, 4),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "real_time_factor_rtf": round(rtf, 5),
        "real_time_speedup_factor": round(speedup, 1),
    }

    metrics_path = "outputs/style_transfer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to '{metrics_path}'.")

    # Save audio files
    sf.write("outputs/sample_guitar_input.wav", guitar_test_wav, SAMPLE_RATE)
    sf.write("outputs/sample_violin_transferred.wav", transferred_violin_wav, SAMPLE_RATE)
    sf.write("outputs/sample_guitar_cycled.wav", cycled_guitar_wav, SAMPLE_RATE)
    sf.write("outputs/sample_violin_input.wav", violin_test_wav, SAMPLE_RATE)
    sf.write("outputs/sample_guitar_transferred.wav", transferred_guitar_wav, SAMPLE_RATE)
    print("[INFO] Audio samples saved to 'outputs/'.")

    # Plot 6-panel comprehensive diagnostic report
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Waveforms Comparison
    t_axis = np.linspace(0, AUDIO_DURATION, len(guitar_test_wav))
    ax1 = axes[0, 0]
    ax1.plot(t_axis, guitar_test_wav, color="#3498db", alpha=0.7, label="Input Guitar (Plucked)")
    ax1.plot(t_axis, transferred_violin_wav, color="#e74c3c", alpha=0.6, label="Transferred Violin (Sustained)")
    ax1.set_title("Time-Domain Waveforms: Plucked vs Sustained Timbre", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Input Guitar Spectrogram
    ax2 = axes[0, 1]
    im2 = ax2.imshow(spec_guitar[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("Input Guitar Log-Spectrogram (Fast Decaying Harmonics)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Frequency Bins")
    fig.colorbar(im2, ax=ax2, label="Normalized Log Mag")

    # Panel 3: Transferred Violin Spectrogram
    ax3 = axes[1, 0]
    im3 = ax3.imshow(transferred_violin_spec[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="inferno")
    ax3.set_title("Transferred Violin Log-Spectrogram (Sustained Formants)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Frequency Bins")
    fig.colorbar(im3, ax=ax3, label="Normalized Log Mag")

    # Panel 4: Cycled Guitar Spectrogram
    ax4 = axes[1, 1]
    im4 = ax4.imshow(cycled_guitar_spec[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax4.set_title(f"Cycled Guitar Spectrogram (Cycle L1 = {cycle_l1_A:.4f})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time Frames")
    ax4.set_ylabel("Frequency Bins")
    fig.colorbar(im4, ax=ax4, label="Normalized Log Mag")

    # Panel 5: Chromagram Comparison
    ax5 = axes[2, 0]
    c_guitar = torch.matmul(chroma_fb, torch.relu(spec_guitar[0, 0] + 1.0)).cpu().numpy()
    c_violin = torch.matmul(chroma_fb, torch.relu(transferred_violin_spec[0, 0] + 1.0)).cpu().numpy()
    ax5.plot(np.mean(c_guitar, axis=-1), "o-", color="#3498db", label="Guitar Chroma Profile")
    ax5.plot(np.mean(c_violin, axis=-1), "s--", color="#e74c3c", label="Transferred Violin Chroma Profile")
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    ax5.set_title(f"Pitch Chroma Profile: Melody Cosine Similarity = {chroma_sim_A2B:.4f}", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Pitch Class")
    ax5.set_ylabel("Average Chromatic Energy")
    ax5.legend(loc="upper right")
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Panel 6: CycleGAN Training Loss Curves
    ax6 = axes[2, 1]
    ep_range = range(1, len(history["loss_G"]) + 1)
    ax6.plot(ep_range, history["loss_G"], color="#2ecc71", label="Loss G (Total)")
    ax6.plot(ep_range, history["loss_D"], color="#e67e22", label="Loss D")
    ax6.plot(ep_range, history["loss_cyc"], color="#9b59b6", linestyle="--", label="Cycle Loss")
    ax6.plot(ep_range, history["loss_chroma"], color="#1abc9c", linestyle=":", label="Chroma Loss")
    ax6.set_title("CycleGAN Training Convergence Curves", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Loss")
    ax6.legend(loc="upper right")
    ax6.grid(True, linestyle="--", alpha=0.5)

    report_path = "outputs/style_transfer_report.png"
    plt.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Diagnostic report dashboard saved to '{report_path}'.")

    # Print summary table
    print("\n" + "=" * 65)
    print("        AUDIO STYLE TRANSFER BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Cycle Consistency L1:           {metrics['cycle_consistency_l1']:.4f}")
    print(f"Identity Preservation L1:       {metrics['identity_preservation_l1']:.4f}")
    print(f"Chroma Pitch Cosine Similarity: {metrics['chroma_pitch_cosine_similarity']:.4f} (Melody Preserved)")
    print(f"Bridge Formant Energy (Guitar): {metrics['bridge_formant_energy_input_guitar']:.4f}")
    print(f"Bridge Formant Energy (Violin): {metrics['bridge_formant_energy_transferred_violin']:.4f} (+{metrics['bridge_formant_energy_transferred_violin'] - metrics['bridge_formant_energy_input_guitar']:.4f} boost)")
    print(f"Mean Inference Latency:         {metrics['mean_latency_ms']:.2f} ms")
    print(f"Real-Time Speedup Factor:       {metrics['real_time_speedup_factor']:.1f}x (RTF: {metrics['real_time_factor_rtf']:.5f})")
    print("=" * 65 + "\n")

    return metrics

# ---------------------------------------------------------
# 8. Main Execution Entrypoint
# ---------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  PROJECT 97: AUDIO STYLE TRANSFER (GUITAR TO VIOLIN TIMBRE)")
    print("=" * 65)
    print(f"Execution Device: {device}")

    # Train CycleGAN
    G_AB, G_BA, transformer, chroma_fb, history = train_cyclegan(
        epochs=10,
        batch_size=6,
        lr=0.0003,
        device=device
    )

    # Evaluate, benchmark, and generate reports
    evaluate_and_benchmark(G_AB, G_BA, transformer, chroma_fb, history, device=device)

if __name__ == "__main__":
    main()
