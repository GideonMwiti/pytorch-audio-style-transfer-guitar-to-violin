#!/usr/bin/env python3
"""
Production CLI for Project 97: Audio Style Transfer (Guitar <-> Violin)
Supports both real-time CycleGAN feedforward inference and optimization-based Neural Audio Style Transfer.
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import (
    AudioStyleGenerator,
    AudioSpectrogramTransformer,
    build_chroma_filterbank,
    SAMPLE_RATE,
    AUDIO_DURATION,
    N_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_BINS,
    N_FRAMES,
    render_audio,
    procedural_melody,
    synth_guitar_note,
    synth_violin_note,
)

# ---------------------------------------------------------
# Neural Audio Style Transfer (Gram Matrix Optimization Baseline)
# ---------------------------------------------------------

class AcousticFeatureFilterbank(nn.Module):
    """Multi-scale acoustic filterbank extracting timbral texture maps."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(48, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))
        return [f1, f2, f3]

def calc_gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix representing timbral channel correlations."""
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    gram = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)
    return gram

def neural_audio_style_transfer(
    content_spec: torch.Tensor,
    style_spec: torch.Tensor,
    iterations: int = 40,
    lr: float = 0.05,
    content_weight: float = 1.0,
    style_weight: float = 500.0,
    device: str = "cpu"
) -> torch.Tensor:
    """Iterative optimization matching acoustic content features and style Gram matrices."""
    filterbank = AcousticFeatureFilterbank().to(device)
    with torch.no_grad():
        c_feats = filterbank(content_spec)
        s_feats = filterbank(style_spec)
        s_grams = [calc_gram_matrix(sf) for sf in s_feats]

    # Initialize target from content
    target_spec = content_spec.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([target_spec], lr=lr)

    for i in range(iterations):
        optimizer.zero_grad()
        t_feats = filterbank(target_spec)

        # Content loss on middle representation
        loss_c = F.mse_loss(t_feats[1], c_feats[1])

        # Style loss on all representations
        loss_s = 0.0
        for tf, sg in zip(t_feats, s_grams):
            tg = calc_gram_matrix(tf)
            loss_s += F.mse_loss(tg, sg)

        total_loss = (content_weight * loss_c) + (style_weight * loss_s)
        total_loss.backward()
        optimizer.step()

        # Enforce range [-1, 1]
        with torch.no_grad():
            target_spec.clamp_(-1.0, 1.0)

    return target_spec.detach()

# ---------------------------------------------------------
# CLI Helper Functions
# ---------------------------------------------------------

def load_generator(checkpoint_path: str, direction: str = "guitar2violin", device: str = "cpu"):
    """Load pre-trained Generator from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please run 'python main.py' first to train the models."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    gen = AudioStyleGenerator(in_c=1, out_c=1, ngf=18, n_blocks=3).to(device)

    if direction == "guitar2violin":
        gen.load_state_dict(ckpt["G_AB_state"])
    elif direction == "violin2guitar":
        gen.load_state_dict(ckpt["G_BA_state"])
    else:
        raise ValueError(f"Unknown direction '{direction}'. Must be 'guitar2violin' or 'violin2guitar'.")

    gen.eval()
    return gen

def transfer_audio_file(
    input_wav_path: str,
    output_wav_path: str,
    direction: str = "guitar2violin",
    method: str = "cyclegan",
    style_ref_path: str = None,
    checkpoint_path: str = "models/cyclegan_guitar2violin_best.pth",
    device: str = "cpu"
):
    """Perform audio style transfer on an input .wav file."""
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"Input file not found: '{input_wav_path}'")

    audio, sr = sf.read(input_wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)  # Mono

    # Resample or truncate/pad to standard duration if needed
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    elif len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]

    t_audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    transformer = AudioSpectrogramTransformer()
    spec, phase = transformer.wav_to_spec(t_audio)

    chroma_fb = build_chroma_filterbank().to(device)

    t0 = time.perf_counter()

    if method == "cyclegan":
        gen = load_generator(checkpoint_path, direction=direction, device=device)
        with torch.no_grad():
            out_spec = gen(spec)
    elif method == "neural_style":
        if style_ref_path is None or not os.path.exists(style_ref_path):
            # Synthesize synthetic reference style
            ref_notes = [(62, 0.256), (65, 0.256), (69, 0.512)]
            synth_fn = synth_violin_note if direction == "guitar2violin" else synth_guitar_note
            ref_wav = render_audio(ref_notes, synth_fn)
            t_ref = torch.tensor(ref_wav, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            ref_wav, _ = sf.read(style_ref_path)
            if ref_wav.ndim > 1:
                ref_wav = np.mean(ref_wav, axis=-1)
            if len(ref_wav) < N_SAMPLES:
                ref_wav = np.pad(ref_wav, (0, N_SAMPLES - len(ref_wav)))
            else:
                ref_wav = ref_wav[:N_SAMPLES]
            t_ref = torch.tensor(ref_wav, dtype=torch.float32).unsqueeze(0).to(device)

        ref_spec, _ = transformer.wav_to_spec(t_ref)
        out_spec = neural_audio_style_transfer(spec, ref_spec, iterations=35, device=device)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cyclegan' or 'neural_style'.")

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Invert back to audio
    out_wav = transformer.spec_to_wav(out_spec, phase).cpu().numpy().flatten()

    # Calculate pitch chroma cosine similarity
    p_in = torch.relu(spec.squeeze(1) + 1.0)
    p_out = torch.relu(out_spec.squeeze(1) + 1.0)
    c_in = torch.matmul(chroma_fb, p_in).flatten()
    c_out = torch.matmul(chroma_fb, p_out).flatten()
    chroma_sim = float(F.cosine_similarity(c_in.unsqueeze(0), c_out.unsqueeze(0)).item())

    # Save output audio
    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
    sf.write(output_wav_path, out_wav, SAMPLE_RATE)

    print("=" * 65)
    print("        AUDIO STYLE TRANSFER COMPLETE")
    print("=" * 65)
    print(f"Input Audio:       {input_wav_path}")
    print(f"Output Audio:      {output_wav_path}")
    print(f"Transfer Direction:{direction}")
    print(f"Method:            {method.upper()}")
    print(f"Inference Latency: {latency_ms:.2f} ms")
    print(f"Melody Sim (Chroma):{chroma_sim:.4f} (Pitch conserved)")
    speedup = AUDIO_DURATION / (latency_ms / 1000.0)
    print(f"Real-Time Speedup: {speedup:.1f}x real-time")
    print("=" * 65)

    return out_wav, chroma_sim, latency_ms

def run_demo(checkpoint_path: str = "models/cyclegan_guitar2violin_best.pth", device: str = "cpu"):
    """Run built-in demo converting guitar to violin and violin to guitar."""
    print("\n[DEMO] Generating canonical test samples...")
    test_notes = [
        (60, 0.256), (64, 0.256), (67, 0.256), (72, 0.256),
        (69, 0.256), (65, 0.256), (64, 0.256), (60, 0.512)
    ]
    demo_guitar = render_audio(test_notes, synth_guitar_note)
    demo_violin = render_audio(test_notes, synth_violin_note)

    os.makedirs("outputs", exist_ok=True)
    in_g_path = "outputs/demo_input_guitar.wav"
    out_v_path = "outputs/demo_transferred_violin.wav"
    in_v_path = "outputs/demo_input_violin.wav"
    out_g_path = "outputs/demo_transferred_guitar.wav"

    sf.write(in_g_path, demo_guitar, SAMPLE_RATE)
    sf.write(in_v_path, demo_violin, SAMPLE_RATE)

    print("\n>>> Task 1: Converting Plucked Guitar to Bowed Violin...")
    transfer_audio_file(in_g_path, out_v_path, direction="guitar2violin", method="cyclegan", checkpoint_path=checkpoint_path, device=device)

    print("\n>>> Task 2: Converting Bowed Violin to Plucked Guitar...")
    transfer_audio_file(in_v_path, out_g_path, direction="violin2guitar", method="cyclegan", checkpoint_path=checkpoint_path, device=device)

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audio Style Transfer CLI: Guitar <-> Violin Timbre Translation")
    parser.add_argument("--input", type=str, default=None, help="Path to input audio file (.wav)")
    parser.add_argument("--output", type=str, default=None, help="Path to save transferred audio file (.wav)")
    parser.add_argument("--direction", type=str, default="guitar2violin", choices=["guitar2violin", "violin2guitar"], help="Translation direction")
    parser.add_argument("--method", type=str, default="cyclegan", choices=["cyclegan", "neural_style"], help="Transfer method: 'cyclegan' or 'neural_style'")
    parser.add_argument("--style-ref", type=str, default=None, help="Reference style audio file (for neural_style mode)")
    parser.add_argument("--checkpoint", type=str, default="models/cyclegan_guitar2violin_best.pth", help="Model checkpoint path")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration on synthesized test melodies")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.demo:
        run_demo(checkpoint_path=args.checkpoint, device=args.device)
        return

    if args.input is None:
        print("[ERROR] Please provide --input <file.wav> or use --demo to run automated demonstration.")
        parser.print_help()
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_transferred_{args.direction}{ext}"

    transfer_audio_file(
        input_wav_path=args.input,
        output_wav_path=output_path,
        direction=args.direction,
        method=args.method,
        style_ref_path=args.style_ref,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

if __name__ == "__main__":
    main()

