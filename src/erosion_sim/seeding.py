"""Random streams. Each is seeded from a public key by hashing its canonical JSON with SHA-256, so the
same key gives the same stream in any process, on any device, in any order."""

import hashlib
import json

import torch


def step_generator(seed: int, step: int, device) -> torch.Generator:
    """Random stream for one training step. It depends only on the seed and the step, so a resumed run
    draws the same noise as an uninterrupted one."""
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + step)


def noise_seed(*key) -> int:
    """A 64-bit seed from a public key such as ``(seed, pair, sample)``, by hashing its canonical JSON."""
    payload = json.dumps(["erosion-sim-diffusion-v1", *key], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
