HOME_PATH = "." # path where you cloned musicfm

import os
import sys
import torch
from tqdm import tqdm

sys.path.append(HOME_PATH)
from musicfm.model.musicfm_25hz import MusicFM25Hz

# load MusicFM
musicfm = MusicFM25Hz(
    is_flash=True,
    stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
    model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
)
musicfm = musicfm.to("mps")
musicfm.eval()

# dummy audio (30 seconds, 24kHz)
for i in tqdm(range(10000)):
    wav = (torch.rand(4, 24000 * 30) - 0.5) * 2

    # # to GPUs
    wav = wav.to("mps")

    # get embeddings
    emb = musicfm.get_latent(wav, layer_ix=7)

    torch.save(emb, f"emb.pt")