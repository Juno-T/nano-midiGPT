import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, ImageClip

import torch
import miditok
import muspy

from nn.remi_torch.model.nanoGPT import GPT

def no_prompt_generation(out_dir, model: GPT=None, tokenizer: miditok.REMI=None, max_len=100, seed = -1):
    if seed > 0:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
    else:
        seed = np.random.randint(1e6)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
    out_dir = Path(out_dir)
    generation = None
    if not Path(out_dir / "midi.midi").exists():
        assert model is not None and tokenizer is not None
        out_dir.mkdir(exist_ok=True, parents=True)
        model.eval()
        generation_tok = model.generate(idx=torch.LongTensor([[tokenizer['BOS_None']]]), max_new_tokens=max_len)
        model.train()
        generation = tokenizer(generation_tok)
        generation.dump(out_dir / "midi.midi")
    music = muspy.read_midi(out_dir / "midi.midi")
    muspy.write_audio(out_dir / "audio.wav", music)
    piano_roll = music.show_pianoroll(preset="frame")
    plt.savefig(out_dir / "piano_roll.png")
    score = music.show_score(figsize=(16,9))
    plt.savefig(out_dir / "score.png")
    return out_dir / "midi.midi", music



def add_static_image_to_audio(generation_path):
    generation_path = Path(generation_path)
    image_path, audio_path, output_path = generation_path / "score.png", generation_path / "audio.wav", generation_path / "clip.mp4"
    
    assert image_path.exists(), f"{image_path} does not exist"
    assert audio_path.exists(), f"{audio_path} does not exist"

    audio_clip = AudioFileClip(str(audio_path))
    image_clip = ImageClip(str(image_path))

    video_clip = image_clip.set_audio(audio_clip)
    video_clip.duration = audio_clip.duration
    video_clip.fps = 30
    video_clip.write_videofile(str(output_path), audio_codec="aac")

add_static_image_to_audio("./generation/02")