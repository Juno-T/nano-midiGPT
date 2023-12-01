from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from pathlib import Path

def tokenize(data_path, tokenizer_save_path):
    config = TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False)
    tokenizer = REMI(config)

    root_path = data_path.parent
    dir_name = data_path.stem
    save_path_nobpe = root_path / f"{dir_name}_tokens_noBPE"
    save_path_bpe = data_path / f"{dir_name}_tokens_BPE"
    midi_paths = list(data_path.glob("**/*.midi"))
    midi_paths += list(data_path.glob("**/*.mid"))
    data_augmentation_offsets = [1, 1, 0]
    tokenizer.tokenize_midi_dataset(midi_paths, save_path_nobpe,
                                    data_augment_offsets=data_augmentation_offsets)

    tokenizer.save_params(tokenizer_save_path)
