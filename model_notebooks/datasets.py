from torch.utils.data import Dataset
import numpy as np
import torch
import os
from tqdm import tqdm
import random
import math
import wav2clip


PREENCODE_DIR = (
    "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"
)


def training_mask(dac_encodings, codebook_size):
    """
    K layers => K unique masks, 1 per layer
    1. draw from U(0, pi/2)
    2. p = cos(u)
    3. mask ~ bernoulli(p)
    """
    K = dac_encodings.shape[2]

    u = random.uniform(0, math.pi / 2)

    shape = dac_encodings.shape
    p = math.cos(u)
    prob_tensor = torch.full(shape, p)
    mask_tokens = torch.arange(0, K * codebook_size, codebook_size).reshape(1, 1, -1)
    mask = torch.bernoulli(prob_tensor)

    masked_encodings = torch.where(mask == 0, mask_tokens, dac_encodings)

    # one hots
    targets = torch.where(mask == 0, dac_encodings, 0)
    targets = torch.nn.functional.one_hot(targets, num_classes=codebook_size)

    return masked_encodings, targets


def inference_mask(predictions, cur_step, max_steps):
    """
    predictions : (batch, seq_len, K, codebook_size)

    cos(cur_step/max_steps * math.pi/2)
    """
    _, _, K, codebook_size = predictions.shape
    c = (cur_step + 1) / max_steps
    cur_percentage = 1 - math.cos(c * math.pi / 2)

    # (batch, seq_len, K)
    tokens = torch.argmax(predictions, dim=3)
    confidences = torch.max(predictions, dim=3)
    mask_tokens = torch.arange(0, K * codebook_size, codebook_size).reshape(1, 1, -1)

    # (batch, seq_len * K)
    flat_confidences = torch.flatten(confidences, start_dim=1)
    k = math.ceil(flat_confidences.shape[1] * cur_percentage)
    _, top_k_indices = torch.topk(flat_confidences, k, dim=1)

    mask = torch.zeros(flat_confidences.shape)
    mask.scatter_(1, top_k_indices, 1).reshape(tokens.shape)

    unmasked = torch.where(mask == 1, tokens, mask_tokens)

    return unmasked


class Metrics:
    """
    Fréchet distance
        -> FAD : VGGIsh embedding
        -> FDM : MFCC
        -> FDD : DAC
    Cosine Similarity
        -> target vs generated
    Semantic Matching:
        -> Wav2Clip
        -> embed target embed generated then cosine similarity
        -> embed generated and compare against original CLIP embddings
    Alignment of generated audio:
        -> novelity score: pearson coefficient between BEATS(target) and BEATS(generated)
        -> SparseSync: mean of offset (not used)
    """

    """
    frechet distance VGGish
    """

    def FAD():
        pass

    """
    frechet distance MFCC
    """

    def FDM():
        pass

    """
    frechet distance DAC
    """

    def FDD():
        pass

    def wav_cos_sim():
        pass

    def wave_clip(prediction, target):

        wav2clip_model = wav2clip.get_model()
        emb_preds

    def cycle_clip():
        pass

    def novelty_score():
        pass

    def get_metrics(predictions, targets):
        pass


class AudioVideoDataset(Dataset):
    def __init__(self, data_paths, with_beats=False):
        self.data_paths = data_paths
        self.with_beats = with_beats

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        audio_file_path, video_file_path = self.data_paths[index]
        audio_encoding = np.load(audio_file_path)
        video_encoding = np.load(video_file_path)

        ret_list = [
            audio_encoding["dac"],
            video_encoding["clip"],
            video_encoding["s3d"],
        ]
        if self.with_beats:
            ret_list.append(audio_encoding["beats"])

        return ret_list


def get_datasets(root, validation_ratio=0.05, with_beats=False):
    data_paths = []

    dirs = os.listdir(root)
    for dir in dirs:
        audio_encode_pth = os.path.join(root, dir, "audio_encode")
        video_encode_pth = os.path.join(root, dir, "video_encode")

        if not os.path.exists(audio_encode_pth):
            continue
        names = os.listdir(audio_encode_pth)

        for name in tqdm(names):
            audio_file_path = os.path.join(audio_encode_pth, name)
            video_file_path = os.path.join(video_encode_pth, name)

            data_paths.append((audio_file_path, video_file_path))

    num_validation = math.ceil(len(data_paths) * validation_ratio)
    valid_dataset = AudioVideoDataset(data_paths[:num_validation], with_beats)
    train_dataset = AudioVideoDataset(data_paths[num_validation:])

    return train_dataset, valid_dataset


def main():
    audio_encodings = AudioVideoDataset(PREENCODE_DIR)

    dac, beats = audio_encodings.__getitem__(0)
    print(dac.shape, beats.shape)
    print(np.max(dac), np.max(beats))


if __name__ == "__main__":
    main()
