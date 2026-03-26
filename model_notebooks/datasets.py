from torch.utils.data import Dataset
import numpy as np
import torch
import os
from tqdm import tqdm
import random
import math
import wav2clip  # add metrics dependencies to environment.yml
import torchaudio  # add metrics dependencies to environment.yml
from torchaudio.functional import frechet_distance
import torch.nn.functional as F
import sys
import dac
from vggish.vggish_input import waveform_to_examples

PREENCODE_DIR = (
    "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"
)


#batched
def training_mask(dac_encodings, codebook_size):
    """
    (batch, seq_len, layers) = (batch, 862, 9)
    K layers => K unique mask tokens, 1 per layer
    1. draw from U(0, pi/2)
    2. p = cos(u)
    3. mask ~ bernoulli(p)
    4. Probability is computed batchwise not per forward pass
    could impact training
    """
    dac_device = dac_encodings.device
    batch, seq_len, K = dac_encodings.shape

    eps = 1e-8
    #generate unique probability for each batch
    u = torch.rand((batch, 1, 1), device=dac_device) * (math.pi / 2 - 2 * eps) + eps
    p = torch.cos(u)

    #generate mask tensor
    prob_tensor = p.expand(-1, seq_len, 1)
    mask_tokens = codebook_size
    mask = torch.bernoulli(prob_tensor)
    #duplicate masks across K layers
    mask = mask.expand(-1, -1, K)

    # mask encodings for forward pass
    masked_encodings = torch.where(mask == 0, mask_tokens, dac_encodings)

    # make targets where -100 indicates ignoring the mask position
    # -100 is the default ignore token of CrossEntropyLoss
    targets = torch.where(mask == 0, dac_encodings, -100)

    return masked_encodings, targets


def inference_mask(predictions, cur_step, max_steps, temperature=1.0):
    """
    takes raw predictions
        -> predictions : (batch, seq_len, K, codebook_size)
        -> cos(cur_step/max_steps * math.pi/2)
    """
    batch, seq_len, K, codebook_size = predictions.shape

    #determine what percentage of sequence to unmask
    c = (cur_step + 1) / max_steps
    cur_percentage = 1 - math.cos(c * math.pi / 2)

    probs = F.softmax(predictions / temperature, dim=-1)
    #average probabilities over the layers 

    #sample tokens using a mutlinomial distribution
    flat_probs = probs.reshape(-1, codebook_size)
    tokens = torch.multinomial(flat_probs, num_samples=1)
    tokens = tokens.reshape(batch, seq_len, K)

    # (batch, seq_len)
    confidences = torch.max(probs, dim=-1).values
    temporal_confidences = torch.mean(confidences, dim=-1)

    #get topk positions temporally
    k = math.ceil(temporal_confidences.shape[1] * cur_percentage)
    _, top_k_indices = torch.topk(temporal_confidences, k, dim=1)

    mask = torch.zeros_like(temporal_confidences)
    mask.scatter_(1, top_k_indices, 1)

    mask = mask.unsqueeze(-1).expand(-1, -1, K)
    mask_tokens = codebook_size

    unmasked = torch.where(mask.bool(), tokens, mask_tokens)

    return unmasked


class Metrics:
    """
    Fréchet Inceptio Distance
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

    def __init__(self, model_metrics=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=16000
        ).to(self.device)

        n_mfcc = 64
        self.mfcc_transform = torchaudio.transforms.MFCC(
            n_mfcc=n_mfcc,
            sample_rate=44100,
            melkwargs={"n_mels": 128, "win_length": 2048, "hop_length": 512, "n_fft": 2048},
        ).to(self.device)

        self.model_metrics = model_metrics
        if self.model_metrics:
            # load wav2clip
            self.wav2clip_model = wav2clip.get_model()
            self.wav2clip_model.eval()

            # load vggish
            self.vggish_model = torch.hub.load("harritaylor/torchvggish", "vggish", preprocess=False)
            self.vggish_model.eval()

            # load DAC
            dac_model_path = dac.utils.download(model_type="44khz")
            self.dac_model = dac.DAC.load(dac_model_path)
            self.dac_model.to("cuda")
            self.dac_model.eval()

    # (batch, seq_len, embed_dim)
    # could be numerically unstable
    def embed_avg_frechet_distance(self, prediction, target, eps=1e-6):
        # (batch * seq_len, embed_dim) basically one long sequence
        predictions = prediction.reshape(-1, prediction.shape[-1]).double()
        targets = target.reshape(-1, target.shape[-1]).double()

        mu_p = torch.mean(predictions, dim=0)
        mu_t = torch.mean(targets, dim=0)

        sigma_p = torch.cov(predictions.T)
        sigma_t = torch.cov(targets.T)

        sigma_p += eps * torch.eye(sigma_p.shape[0], dtype=sigma_p.dtype, device=sigma_p.device)
        sigma_t += eps * torch.eye(sigma_t.shape[0], dtype=sigma_t.dtype, device=sigma_t.device)

        distances = frechet_distance(mu_p, sigma_p, mu_t, sigma_t)
        return distances.item()

    """
    frechet distance VGGish
    assume (batch, 1, audio_len)
    """

    def FAD(self, prediction, target):

        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        batch = prediction.shape[0]

        pred_resample = self.resampler(prediction)
        tar_resample = self.resampler(target)

        pred_batch_embeds = []
        tar_batch_embeds = []
        for b in range(batch):
            pred_audio = pred_resample[b].cpu().numpy()
            tar_audio = tar_resample[b].cpu().numpy()

            pred_mel = waveform_to_examples(pred_audio, 16000)
            pred_tensor = torch.from_numpy(pred_mel).float()
            tar_mel = waveform_to_examples(tar_audio, 16000)
            tar_tensor = torch.from_numpy(tar_mel).float()

            with torch.no_grad():
                pred_embed = self.vggish_model(pred_tensor)
                pred_batch_embeds.append(pred_embed)
                tar_embed = self.vggish_model(tar_tensor)
                tar_batch_embeds.append(tar_embed)

        generated_embed = torch.concat(pred_batch_embeds, dim=0).unsqueeze(0)
        target_embed = torch.concat(tar_batch_embeds, dim=0).unsqueeze(0)

        return self.embed_avg_frechet_distance(generated_embed, target_embed)

    """
    frechet distance MFCC
    assume (batch, 1, audio_len)
    """

    def FDM(self, prediction, target):
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        mfcc_pred = self.mfcc_transform(prediction).squeeze(1).permute(0, 2, 1)
        mfcc_tar = self.mfcc_transform(target).squeeze(1).permute(0, 2, 1)

        return self.embed_avg_frechet_distance(mfcc_pred, mfcc_tar)

    """
    frechet distance DAC
    assume (batch, 1, audio_len)
    """

    def FDD(self, prediction, target):
        # preprocess
        pred_signal = self.dac_model.preprocess(prediction, 44100)
        tar_signal = self.dac_model.preprocess(target, 44100)

        # get embeddings (seem to be unormalized continous embeddings)
        target_embed = self.dac_model.encode(tar_signal)[0]
        generated_embed = self.dac_model.encode(pred_signal)[0]

        target_embed = target_embed.permute(0, 2, 1)
        generated_embed = generated_embed.permute(0, 2, 1)

        return self.embed_avg_frechet_distance(generated_embed, target_embed)

    def wav_avg_cos_sim(self, prediction, target):
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        sim = torch.nn.functional.cosine_similarity(target, prediction, dim=1)
        avg_sim = torch.mean(sim, dim=0)

        return avg_sim.item()

    def wave_clip(self, predictions, targets):
        """
        prediction.shape: [batch, channels, samples_rate  * T]

        generated audio and target auido
        L2 normalize
        cosine similarity

        return:  average wave clip score for batch
        """
        wcs = []
        predictions = predictions.squeeze(1)
        targets = targets.squeeze(1)
        batch, _ = predictions.shape
        # Load model
        wav2clip_model = self.wav2clip_model

        # wav2clip expects 16KHz audio, ours are 44.1KHz

        predictions = self.resampler(predictions)
        targets = self.resampler(targets)
        # detach and change it to numpy
        for B in range(batch):
            pred = predictions[B].detach().cpu().numpy().flatten()
            target = targets[B].detach().cpu().numpy().flatten()
            # wav2clip expects 16KHz audio, ours are 44.1KHz

            pred_emb = wav2clip.embed_audio(pred, wav2clip_model)
            target_emb = wav2clip.embed_audio(target, wav2clip_model)
            pred_emb = torch.from_numpy(pred_emb)
            target_emb = torch.from_numpy(target_emb)
            #  Torch will automatically L2 normalize
            wc = torch.nn.functional.cosine_similarity(pred_emb, target_emb, dim=-1)
            wcs.append(wc)

        return torch.mean(torch.cat(wcs).reshape(batch, 1))

    def cycle_clip(self, predictions, targets):
        pass

    def _get_novelty_curve(self, tensor, kernel, k):
        # compute self-similarity matrix
        ssm = torch.matmul(tensor, tensor.transpose(-1, -2)).unsqueeze(1)

        res = F.conv2d(
            ssm,
            kernel,
            padding=k // 2,
        ).squeeze(1)

        nov = torch.diagonal(res, dim1=1, dim2=2)
        nov = nov - nov.mean(dim=-1, keepdim=True)
        nov = F.normalize(nov, dim=-1)

        return nov

    def novelty_score(self, predictions, targets):
        # TODO: convert waveform into BEATs encoding here
        k = 16  # kernel will be 16x16 for now
        kernel = torch.ones(k, k)
        kernel[: k // 2, : k // 2] = -1
        kernel[k // 2 :, k // 2 :] = -1
        kernel = kernel.unsqueeze(0).unsqueeze(0).to("cuda")

        pred_nov = self._get_novelty_curve(predictions, kernel, k)
        targ_nov = self._get_novelty_curve(targets, kernel, k)

        corr = (pred_nov * targ_nov).sum(dim=-1, keepdim=True)

        return corr.mean()

    """
    Takes in a tensor (batch, 1, audio_len)
        -> audio should be mono
        -> audio is 44.1khz
    """

    def get_metrics(
        self,
        predictions,
        targets,
        clip,
        avg_cosine_similarity=False,
        FAD=False,
        FDM=False,
        FDD=False,
        wave_clip=False,
        cycle_clip=False,
        novelty=False,
    ):
    """
    predictions, targets : raw waveforms, embeddings are calculated on the fly
    """

        results = dict()
        """
        -> cos similarity is bad due to phase issues on raw audio.
        -> Use cos similarity on embeddings if anything
        -> make frechet distance computed batchwise instead of example wise
        """
        if avg_cosine_similarity:
            results["cos"] = self.wav_avg_cos_sim(predictions, targets)
        if FAD and self.model_metrics:
            results["FAD"] = self.FAD(predictions, targets)
        if FDM:
            results["FDM"] = self.FDM(predictions, targets)
        if FDD and self.model_metrics:
            results["FDD"] = self.FDD(predictions, targets)
        if wave_clip and self.model_metrics:
            results["wave_clip"] = self.wave_clip(predictions, targets)
        if cycle_clip and self.model_metrics:
            results["cycle_clip"] = self.cycle_clip(predictions, clip)
        if novelty and self.model_metrics:
            results["NS"] = self.novelty_score(predictions, targets)

        return results


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

        # dac = audio_encoding["dac"]
        # clip = video_encoding["clip"]
        # s3d = video_encoding["s3d"]

        # if np.isnan(dac).any() or np.isnan(clip).any() or np.isnan(s3d).any():
        #     print(audio_file_path, video_file_path)

        # if np.isinf(dac).any() or np.isinf(clip).any() or np.isinf(s3d).any():
        #     print(audio_file_path, video_file_path)

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

            if os.path.exists(audio_file_path) and os.path.exists(video_file_path):
                data_paths.append((audio_file_path, video_file_path))

    num_validation = math.ceil(len(data_paths) * validation_ratio)
    valid_dataset = AudioVideoDataset(data_paths[:num_validation], with_beats)
    train_dataset = AudioVideoDataset(data_paths[num_validation:])

    return train_dataset, valid_dataset

def verify_datasets(root): 
    train_d, valid_d =  get_datasets(root, validation_ratio=0.0, with_beats=False)

    prev_dac, prev_clip, prev_s3d = None, None, None
    for i, encoding in tqdm(enumerate(train_d)): 
        dac, clip, s3d = encoding 
        if i == 0: 
            prev_dac, prev_clip, prev_s3d = dac.shape, clip.shape, s3d.shape
            print(f'dac_shape : {prev_dac} clip_shape : {prev_clip} s3d_shape : {prev_s3d}')

        else: 
            #check dac
            if dac.shape != prev_dac: 
                print(f"DAC anomaly prev_shape = {prev_dac}, cur_shape = {dac.shape}")
            else: 
                prev_dac = dac.shape
            #check clip

            if clip.shape != prev_clip: 
                print(f"CLIP anomaly prev_shape = {prev_clip}, cur_shape = {clip.shape}")
            else:
                prev_clip = clip.shape

            #check s3d
            if s3d.shape != prev_s3d: 
                print(f"S3D anomaly prev_shape = {prev_s3d}, cur_shape = {s3d.shape}")
            else:
                prev_s3d = s3d.shape

#verify dataset
if __name__ == "__main__": 
    verify_datasets(PREENCODE_DIR)
