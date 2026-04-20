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
from data_scripts.beats.BEATs import BEATs, BEATsConfig

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
        #get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #resampler for audio signals
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=16000
        ).to(self.device)

        #MFCC transform
        n_mfcc = 64
        self.mfcc_transform = torchaudio.transforms.MFCC(
            n_mfcc=n_mfcc,
            sample_rate=44100,
            melkwargs={"n_mels": 128, "win_length": 2048, "hop_length": 512, "n_fft": 2048},
        ).to(self.device)

        #if using model Frechet Distance initialize an dload models
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

            # load BEATs
            beats_path = "./data_scripts/beats/BEATs_iter3_plus_AS2M.pt"
            beats_checkpoint = torch.load(beats_path)
            cfg = BEATsConfig(beats_checkpoint["cfg"])
            self.beats_model = BEATs(cfg)
            self.beats_model.load_state_dict(beats_checkpoint["model"])
            self.beats_model.to("cuda")
            self.beats_model.eval()

        self.reset_embedding_lists()

    def reset_embedding_lists(self): 
        """
        accumulate prediction and target embeddings for Frechet Distance
        over entire validation set. 
        """
        self.FAD_preds = []
        self.FAD_targs = []
        self.FDM_preds = []
        self.FDM_targs = []
        self.FDD_preds = []
        self.FDD_targs = []

    def frechet_distance_over_distribution(self, predictions, targets, eps=1e-6):
        """
        predictions : (seq_len, embed_dim)
        target : (seq_len, embed_dim)
        """
        predictions = predictions.double()
        targets = targets.double()

        mu_p = torch.mean(predictions, dim=0)
        mu_t = torch.mean(targets, dim=0)

        sigma_p = torch.cov(predictions.T)
        sigma_t = torch.cov(targets.T)

        sigma_p += eps * torch.eye(sigma_p.shape[0], dtype=sigma_p.dtype, device=sigma_p.device)
        sigma_t += eps * torch.eye(sigma_t.shape[0], dtype=sigma_t.dtype, device=sigma_t.device)

        distances = frechet_distance(mu_p, sigma_p, mu_t, sigma_t)
        return distances.item()

    #functions to accumulate embeddings for frechet distance
    def store_FAD(self, predictions, targets): 
        """
        predictions (raw signal) : (batch, channel, length) 
        targets (raw signal) : (batch, channel, length)
        """
        #remove channel dimension
        predictions = predictions.squeeze(1)
        targets = targets.squeeze(1)
        batch = predictions.shape[0]

        #resample to 16khz
        pred_resample = self.resampler(predictions).cpu().numpy()
        tar_resample = self.resampler(targets).cpu().numpy()

        #preprocess embeddings for vggish
        pred_mfcc = []
        target_mfcc = []
        for b in range(batch):
            pred_audio = pred_resample[b]
            tar_audio = tar_resample[b]

            pred_mel = waveform_to_examples(pred_audio, 16000)
            tar_mel = waveform_to_examples(tar_audio, 16000)

            pred_mfcc.append(torch.from_numpy(pred_mel).float())
            target_mfcc.append(torch.from_numpy(tar_mel).float())

        #calculate actual VGGish embeddings
        pred_tensor = torch.cat(pred_mfcc, dim=0).to(self.device)
        target_tensor = torch.cat(target_mfcc, dim=0).to(self.device)
        with torch.no_grad():
            pred_embed = self.vggish_model(pred_tensor)
            tar_embed = self.vggish_model(target_tensor)

            self.FAD_preds.append(pred_embed.detach().cpu())
            self.FAD_targs.append(tar_embed.detach().cpu())

    def store_FDD(self, predictions, targets): 
        """
        predictions (raw signal) : (batch, channel, length) 
        targets (raw signal) : (batch, channel, length)
        """

        # preprocess (no need to remove channel due to dac.preprocess)
        pred_signal = self.dac_model.preprocess(predictions, 44100)
        tar_signal = self.dac_model.preprocess(targets, 44100)

        # get embeddings (seem to be unormalized continous embeddings)
        # (batch, seq_len, embed_dim)
        with torch.no_grad(): 
            target_embed = self.dac_model.encode(tar_signal)[0].permute(0, 2, 1)
            generated_embed = self.dac_model.encode(pred_signal)[0].permute(0, 2, 1)

        #flatten to (batch * seq_len, embed_dim)
        target_embed = target_embed.reshape(-1, target_embed.shape[-1])
        generated_embed = generated_embed.reshape(-1, generated_embed.shape[-1])

        #store embeddigns
        self.FDD_preds.append(generated_embed.detach().cpu())
        self.FDD_targs.append(target_embed.detach().cpu())

    def store_FDM(self, predictions, targets): 
        """
        predictions (raw signal) : (batch, channel, length) 
        targets (raw signal) : (batch, channel, length)
        """
        #remove channel
        predictions = predictions.squeeze(1)
        targets = targets.squeeze(1)

        #generate mfcc representation (batch, seq_len, mfcc_dim)
        mfcc_pred = self.mfcc_transform(predictions).permute(0, 2, 1)
        mfcc_tar = self.mfcc_transform(targets).permute(0, 2, 1)

        #(batch * seq_len , mfcc_dim)
        mfcc_pred = mfcc_pred.reshape(-1, mfcc_pred.shape[-1])
        mfcc_tar = mfcc_tar.reshape(-1, mfcc_tar.shape[-1])

        #store representation
        self.FDM_preds.append(mfcc_pred.detach().cpu())
        self.FDM_targs.append(mfcc_tar.detach().cpu())

    #functions to calculate frechet distance over a full validation pass
    def get_epoch_FAD(self): 
        """
        calculate VGGish frechet distance over distribution of entire 
        validation run 
        """
        #(seq_len * N * batch_size, embed_dim), N = num baches
        predictions = torch.cat(self.FAD_preds, dim=0).to(self.device)
        targets = torch.cat(self.FAD_targs, dim=0).to(self.device)

        return self.frechet_distance_over_distribution(predictions, targets)

    def get_epoch_FDD(self): 
        """
        calculate DAC frechet distance over distribution of entire 
        validation run 
        """
        #(seq_len * N * batch_size, embed_dim), N = num baches
        predictions = torch.cat(self.FDD_preds, dim=0).to(self.device)
        targets = torch.cat(self.FDD_targs, dim=0).to(self.device)

        return self.frechet_distance_over_distribution(predictions, targets)

    def get_epoch_FDM(self): 
        """
        calculate MFCC frechet distance over distribution of entire 
        validation run 
        """
        #(seq_len * N * batch_size, embed_dim), N = num baches
        predictions = torch.cat(self.FDM_preds, dim=0).to(self.device)
        targets = torch.cat(self.FDM_targs, dim=0).to(self.device)

        #normalize with z-score normalization
        predictions = (predictions - predictions.mean(dim=0)) / (predictions.std(dim=0) + 1e-6)
        targets = (targets - targets.mean(dim=0)) / (targets.std(dim=0) + 1e-6)

        return self.frechet_distance_over_distribution(predictions, targets)

    ######### THESE METRICS NOT PROPERLY IMPLEMENTED / VERIFIED YET #############

    def wav_avg_cos_sim(self, prediction, target):
        """
        should not be used but retained for legacy purposes
        """
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
        #remove channel dimension
        predictions = predictions.squeeze(1)
        targets = targets.squeeze(1)

        # wav2clip expects 16KHz audio, ours are 44.1KHz
        # resample down to 16khz
        predictions = self.resampler(predictions)
        targets = self.resampler(targets)

        pred_audio = predictions.detach().cpu().numpy()
        tar_audio = targets.detach().cpu().numpy()

        #could require list wrapper around pred_audio and tar_audio
        pred_emb = wav2clip.embed_audio(pred_audio, self.wav2clip_model)
        target_emb = wav2clip.embed_audio(tar_audio, self.wav2clip_model)

        pred_emb = torch.from_numpy(pred_emb)
        target_emb = torch.from_numpy(target_emb)

        #  Torch will automatically L2 normalize
        batch_sims = torch.nn.functional.cosine_similarity(pred_emb, target_emb, dim=-1)

        return batch_sims.mean().item()

    def cycle_clip(self, predictions, targets):
        pass

    def _get_beats_encodings(self, audio):
        audio = audio.squeeze(1)

        with torch.no_grad():
            mask = torch.zeros(audio.shape).bool().to(self.device)
            features = self.beats_model.extract_features(
                audio, padding_mask=mask,
            )[0]

        return features

    def _get_novelty_curve(self, features, kernel, k):
        # normalize
        features_norm = F.normalize(features, dim=-1)

        # compute self-similarity matrix
        ssm = torch.bmm(features_norm, features_norm.transpose(1, 2))
        seq_len = ssm.shape[1]

        # perform 2d conv
        ssm_unsqueeze = ssm.unsqueeze(1)
        ssm_padded = F.pad(ssm_unsqueeze, (k, k, k, k))
        novelty_2d = F.conv2d(ssm_padded, kernel)

        # extract diag
        novelty_2d = novelty_2d.squeeze(1)
        novelty_curve = torch.diagonal(novelty_2d, dim1=1, dim2=2)

        # removing padding artifacts
        return novelty_curve[:, :seq_len]

    def novelty_score(self, predictions, targets):
        beats_pred = self._get_beats_encodings(predictions)
        beats_targ = self._get_beats_encodings(targets)

        # create kernel
        k = 16  # kernel will be 16x16 for now
        L = k // 2
        kernel = torch.ones((k, k), device=self.device)
        kernel[:L, :L] = -1
        kernel[L:, L:] = -1

        # add gaussian tapering
        var = 0.5
        grid = torch.arange(-L, L, dtype=torch.float32, device=self.device) + 0.5
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        gaussian = torch.exp(-(x**2 + y**2) / (2 * (var * L)**2))

        kernel = kernel * gaussian
        kernel = kernel / kernel.abs().sum() # normalize
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        pred_nov = self._get_novelty_curve(beats_pred, kernel, k)
        targ_nov = self._get_novelty_curve(beats_targ, kernel, k)

        # compute pearson while normalizing curves
        pred_centered = pred_nov - pred_nov.mean(dim=-1, keepdim=True)
        targ_centered = targ_nov - targ_nov.mean(dim=-1, keepdim=True)

        cov = (pred_centered * targ_centered).sum(dim=-1)
        std_pred = torch.sqrt((pred_centered**2).sum(dim=-1))
        std_targ = torch.sqrt((targ_centered**2).sum(dim=-1))

        corr = cov / (std_pred * std_targ + 1e-8)  # add epsilon
        return corr.sum()  # we'll divide by the total items at the end.

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
