from torch.utils.data import Dataset
import numpy as np
import torch
import os
from tqdm import tqdm
import random 
import math 
import wav2clip #add metrics dependencies to environment.yml
import torchaudio #add metrics dependencies to environment.yml
from torchaudio.functional import frechet_distance
import torch.nn.functional as F
import sys
import dac

#append VGGish dir
sys.path.append(torch.hub.get_dir() + '/harritaylor_torchvggish_master')
from vggish_input import waveform_to_examples

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

    u = random.uniform(0, math.pi/2)

    shape = dac_encodings.shape
    p = math.cos(u) 
    prob_tensor = torch.full(shape, p)
    mask_tokens = torch.arange(0, K * codebook_size, codebook_size).reshape(1, 1, -1)
    mask = torch.bernoulli(prob_tensor)

    masked_encodings = torch.where(mask == 0, mask_tokens, dac_encodings)

    #one hots
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

class Metrics(): 
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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000).to(self.device)

        #load wav2clip
        self.wav2clip_model = wav2clip.get_model()
        self.wav2clip_model.eval()

        #load vggish
        self.vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish_model.eval()

        #load DAC
        dac_model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(dac_model_path)
        self.dac_model.to("cuda")
        self.dac_model.eval()


    #(batch, seq_len, embed_dim)
    def embed_avg_frechet_distance(self, prediction, target): 
        mu_p = torch.mean(prediction, dim=0)
        mu_t = torch.mean(target, dim=0)
        
        sigma_p = torch.cov(prediction.T)
        sigma_t = torch.cov(target.T)

        distances = frechet_distance(mu_p, sigma_p, mu_t, sigma_t)
        avg_distance = torch.mean(distances, dim=0)
        return avg_distance.item()


    """
    frechet distance VGGish
    assume (batch, len)
    """
    def FAD(self, prediction, target): 

        batch = prediction.shape[0]
        
        pred_resample = self.resampler(prediction)
        tar_resample = self.resampler(target)
        
        pred_batch_embeds = []
        tar_batch_embeds = []
        for b in range(batch): 
            pred_audio = pred_resample[b].numpy()
            tar_audio = tar_resample[b].numpy()

            pred_mel = waveform_to_examples(pred_audio, 16000)
            pred_tensor = torch.from_numpy(pred_mel).float()
            tar_mel = waveform_to_examples(pred_audio, 16000)
            tar_tensor = torch.from_numpy(tar_mel).float()
            
            with torch.no_grad():
                pred_embed = self.vggish_model(pred_tensor) 
                pred_batch_embeds.appennd(pred_embed)
                tar_embed = self.vggish_model(tar_tensor)
                tar_batch_embeds.appennd(tar_embed)
        
        generated_embed = torch.stack(pred_batch_embeds)
        target_embed = torch.stack(tar_batch_embeds)

        return self.embed_avg_frechet_distance(generated_embed, target_embed) 

    """
    frechet distance MFCC
    assume (batch, len)
    """
    def FDM(self, prediction, target): 
        n_mfcc=64
        mfcc_pred = torchaudio.transforms.MFCC(
            n_mfcc=n_mfcc, 
            sample_rate=44100,
            melkwargs={
               'n_mels'=128,
               'win_length'=2048, 
               'hop_length'=512
            }
        )
        mfcc_tar = torchaudio.transforms.MFCC(
            n_mfcc=n_mfcc, 
            sample_rate=44100,
            melkwargs={
               'n_mels'=128,
               'win_length'=2048, 
               'hop_length'=512
            }
        )

        return self.embed_avg_frechet_distance(mfcc_pred, mfcc_tar) 

    """
    frechet distance DAC
    assume (batch, channel, len)
    """
    def FDD(self, prediction, target): 
        #preprocess
        pred_signal = self.dac_model.preprocess(prediction)
        tar_signal = self.dac_model.preprocess(target)

        #get embeddings
        target_embed, _, _ = self.dac_model.encode(pred_signal)
        generated_embed, _, _ = self.dac_model.encode(tar_signal)

        return self.embed_avg_frechet_distance(generated_embed, target_embed) 

    def wav_avg_cos_sim(self, prediction, target): 
        sim = torch.nn.functional.cosine_similarity(target, prediction, dim=1)
        avg_sim = torch.mean(sim, dim=0)

        return avg_sim.item()

    def wave_clip(self, predictions, targets):
        """
        prediction.shape: [batch, samples_rate  * T]
        
        generated audio and target auido
        L2 normalize
        cosine similarity

        return:  tesnor of shape [batch, 1]
        """
        wcs = []
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
            wc = torch.nn.functional.cosine_similarity(pred_emb, target_emb, dim = -1)
            wcs.append(wc)
        return  torch.cat(wcs).reshape(batch, 1)

    def cycle_clip(self, predictions, targets): 
        pass

    def _get_novelty_curve(self, tensor, kernel, k):
        # compute self-similarity matrix
        ssm = torch.matmul(tensor, tensor.transpose(-1, -2)).unsqueeze(1)
        
        res = F.conv2d(
            ssm,
            kernel,
            padding=k//2,
        ).squeeze(1)

        nov = torch.diagonal(res, dim1=1, dim2=2)
        nov = nov - nov.mean(dim=-1, keepdim=True)
        nov = F.normalize(nov, dim=-1)

        return nov


    def novelty_score(self, predictions, targets):
        k = 16 # kernel will be 16x16 for now
        kernel = torch.ones(k, k)
        kernel[:k//2, :k//2] = -1
        kernel[k//2:, k//2:] = -1
        kernel = kernel.unsqueeze(0).unsqueeze(0).to("cuda")

        pred_nov = self._get_novelty_curve(predictions, kernel, k)
        targ_nov = self._get_novelty_curve(targets, kernel, k)

        corr = (pred_nov * targ_nov).sum(dim=-1, keepdim=True)

        return corr.mean()

    """
    Takes in a tensor (batch, channels, len)
        -> channels should be mono (1 channe)
        -> audio is 44.1khz
    """
    def get_metrics(
        self, predictions, targets, clip, 
        avg_cosine_similarity=False,
        FAD=False, 
        FDM=False, 
        FDD=False, 
        wave_clip=False, 
        cycle_clip=False, 
        novelty=False, 
    ): 
        results = dict()
        if avg_cosine_similarity: 
            results['cos'] = self.wav_avg_cos_sim(predictions, targets)
        if FAD: 
            results['FAD'] = self.FAD(predictions, targets)
        if FDM: 
            results['FDM'] = self.FDM(predictions, targets)
        if FDD: 
            results['FDD'] = self.FDD(predictions, targets)
        if wave_clip: 
            results['wave_clip'] = self.wave_clip(predictions, targets)
        if cycle_clip: 
            results['cycle_clip'] = self.cycle_clip(predictions, clip)
        if novelty: 
            results['NS'] = self.novelty_score(predictions, targets)
            
        return results


class AudioVideoDataset(Dataset):
    def __init__(self, root, with_beats=False):
        self.data_paths = []
        self.with_beats = with_beats

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

                self.data_paths.append((audio_file_path, video_file_path))

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


def main():
    audio_encodings = AudioVideoDataset(PREENCODE_DIR)

    dac, beats = audio_encodings.__getitem__(0)
    print(dac.shape, beats.shape)
    print(np.max(dac), np.max(beats))


if __name__ == "__main__":
    main()
