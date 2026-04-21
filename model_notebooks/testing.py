import torch
import torchaudio
import soundfile as sf

from tqdm import tqdm
from datasets import inference_mask, Metrics

@torch.no_grad()
def valid_epoch(
    model,
    ema_model,
    dac_model, 
    dataloader,
    device,
    metrics: Metrics,
    steps=15,
    codebook_size=1024,
    distributed=False,
    save_folder="./videos",
):
    torch.cuda.empty_cache()

    ema_model.eval()
    model.eval()

    progress = tqdm(
        dataloader,
        desc="Val (Cls)",
        dynamic_ncols=True,
        leave=False,
        mininterval=2.0
    )

    #reset frechet distance accumulators
    metrics.reset_embedding_lists()
    novelty_score = 0.0
    wave_clip = 0.0
    cycle_clip = 0.0
    total_items = 0

    for encodings in progress:

        #unpack encoding
        dac_encoding, clip_encoding, s3d_encoding = encodings

        #change device
        dac_encoding = dac_encoding.to(device)
        clip_encoding = clip_encoding.to(device)
        s3d_encoding = s3d_encoding.to(device)

        #apply masking
        masked_encodings = torch.full(dac_encoding.shape, codebook_size, device=dac_encoding.device)
        omega = 3.0
        #inference forward pass
        for step in range(steps):
            # Forward pass (inference-only)
            # outputs = model(masked_encodings, clip_encoding, s3d_encoding)
            # masked_encodings = inference_mask(outputs, step, steps)

            with torch.amp.autocast(device_type='cuda'):
                outputs_cond = ema_model(masked_encodings, clip_encoding, s3d_encoding)
                output_uncond = ema_model(masked_encodings, torch.zeros_like(clip_encoding), torch.zeros_like(s3d_encoding))
                outputs = outputs_cond + omega * (outputs_cond - output_uncond)

            del outputs_cond, output_uncond
            masked_encodings = inference_mask(outputs, step, steps)


        ######## Get/update model metrics ##########
        #frechet distance metrics
        if distributed:
            predictions = model.module.decode(dac_model, masked_encodings)
            targets = model.module.decode(dac_model, dac_encoding)
        else:
            predictions = model.decode(dac_model, masked_encodings)
            targets = model.decode(dac_model, dac_encoding)
        
        # save each predicted audio
        batch_size = predictions.shape[0]
        for i in range(batch_size):
            sf.write(f"{save_folder}/{total_items}.wav", predictions[i].numpy(force=True).T, 44100)
            # torchaudio.save(f"{save_folder}/{total_items}.wav", predictions[i], 44100)
            total_items += 1
            if i == 0:
                sf.write(f"{save_folder}/{total_items}_true.wav", targets[i].numpy(force=True).T, 44100)
                print(torch.max(predictions[i]))
                print(torch.min(predictions[i]))
                print(torch.max)

        #update frechet distance embedding accumulators
        if metrics.model_metrics: 
            metrics.store_FAD(predictions, targets)
            metrics.store_FDD(predictions, targets)
            novelty_score += metrics.novelty_score(predictions, targets)
            wave_clip += metrics.wave_clip(predictions, targets)
            cycle_clip += metrics.cycle_clip(predictions, clip_encoding)

        metrics.store_FDM(predictions, targets)

        # Progress bar update
        # progress.set_postfix(
        #     distance=f"{batch_fdm:.4f} ({distance_meter.avg:.4f})",
        #     # waveclip=f"{batch_cos:.2f}% ({waveclip_meter.avg:.2f}%)",
        # )

    #get epoch frechet distances
    FDD, FAD = None, None
    if metrics.model_metrics: 
        FDD = metrics.get_epoch_FDD()
        FAD = metrics.get_epoch_FAD()

        if total_items != 0:
            novelty_score /= total_items
            wave_clip /= total_items
            cycle_clip /= total_items
            print("novelty_score:", novelty_score)
            print("wave_clip:", wave_clip)
            print("cycle_clip:", cycle_clip)

    FDM = metrics.get_epoch_FDM()

    #return only frechet distances
    return FDM, FDD, FAD