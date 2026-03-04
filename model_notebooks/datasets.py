from torch.utils.data import Dataset
import numpy as np

import os
from tqdm import tqdm


PREENCODE_DIR = (
    "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"
)


class TrainDataset(Dataset):
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


class ValidationDataset(Dataset):
    def __init__(self, root):
        self.data_paths = []

        dirs = os.listdir(root)

        for dir in dirs:
            audio_encode_pth = os.path.join(root, dir, "audio_encode")

            if not os.path.exists(audio_encode_pth):
                continue
            names = os.listdir(audio_encode_pth)

            for name in tqdm(names):
                audio_file_path = os.path.join(audio_encode_pth, name)

                self.data_paths.append(audio_file_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        audio_file_path = self.data_paths[index]
        audio_encoding = np.load(audio_file_path)

        return audio_encoding["dac"]


def main():
    audio_encodings = TrainDataset(PREENCODE_DIR)

    dac, beats = audio_encodings.__getitem__(0)
    print(dac.shape, beats.shape)
    print(np.max(dac), np.max(beats))


if __name__ == "__main__":
    main()
