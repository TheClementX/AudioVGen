"""
This script shards the shard_dir in order
to seperate the data into manageable chunks
"""

import os 
from tqdm import tqdm 

SHARD_DIR = '/home/clem3nti/data/vggsound_data/VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'

def shard(): 
    print("beginning sharding ...")
    count = 0
    with os.scandir(SHARD_DIR) as entries:
        for entry in tqdm(entries): 
            if entry.is_file(): 
                filename = entry.name
                fc = filename[0]

                subfolder = os.path.join(SHARD_DIR, fc)
                os.makedirs(subfolder, exist_ok=True)

                old_path = entry.path
                new_path = os.path.join(subfolder, filename)

                os.rename(old_path, new_path)
                count += 1
    print(f"sharding finished, {count} files moved")

if __name__ == "__main__":
    shard()
