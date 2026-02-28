"""
Description: 
    This is a script to fully download and 
    preproccess the VGGSound dataset from hugging face
Usage: 
    Set the download flags and run the script 

Hugging Face Data Repo: 
    https://huggingface.co/datasets/Loie/VGGSound

Dependencies: 
    -> all dependencies handled in environment.yml
    -> there are sometimes issues with ffmpeg and conda

"""

#imports
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
import os
import tarfile
from pathlib import Path
from tqdm import tqdm 
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed 

"""
DOWNLOADING AND PREPROCESSING FLAGS
    -> downloading flags
    -> untaring flags
    -> sharding flags
    -> data_prep flags
"""
#download flags
FULL_DOWNLOAD = False #download all of the dataset
SELECT_DOWNLOAD = True #download a select amount of data
TAR_DOWNLOAD_DIR = 'VGGSound_tar'
TAR_FILES = []

if FULL_DOWNLOAD: 
    TAR_FILES = [i for i in range(0, 20)]
elif SELECT_DOWNLOAD: 
    #set files you want to download as integers in [0, 20]
    TAR_FILES = [0]

#untar flags
TAR_SOURCE_DIR = './VGGSound_tar'
TAR_DEST_DIR = './VGGSound_raw_data'

#shard flags
#shard dir should be the dir where you downloaded VGGSound
SHARD_DIR = './VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'

#data preprocess flags
#process dir should be the dir where you downloaded VGGSound and sharded
PROCESS_DIR = './VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
MAX_WORKERS = os.cpu_count()

"""
data processing functions
"""
def generate_tar_names(): 
    names = []
    for fidx in TAR_FILES: 
        if fidx < 10: 
            name = 'vggsound_0' + str(fidx) + '.tar.gz'
            names.append(name)
        else: 
            name = 'vggsound_' + str(fidx) + '.tar.gz'
            names.append(name)

    return names 

def download_data():
    if FULL_DOWNLOAD: 
        print("Starting full download of VGGSound... this may take a while.")
        snapshot_download(
            repo_id="Loie/VGGSound", 
            repo_type="dataset",
            local_dir=TAR_DOWNLOAD_DIR, 
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Full Download complete!")

    #download single file
    if SELECT_DOWNLOAD: 
        print("Starting select download of VGGSound... this may take a while.")
        tar_names = generate_tar_names()
        for name in tar_names: 
            print(f'downloading {name}')
            hf_hub_download(
                repo_id="Loie/VGGSound", 
                repo_type="dataset",
                filename=name,
                local_dir=TAR_DOWNLOAD_DIR, 
                local_dir_use_symlinks=False,
                force_download=True
            )
        print("Select Download complete!")

    return 1

def untar(tar): 
    try:
        with tarfile.open(tar, "r:gz") as t: 
            t.extractall(path=TAR_DEST_DIR)
        print(f"Tar File {tar} extracted succesfully")

    except tarfile.ReadError:
        print(f"-> ReadError reading tarfile: {tar}: ReadError")
    except Exception as e: 
        print(f"-> Exception reading tarfile: {tar}: {e}")

def untar_data(): 
    #make destination folder
    os.makedirs(TAR_DEST_DIR, exist_ok=True)

    #list tar files
    tar_files = list(Path(TAR_SOURCE_DIR).glob("*.tar.gz"))
    
    if not tar_files: 
        print("error finding tarfiles")
        return -1

    
    print(f"{len(tar_files)} Tar files found... beginning extraction")

    for tar in tar_files: 
        untar(tar)
        
    print("full extraction finished")

    return len(tar_files)

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

def process_video(file, fname, video_path, audio_path): 
    audio_out = os.path.join(audio_path, fname + '.wav')
    audio_command = [
        'ffmpeg', '-i', file, '-vn', 
        '-t', '10', 
        '-acodec', 'pcm_s16le', 
        '-ar', '44100',
        '-y', '-loglevel', 'error',
        audio_out
    ]

    video_out = os.path.join(video_path, fname + '.mp4')
    video_command = [
        'ffmpeg', '-i', file, '-an', 
        '-t', '10',
        '-vcodec', 'copy', 
        '-y', '-loglevel', 'error',
        video_out
    ]

    try: 
        subprocess.run(video_command, check=True)
        subprocess.run(audio_command, check=True)

        os.remove(file)
        return True
    
    except subprocess.CalledProcessError as e: 
        print(f"error processing {file}, Error: {e}")
        return False

def prep_data(): 
    print("preprocessing data")
    dirs = os.listdir(PROCESS_DIR)
    tasks = []
    print(f"subdirs to be processed {dirs}")

    #iterate through each shard dir
    print("scanning directories and collecting splitting tasks")
    iter = tqdm(dirs, desc="Scannig Files", leave=False)
    for dir_name in iter: 
        parent = os.path.join(PROCESS_DIR, dir_name)

        if not os.path.isdir(parent): 
            continue

        audio_path = os.path.join(parent, "audio")
        video_path = os.path.join(parent, "video")

        os.makedirs(audio_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)
        
        #iterate through each file in the shard dir
        #and split it into audio and silent video
        with os.scandir(parent) as entries:
            for entry in entries: 
                if entry.is_file() and entry.name.endswith('.mp4'): 
                    filename = entry.name
                    fname, ext = os.path.splitext(filename)

                    tasks.append((entry.path, fname, video_path, audio_path))

    print(f"total tasks found {len(tasks)}")
    if(len(tasks) == 0):
        print("no tasks found")
        return 

    #process all found videos
    successes=0
    fails=0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: 
        #dictionary comprehension future : data
        futures = {executor.submit(process_video, *task): task for task in tasks}

        with tqdm(total = len(tasks), desc="Processing files", leave=False) as pbar:
            for future in as_completed(futures):
                if future.result(): 
                    successes+=1
                else: 
                    fails+=1

                pbar.set_postfix(succeded=successes, failed=fails)
                pbar.update(1)

    print("finished processing data")

#link all commands for processing
def main(): 
#     download_data()
    untar_data()
    shard()
    prep_data()

if __name__ == "__main__": 
    main()

