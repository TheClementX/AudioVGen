"""
A simple script to turn mp4 to silent audio and video with a target folder
"""

import os
import subprocess 
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed 

TARGET_DIR = '/home/clem3nti/data/vggsound_data/VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
MAX_WORKERS = os.cpu_count()

"""
splits a video into its respective silent video
and audio. Crops the contents of all videos to exactly
the same length. 

This routine is run concurrently 
"""
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

"""
processes all video files using multithreading to increase speed
"""
def process_all(target_dir): 
    dirs = os.listdir(target_dir)
    tasks = []
    print(f"subdirs to be processed {dirs}")

    #iterate through each shard dir
    print("scanning directories and collecting splitting tasks")
    iter = tqdm(dirs, desc="Scannig Files")
    for dir in iter: 
        parent = os.path.join(target_dir, dir)

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

        with tqdm(total = len(tasks), desc="Processing files") as pbar:
            for future in as_completed(futures):
                if future.result(): 
                    successes+=1
                else: 
                    fails+=1

                pbar.set_postfix(succeded=successes, failed=fails)
                pbar.update(1)

    print("finished processing data")

if __name__ == "__main__":
    process_all(TARGET_DIR)

