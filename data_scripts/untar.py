"""
this script untars a set of tar files using the parameters specified at the top of the 
script
"""

import os
import tarfile
from pathlib import Path

SOURCE_DIR = './VGGSound_tar'
DEST_DIR = './VGGSound_raw_data'
#the number of files you want to use 
NUM_FILES = 2

def untar(tar): 
    try:
        with tarfile.open(tar, "r:gz") as t: 
            t.extractall(path=DEST_DIR)
        print(f"Tar File {tar} extracted succesfully")

    except tarfile.ReadError:
        print(f"-> ReadError reading tarfile: {tar}: ReadError")
    except Exception as e: 
        print(f"-> Exception reading tarfile: {tar}: {e}")


def untar_all(): 
    #make destination folder
    os.makedirs(DEST_DIR, exist_ok=True)

    #list tar files
    tar_files = list(Path(SOURCE_DIR).glob("*.tar.gz"))
    
    if not tar_files: 
        print("error finding tarfiles")
        return -1

    
    print(f"{len(tar_files)} Tar files found... beginning extraction")

    if NUM_FILES > len(tar_files):
        return 

    for i in range(NUM_FILES): 
        tar = tar_files[i]
        untar(tar)
        
    print("full extraction finished")

    return len(tar_files)

if __name__ == '__main__': 
    untar_all()
