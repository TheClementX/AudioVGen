"""
download script to download VGGSound from hugging face. 
"""

from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

#set flags for desired download type 
FULL_DOWNLOAD = False
SINGLE_DOWNLOAD = False

#This will download the dataset to a folder named "vggsound_data" in the current directory
if FULL_DOWNLOAD: 
    print("Starting full download... this may take a while.")
    snapshot_download(
        repo_id="Loie/VGGSound", 
        repo_type="dataset",
        local_dir="vggsound_data",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Full Download complete!")

#download single file
if SINGLE_DOWNLOAD: 
    print("Starting single download... this may take a while.")
    hf_hub_download(
        repo_id="Loie/VGGSound", 
        repo_type="dataset",
        filename="vggsound_08.tar.gz",
        local_dir="vggsound_data",
        local_dir_use_symlinks=False,
        force_download=True
    )
    print("Single Download complete!")
