# AudioVGen

I. Already done
    a. data processing scripts created. 

II. Todo / Timeline (near future)
  implement data loaders
  finish pre-encodidng script in download_preprocess.py
  implement maskvat_adaln

DOCUMENTATION: 
I. Data 
    a. Our dataset (VGGSound): https://huggingface.co/datasets/Loie/VGGSound/tree/main
    b. Data Layout: our data follows a sharded data structure where every video is split into 
    seperate files based on the first letter in its filename. These directories are further split 
    into audio and silent video folders. 

II. data processing scripts
    a. shard.py
        -> shards the target directory into subdirectories based on a filename prefix
    b. data_prep.py
        -> splits videos into audio and silent video
    c. vggsound_install.py
        -> downloads the specified dataset from hugging face
    d. untar.py
        -> untars all downloaded tar files from hugging face
    e. download_preprocess.py
        -> does everything all in one
        -> includes option to pre-encode wav files (need to implement
    e. data_comp.py
        -> (MUST BE MADE) combines the top 4 scripts into one for comprehensive datadownload 
        and preproccessing


