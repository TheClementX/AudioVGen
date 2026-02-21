# AudioVGen

I. Already done
    a. data processing scripts created. 

II. Todo / Timeline (near future)
    a. select a maskvat architecture to implement 
    b. after selection determine what input needs to look like
    c. make pytorch datasets for data
    d. implement model 

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
    e. data_comp.py
        -> (MUST BE MADE) combines the top 4 scripts into one for comprehensive datadownload 
        and preproccessing


