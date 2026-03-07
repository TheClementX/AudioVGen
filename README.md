# AudioVGen
## TODO
### Already done
* data processing scripts
* adalnzero model created
* datasets created 
* training / inference mask functions created
* metrics implemented
* training notebook adaln_zero implemented
* psc applied and accepted
* training pipeline created

### Todo / Timeline (near future)
* wait on psc do give resources
* test all code with psc resources
* run dummy training run
* update documentation
* brain storm research objectives

## DOCUMENTATION: 
### Data 
* Our dataset (VGGSound): https://huggingface.co/datasets/Loie/VGGSound/tree/main
* Data Layout: our data follows a sharded data structure where every video is split into seperate files based on the first letter in its filename. These directories are further split nto audio and silent video folders. 

### Data Processing Scripts
* shard.py (legacy)
    * shards the target directory into subdirectories based on a filename prefix
* data_prep.py (legacy)
    * splits videos into audio and silent video
* vggsound_install.py (legacy)
    * downloads the specified dataset from hugging face
* untar.py (legacy)
    * untars all downloaded tar files from hugging face
* download_preprocess.py 
    * downloads, untars, shards, preprocesses, encodes audio, encodes video
    * simply select which self contained functions you want to use in main and set
      usage flags

### Model

### Training



