# Multi-Temporal Ensemble for Few-Shot Action Recognition(MTE)



## Overview
![overview](overall.png)

## Content 
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Acknowledgments](#Acknowledgments)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) >= 1.8
- tensorboardX
- pprint
- tqdm
- dotmap
- yaml
- csv

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).


## Data Preparation
We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.
We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [SthV2](https://developer.qualcomm.com/software/ai-datasets/something-something) ,[UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). 

## Training
We provided several examples to train SloshNet with this repo:
- To train on  Kinetics or SthV2 or Hmdb51 or UCF101 from Imagenet pretrained models, you can run:
```
# train Kinetics
 bash ./scripts/train_kin-1s.sh 

# train SthV2
 bash ./scripts/train_ssv2-1s.sh 

# train HMDB
bash ./scripts/train_hmdb-1s.sh 

# train UCF
bash ./scripts/train_ucf-1s.sh 
 ```
## Testing
To test the trained models, you can run `scripts/run_test.sh`. For example:
```
bash ./scripts/test.sh
```

## Acknowledgments
Our code is based on [TRX](https://github.com/tobyperrett/trx).
]()
