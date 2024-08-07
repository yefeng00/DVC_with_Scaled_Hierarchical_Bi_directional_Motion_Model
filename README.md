# DVC_with_Scaled_Hierarchical_Bi_directional_Motion_Model

## Installation

1. (Option) Create a virtual environment and activate it.
```
   conda create -n ENV_NAME python=3.7
   conda activate ENV_NAME
```

2. Install required package. 
```
   pip install -r requirements.txt
```

## Tests
### Prepare the test datasets
Format of video data. 
   ```
    DATA_Root
    │   video1.yuv
    │   video2.yuv
    │   ...
   ```

### Get trained model
Downloaded from: https://drive.google.com/drive/folders/1qHRbCriy224rna7FfveVg06oinKxOJ7l

### Testing 
1. Change the `src` in `test.py`(Dir of test datasets).
2. Put pretrain models into ./ckpts folder
2. `Run test.py`:

## Paper
Our work has been accept by ACM MM 2024: Feng Ye, Li Zhang, and Chuanmin Jia. 2024. Deep Video Compression with Scaled Hierarchical Bi-directional Motion Model. In Proceedings of the 32nd ACM International Conference on Multimedia (MM ’24), October 28-November 1, 2024, Melbourne, VIC, Australia. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3664647.3685524
