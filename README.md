# DVC_with_Scaled_Hierarchical_Bi_directiona_Motion_Model

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
