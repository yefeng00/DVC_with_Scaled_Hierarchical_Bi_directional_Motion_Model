import os

names = ["BQTerrace_1920x1080_60", "BasketballDrive_1920x1080_50", "Cactus_1920x1080_50",
        "Kimono_1920x1080_24","ParkScene_1920x1080_24","FourPeople_1280x720_60",
        "Johnny_1280x720_60","KristenAndSara_1280x720_60","BQMall_832x480_60",
        "BasketballDrill_832x480_50","PartyScene_832x480_50","RaceHorsesC_832x480_30",
        "BasketballPass_416x240_50","BlowingBubbles_416x240_50","BQSquare_416x240_60",
        "RaceHorses_416x240_30"]

widths = [1920, 1920, 1920, 1920, 1920, 
          1280, 1280, 1280, 
          832, 832, 832, 832, 
          416, 416, 416, 416]

heights = [1080, 1080, 1080, 1080, 1080, 
           720, 720, 720, 
           480, 480, 480, 480, 
           240, 240, 240, 240]


for idx in range(0,16):
    width = widths[idx]
    height = heights[idx]
    for i in ['256','512','1024','2048']:
        os.system('python3 eval.py --metric mse --gop_size 16 --lambda_i 3 --intra_period 16 \
          --pretrain ckpts/' + i + '.pth \
          --eval_lambda ' + i + ' --src /data/test/' + names[idx] +  \
          ' --seq_name ' + names[idx] + ' --num_frames 97 \
          --width ' + str(width) + ' --height ' + str(height) )
        #src: dir of test sequences
