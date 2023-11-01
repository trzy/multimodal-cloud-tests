# Multimodal Model Cloud Performance Test

## Overview

Performance tests of multimodal models on cloud providers.

## LLaVA on Runpod

### Configuration

[Runpod](https://runpod.io) has generally good availability but is more expensive than Amazon.

[LLaVA](https://github.com/haotian-liu/LLaVA) (the PyTorch repo) was tested using the `llava-test.py` script to connect to a worker process. The dataset used was `llava-v1.5-7b` [here](https://huggingface.co/liuhaotian/llava-v1.5-7b).

**Note:** Very important that the model files be in directories that are named with `llava-v1.5-` in order for the conversation mode to be picked up correctly.

Commands to run on the server (in this order) to bring up the required controller process (redundant but the worker expects to find it) and worker after ensuring
all Python dependencies are installed:

```
    python -m llava.serve.controller --host 0.0.0.0 --port 10000
    python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 8080 --worker http://localhost:8080 --model-path models/llava-v1.5-7b

```

Then, copy all the images in this repo's `images` directory along with `llava-test.py` to the LLaVA repo root directory and run:

```
    python llava-test.py
```

### Results

GPU utilization (arbitrary snapshot):

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.56.06    Driver Version: 520.56.06    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100 80G...  On   | 00000000:23:00.0 Off |                    0 |
| N/A   46C    P0   153W / 300W |  16576MiB / 81920MiB |     60%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
 CPU and memory (48 vCPU system):

 ```
 Tasks:  20 total,   1 running,  19 sleeping,   0 stopped,   0 zombie
%Cpu(s): 19.5 us,  2.0 sy,  0.0 ni, 78.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem : 1031909.+total, 146472.5 free,  89076.6 used, 796360.7 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used. 927013.3 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                                                                                                                                                           
  14413 root      20   0   45.2g   6.5g 989636 S 133.6   0.6  66:42.70 python                                                                                                                                                                            
  17484 root      20   0   12.5g 479900 220888 S   0.7   0.0   0:19.10 python                                                                                                                                                                            
      1 root      20   0    1104      4      0 S   0.0   0.0   0:00.15 docker-init                                                                                                                                                                       
      7 root      20   0    4352   3288   2984 S   0.0   0.0   0:00.00 start.sh                                                                                                                                                                          
     40 root      20   0   10324   1088      0 S   0.0   0.0   0:00.00 nginx                                                                                                                                                                             
     41 nobody    20   0   11184   3208   1272 S   0.0   0.0   0:00.00 nginx                                                                                                                                                                             
     54 root      20   0   15412   5472   3848 S   0.0   0.0   0:00.01 sshd                                                                                                                                                                              
     55 root      20   0    4468   1892   1584 S   0.0   0.0   0:00.00 start.sh                                                                                                                                                                          
     60 root      20   0  168440  80768  19344 S   0.0   0.0   0:00.87 jupyter-lab                                                                                                                                                                       
     61 root      20   0    2780   1020    928 S   0.0   0.0   0:00.00 sleep                                                                                                                                                                             
    188 root      20   0   16660  10516   8420 S   0.0   0.0   0:00.38 sshd                                                                                                                                                                              
    213 root      20   0    5036   3964   3360 S   0.0   0.0   0:00.01 bash                                                                                                                                                                              
    640 root      20   0   12.5g 491632 226124 S   0.0   0.0   0:21.21 python                                                                                                                                                                            
  14313 root      20   0   16660  10484   8404 S   0.0   0.0   0:00.32 sshd                                                                                                                                                                              
  14324 root      20   0    5036   4000   3400 S   0.0   0.0   0:00.00 bash                                                                                                                                                                              
  14766 root      20   0   16660  10484   8400 S   0.0   0.0   0:00.18 sshd                                                                                                                                                                              
  14777 root      20   0    5036   4024   3408 S   0.0   0.0   0:00.02 bash                                                                                                                                                                              
  18456 root      20   0   16660  10488   8404 S   0.0   0.0   0:00.03 sshd                                                                                                                                                                              
  18468 root      20   0    5036   4044   3444 S   0.0   0.0   0:00.01 bash                                                                                                                                                                              
  19389 root      20   0    7856   3512   2916 R   0.0   0.0   0:00.03 top                                                                                                                                                                               
 ```

 ```
 Timing Results:
  Mean   = 2.93 s, 453.72 chars/s
  Median = 2.45 s, 229.60 char/s
  1%     = 0.16 s, 93.81 chars/s
  5%     = 0.17 s, 107.65 chars/s
  10%    = 0.19 s, 128.73 chars/s
  90%    = 6.41 s, 1253.40 chars/s
  95%    = 9.11 s, 1397.22 chars/s
  99%    = 13.43 s, 1461.03 chars/s
  ```

## LLaVA via llama.cpp on Runpod

### Configuration

To be consistent with LLaVA, the f16 weights were used from [here](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main) for `llava-v1.5-7b`.

llama.cpp was built with CUDA enabled and run like so:

```
    ./server --mmproj models/llava-1.5-7b/mmproj-model-f16.gguf -m models/llava-1.5-7b/ggml-model-f16.gguf -ngl 99 -t 8
```

`-ngl 99` moves all of the weight layers to GPU (there are far fewer than 99) and `-t 8` uses 8 threads explicitly in an attempt to make CLIP encoding faster,
which is currently CPU-bound on x64 and thus very slow (slower than an M1 MacBook -- 50 ms vs 1 ms per image patch) at the time of this writing.


### Results

GPU utilization is very spikey. Oscillates between ~50% and ~95%:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.56.06    Driver Version: 520.56.06    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100 80G...  On   | 00000000:23:00.0 Off |                    0 |
| N/A   54C    P0   246W / 300W |  14789MiB / 81920MiB |     95%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

CPU and memory:

```
top - 22:08:02 up 303 days,  2:22,  0 users,  load average: 16.06, 16.29, 20.27
Tasks:  17 total,   2 running,  15 sleeping,   0 stopped,   0 zombie
%Cpu(s):  3.5 us,  1.3 sy,  0.0 ni, 95.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem : 1031909.+total, 143527.1 free,  95183.5 used, 793199.2 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used. 919675.6 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                                                                                                                       
  22678 root      20   0   49.8g  16.6g  14.0g R 257.5   1.6  19:48.74 server                                                                                                                                        
  24265 root      20   0    7856   3588   2996 R   0.3   0.0   0:00.04 top                                                                                                                                           
      1 root      20   0    1104      4      0 S   0.0   0.0   0:00.25 docker-init                                                                                                                                   
      7 root      20   0    4352   3288   2984 S   0.0   0.0   0:00.00 start.sh                                                                                                                                      
     40 root      20   0   10324   1088      0 S   0.0   0.0   0:00.00 nginx                                                                                                                                         
     41 nobody    20   0   11184   3208   1272 S   0.0   0.0   0:00.00 nginx                                                                                                                                         
     54 root      20   0   15412   5472   3848 S   0.0   0.0   0:00.02 sshd                                                                                                                                          
     55 root      20   0    4468   1892   1584 S   0.0   0.0   0:00.00 start.sh                                                                                                                                      
     60 root      20   0  168440  80768  19344 S   0.0   0.0   0:00.87 jupyter-lab                                                                                                                                   
     61 root      20   0    2780   1020    928 S   0.0   0.0   0:00.00 sleep                                                                                                                                         
    188 root      20   0   16660  10516   8420 S   0.0   0.0   0:00.95 sshd                                                                                                                                          
    213 root      20   0    5036   4048   3424 S   0.0   0.0   0:00.02 bash                                                                                                                                          
  14313 root      20   0   16660  10484   8404 S   0.0   0.0   0:00.70 sshd                                                                                                                                          
  14324 root      20   0    5036   4008   3400 S   0.0   0.0   0:00.01 bash                                                                                                                                          
  14766 root      20   0   16660  10484   8400 S   0.0   0.0   0:00.47 sshd                                                                                                                                          
  14777 root      20   0    5036   4040   3408 S   0.0   0.0   0:00.04 bash                                                                                                                                          
  23501 root      20   0   30988  24304  10008 S   0.0   0.0   0:00.44 pytho
  ```