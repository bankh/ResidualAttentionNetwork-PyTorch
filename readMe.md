### Notes/ Updates on the Repository
- There are a few bugs in the code corrected for the system- and libraries-under-interest.
- Documentation.
- Added requirements.txt.
- Model saving, Logging and early-stopping are added into train.py.

# ResidualAttentionNetwork-PyTorch
A pytorch code about [Residual Attention Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf).    

This code is based on two  projects from [@liudaizong](https://github.com/liudaizong/Residual-Attention-Network) and [@fwang91](https://github.com/fwang91/residual-attention-network/blob/master/imagenet_model/Attention-92-deploy.prototxt).  
I also added the ResidualAttentionModel_92 for training imagenet, ResidualAttentionModel_448input for larger image input, and ResidualAttentionModel_92_32input_update for training cifar10.  

# Referenced Paper
```
Wang, F., Jiang, M., Qian, C., Yang, S., Li, C., Zhang, H., ... & Tang, X. (2017). Residual attention network for image classification.
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3156-3164).
```

# Setup 
The following setup is specifically for the [target hardware -- (a) in the 3.1 Hardware](https://github.com/bankh/GPU_Compute#31-hardware). Based on the hardware that one might have the setup might need to change.  

- Pull and run Docker container (see [Docker instructions for ROCm](https://github.com/bankh/GPU_Compute/blob/main/Docker_images/AMD/readMe.md)).  
```
$ docker pull rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
$ docker run -it --name caxton_1 \
                 --cap-add=SYS_PTRACE \
                 --security-opt seccomp=unconfined \
                 --device=/dev/kfd --device=/dev/dri \
                 --group-add $(getent group video | cut -d':' -f 3) \
                 --ipc=host \
                 -v /mnt/data_drive:/mnt/data_drive -v /mnt/data:/mnt/data -v /home/ubuntu:/mnt/ubuntu  # Depending on your folder structure\
                 -p 0.0.0.0:6007:6007 \
                 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
                 rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
```

- Inside the docker container, download and install Miniconda.  
```
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ## Check https://repo.anaconda.com/miniconda/ for other Python versions
$ chmod +x Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

- Create and activate the virtual environment.  
```
$ conda create --name ran python=3.8 -y
$ conda activate ran
```

- Install the requirements (changed from the original one without torch and torchvision).  

```
$ pip install -r requirements.txt
```


# Training

__Remark:__  The variable `is_train` should be `True`.   
- Start training by  
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
- For data augmentation, one can use:  
```
CUDA_VISIBLE_DEVICES=0 python train_mixup.py #(with mixup) 
```

- One can train on ResidualAttentionModel_56 or ResidualAttentionModel_448input by modifying the code in train.py as:  
```
from model.residual_attention_network import ResidualAttentionModel_92 as ResidualAttentionModel
```
to  
```
from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
```
__Note:__ Download will start automatically from http://www.cs.toronto.edu/~kriz/cifar.html.

# Testing

__Remark:__ The variable `is_train` should be `False`.
- Start testing by:  
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
- For data augmentation, one can use:
```
CUDA_VISIBLE_DEVICES=0 python train_mixup.py #(with mixup) 
```
# Results
1. cifar-10: Acc-95.4(Top-1 err 4.6) with ResidualAttentionModel_92_32input_update(higher than paper top-1 err 4.99)  
2. cifar-10: Acc-96.65(Top-1 err 3.35) with ResidualAttentionModel_92_32input_update(with mixup).  
3. cifar-10: Acc-96.84(Top-1 err 3.16) with ResidualAttentionModel_92_32input_update(with mixup, with simpler attention module).  

## Acknowledgement 
Thanks to @PistonY, who gave me the advice on mixup.  
For more details on mixup you can reference the project https://github.com/facebookresearch/mixup-cifar10  

The paper only gives the architectural details of attention_92 for Imagenet with 224 input --not for CIFAR10.  

# Supporting Information:
- model_92_sgd.pkl is the trained model file, an accuracy of 0.954
