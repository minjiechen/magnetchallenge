## Preparation

- Windows11/Ubuntu 22.04 LTS
- Python 3.10.11
- Pytorch 2.0.1
- NVIDIA GPU
- Anaconda
- CUDA 11.7 /CUDA11.8 /CUDA12.1
- Recent GPU driver

```
conda create -n torch2 python=3.10
conda activate torch2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements.txt
```

#### torch is not in requirements.txt, because it need index-url to install with cuda-toolkit.

#### In some environments, the torch installed in the requirements will not use the GPU.(maybe fixed in recent release)

------

#### If you only want to use pip to install the requirements, please use virtualenv and use the following command to install

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

------

#### The model in pth_file is Full With MLP Head，Params are 2,528,913. 

#### If you want to use simple model (Params 2,396,048) manually，please replace the Regressor with torch.nn.identity()