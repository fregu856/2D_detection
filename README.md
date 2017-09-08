# 2D_detection

Tensorflow implementaton of SqueezeDet (https://arxiv.org/pdf/1612.01051.pdf) based on the official implementation (https://github.com/BichenWuUCB/squeezeDet), trained on the KITTI dataset (http://www.cvlibs.net/datasets/kitti/).

- The pretrained SqueezeNet is squeezenet_v1.0.caffemodel and deploy.prototxt from https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0. Save these files to 2D_detection/data. To load these weights into Tensorflow, one need to have pycaffe installed (must be able to run "import caffe").



******


- Download KITTI (data_object_image_2.zip and data_object_label_2.zip).

- Install docker-ce:
- - $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
- - $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- - $ sudo apt-get update
- - $ sudo apt-get install -y docker-ce

- Install CUDA drivers (I used NC6, Tesla K80, see "Install CUDA drivers for NC VMs" in https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup):
- - $

- Install nvidia-docker:
- - $ wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
- - $ sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
- - $ sudo nvidia-docker run --rm nvidia/cuda nvidia-smi


