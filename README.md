# 2D_detection

- Tensorflow implementaton of SqueezeDet (https://arxiv.org/pdf/1612.01051.pdf) based on the official implementation (https://github.com/BichenWuUCB/squeezeDet), trained on the KITTI dataset (http://www.cvlibs.net/datasets/kitti/).

- Youtube video of results (https://youtu.be/5BBwjvlUULI):
- [![demo video with results](https://img.youtube.com/vi/5BBwjvlUULI/0.jpg)](https://www.youtube.com/watch?v=5BBwjvlUULI)

- The results in the video can obviously be improved, but because of limited computing resources (personally funded Azure VM) I did not perform any further hyperparameter tuning. 

****
## Training details:

- The SqueezeNet network was initialized with the pretrained model in https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0 (squeezenet_v1.0.caffemodel and deploy.prototxt). To load these weights into TensorFlow, one needs to have pycaffe installed (must be able to run "import caffe"). Run get_caffemodel_weights in utilities.py and save the output as caffemodel_weights.pkl in 2D_detection/data using cPickle. These files (caffemodel_weights.pkl, squeezenet_v1.0.caffemodel and deploy.prototxt) are also included in /data in the repo.

- Batch size: 32.
- For all other hyperparameters I used the same values as in the paper.

- Training loss:
- ![alt text](https://lh3.googleusercontent.com/KX4x--0GCHZPFkaFykI7AC7r6ys_KvH-ypfPK3JmzTSgYAZgx4LgT1pj7SbaVjzhP9ZwOgH8-dio3F4=w1920-h937)

- Validation loss:
- ![alt text](https://lh6.googleusercontent.com/xE6YioyQtzBW0t5Pw1YSnRUn3wRoJENm7pxo9VFJMNdwQJZYhtMzCqUFE1ONHuXXeg3R7_udm4xObOI=w1920-h937)

- The results in the video above was obtained with the model at epoch 58, for which a checkpoint is included in /training_logs/best_model in the repo.


******
## Training on Microsoft Azure:

- Download KITTI (data_object_image_2.zip and data_object_label_2.zip).

- Install docker-ce:
- - $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
- - $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- - $ sudo apt-get update
- - $ sudo apt-get install -y docker-ce

- Install CUDA drivers (I used NC6 on Azure, Tesla K80, see "Install CUDA drivers for NC VMs" in https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup):
- - $ CUDA_REPO_PKG=cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
- - $ wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} 
- - $ sudo dpkg -i /tmp/${CUDA_REPO_PKG}
- - $ rm -f /tmp/${CUDA_REPO_PKG}
- - $ sudo apt-get update
- - $ sudo apt-get install cuda-drivers
- - Reboot the VM.

- Install nvidia-docker:
- - $ wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
- - $ sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
- - $ sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

- Download latest tensorflow docker image with GPU support (tensorflow 1.3) :
- - $ sudo docker pull tensorflow/tensorflow:latest-gpu

- Create start_docker_image.sh containing:
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="fregu856_GPU"


NV_GPU="$GPUIDS" nvidia-docker run -it --rm \
        -p 5584:5584 \
        --name "$NAME""$GPUIDS" \
        -v /home/fregu856:/root/ \
        tensorflow/tensorflow:latest-gpu bash
```

- /root/ will now be mapped to /home/fregu856 (i.e., $ cd -- takes you to the regular home folder). 

- To start the image:
- - $ sudo sh start_docker_image.sh 
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit fregu856_GPU0 tensorflow/tensorflow:latest-gpu
- To stop the image when it’s running:
- - $ sudo docker stop fregu856_GPU0
- To exit the image without killing running code:
- - Ctrl-P + Q
- To get back into a running image:
- - $ sudo docker attach fregu856_GPU0
- To open more than one terminal at the same time:
- - $ sudo docker exec -it fregu856_GPU0 bash

- Intall needed software (inside the docker image):
- - $ apt-get update
- - $ apt-get install nano
- - $ apt-get install sudo
- - $ apt-get install wget
- - $ sudo apt-get install libopencv-dev python-opencv
- Commit changes to the image (otherwise, the installed packages will be removed at exit!).
