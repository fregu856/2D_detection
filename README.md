# 2D_detection

Tensorflow implementaton of SqueezeDet (https://arxiv.org/pdf/1612.01051.pdf) based on the official implementation (https://github.com/BichenWuUCB/squeezeDet), trained on the KITTI dataset (http://www.cvlibs.net/datasets/kitti/).

- The pretrained SqueezeNet is squeezenet_v1.0.caffemodel and deploy.prototxt from https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0. Save these files to 2D_detection/data. To load these weights into Tensorflow, one need to have pycaffe installed (must be able to run "import caffe").



******


- Download KITTI (data_object_image_2.zip and data_object_label_2.zip).

- Install docker-ce:
-- Test
