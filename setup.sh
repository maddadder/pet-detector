#sudo apt-get install qt6-tools-dev
#sudo apt install libxcb-cursor0
cd /home/alice/git3/pet-detector
conda create -n tf python=3.10
conda activate tf
conda install nvidia::cudnn cuda-version=11
pip install -r requirements.txt

cd /home/alice/git3/pet-detector
conda activate tf

python3 serving_default.py



pip freeze > requirements.txt

#https://tfhub.dev/s?module-type=image-object-detection
#Download https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1
#Download https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1

#install cuda:
#https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
#sudo prime-select nvidia
#OR maybe
# sudo ubuntu-drivers autoinstall
# https://www.tensorflow.org/install/pip
