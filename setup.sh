sudo apt-get install qt6-tools-dev
sudo apt install libxcb-cursor0

cd /home/alice/git3/pet-detector
python3 -m venv ./env
source env/bin/activate
pip install -r requirements.txt

cd /home/alice/git3/pet-detector
source env/bin/activate

python main.py



#https://tfhub.dev/s?module-type=image-object-detection
#Download https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1
#Download https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1

# install cuda
# https://www.tensorflow.org/install/pip