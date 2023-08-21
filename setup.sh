sudo apt-get install qt6-tools-dev
sudo apt install libxcb-cursor0

cd /home/alice/git3/pet-detector
python3 -m venv ./env
source env/bin/activate
pip install -r requirements.txt

cd /home/alice/git3/pet-detector
source env/bin/activate

python main.py
