cd /home/alice/git3/pet-detector
bash --init-file <(echo ". "$HOME/.bashrc"; conda activate tf; python3 serving_default.py")
