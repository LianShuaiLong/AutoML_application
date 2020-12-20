export PATH=$PATH:~/.local/bin
export PYTHONPATH=$PYTHONPATH:/usr/bin/python3

nnictl create --config ./config.yml --port 8080
