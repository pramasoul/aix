#!/usr//bin/bash
# FIXME: put directory info in name
docker run --gpus all -d -it -p 8848:8888 -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root --restart on-failure:16 --name gpu-jupyter_1 gpu-jupyter
