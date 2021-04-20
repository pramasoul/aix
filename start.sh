#!/usr//bin/bash
# FIXME: put directory info in name
### docker run --gpus all -d -it -p 8848:8888 -v $(pwd):/home/jovyan/work -v /data-pool/nn:/data -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root --restart on-failure:16 --name gpu-jupyter gpu-jupyter
docker run --gpus all -d -it --rm \
       -p 6006:6006 \
       -p 8848:8888 \
       -v $(pwd):/home/jovyan/work \
       -v /data-pool/nn:/data \
       -e TFDS_DATA_DIR=/data/tfds \
       -e GRANT_SUDO=yes \
       -e JUPYTER_ENABLE_LAB=yes \
       --user root \
       --name gpu-jupyter \
       gpu-jupyter
