#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -g gk77
#PJM -j
#PJM -N data_collection
#PJM -o data_collection
#PJM -e data_collection

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

#python dolma_sample_load.py
python creat_mia_dataset.py