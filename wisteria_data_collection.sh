#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N data_collectionidx0
#PJM -o data_collectionidx0
#PJM -e data_collectionidx0

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

#python dolma_sample_load.py
python create_mia_dataset.py --device wisteria --domain arxiv --batch_size 100