#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -g gk77
#PJM -j
#PJM -N dcpdd160mbrt
#PJM -o dcpdd160mbrt
#PJM -e dcpdd160mbrt

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python dolma_sample_load.py