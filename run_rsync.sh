#!/bin/bash

rsync -ai --out-format="%n" \
    --exclude=.git \
    /Data/Skoltech/mental-states-agent-exps \
    o.inozemcev@10.16.90.27:/home/o.inozemcev
