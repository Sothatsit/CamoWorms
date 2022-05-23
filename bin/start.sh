#!/bin/bash

TARGET_DIR=$(date +%s)

docker run -it --rm -v "$(pwd)/progress/$TARGET_DIR:/app/progress" camo-worms \
    swarm ./images/inputs/original-cropped.png ./progress $1 $2
