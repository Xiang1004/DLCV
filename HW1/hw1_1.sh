#!/bin/bash

wget -O p1-79.pth https://www.dropbox.com/s/qwbgsq560onb2kt/p1-79.pth?dl=1

python3 p1/test.py --img_dir $1 --output_path $2 --checkpoint_path p1-79.pth