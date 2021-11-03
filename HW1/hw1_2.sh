#!/bin/bash

wget -O p2-99.pth https://www.dropbox.com/sh/jt1fd7i3fv6g50e/AAA-bhWUDSykJaUdlyvvwQZ5a?dl=1
python3 p2/test.py --img_dir $1 --output_path $2 --checkpoint_path p2-99.pth