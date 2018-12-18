#!/bin/sh


caffessd_root_dir=$(./readconfig_ini_file.py settings-config.ini DEFAULT caffessd_root_dir)

$cd $caffessd_root_dir

if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
$caffessd_root_dir/build/tools/caffe train -solver="solver_train.prototxt" \
-weights="mobilenet_iter_73000.caffemodel" 
#-gpu 0 
