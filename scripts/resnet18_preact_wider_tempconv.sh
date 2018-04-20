CAFFE_PATH=/home/leizhou/Caffes/caffe_nd/build_cmake/install/python
python ../networks/build_resnet18_preact_wider_tempconv.py -m normal -c $CAFFE_PATH \
-n 64 -o ../prototxts/resnet18_preact_wider_tempconv -u --tr_file train.prototxt --ts_file deploy.prototxt