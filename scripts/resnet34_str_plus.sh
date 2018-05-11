CAFFE_PATH=/home/leizhou/Caffes/caffe_nd/build_cmake/install/python
python ../networks/build_resnet34_str_plus.py -m normal -c $CAFFE_PATH \
-n 64 -o ../prototxts/resnet34_str_plus -u --tr_file train.prototxt --ts_file deploy.prototxt