CAFFE_PATH=/home/leizhou/Caffes/caffe_nd/build_cmake/install/python
python ../networks/build_resnet34_3d_preact_widedecoup_sgp_atten_strongest.py -m normal -c $CAFFE_PATH \
-n 64 -o ../prototxts/resnet34_3d_preact_widedecoup_sgp_atten_strongest -u --tr_file train.prototxt --ts_file deploy.prototxt