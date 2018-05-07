import os
from argparse import ArgumentParser
import pdb

parser = ArgumentParser(description=""" This script generates train & val prototxt files for action recognition""")
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""", required=True)
parser.add_argument('-c', '--caffe_lib', help="""path of caffe lib""", required=True)
parser.add_argument('-n', '--num_output_stage1', help="""Number of filters in stage 1 of resnet""", type=int, default=128)
parser.add_argument('-o', '--output_folder', help="""Folder to contain output prototxts""")
parser.add_argument('-b', '--blocks', type=int, nargs='+', help="""Number of Blocks in the 4 resnet stages""", default=[2, 2, 2, 2])
parser.add_argument('-s', '--sync_bn', action='store_true')
parser.add_argument('-u', '--uni_bn', action='store_true')
parser.add_argument('--tr_file', type=str, help="""name of train prototxt file""")
parser.add_argument('--ts_file', type=str, help="""name of test prototxt file""")
parser.set_defaults(uni_bn=False)
parser.set_defaults(sync_bn=False)
args = parser.parse_args()

# pdb.set_trace()
import sys

sys.path.insert(0, args.caffe_lib)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P

sys.path.append('../lib')
from base import BaseModule
from modules import BNReLUModule, BNReLUConvModule
from blocks import PreActWiderDecoupBlock, PreActWiderDecoupSgpAttenBlock, PreActWiderDecoupSgpAttenPlusTConvBlock

num2letter = ['a', 'b', 'c', 'd', 'e', 'f']

def write_prototxt(is_train, output_folder, \
                    filename, main_branch, \
                    num_output_stage1, \
                    blocks, sync_bn, uni_bn):

    netspec = caffe.NetSpec()

    #### Input Setting ####
    crop_size = 112
    width = 170
    height = 128
    length = 16
    step = 1
    num_segments = 1

    if is_train:
        use_global_stats = False
    else:
        use_global_stats = True

    #### Data layer ####
    if is_train:
        data_train_params = dict(name='data', \
                            ntop=2, \
                            video_data_param=dict( \
                                source="../kinetics_train_list_of.txt", \
                                batch_size=26, \
                                new_width=width, \
                                new_height=height, \
                                new_length=length, \
                                num_segments=num_segments, \
                                modality=1, \
				length_first=True, \
                                step=step, \
                                name_pattern='%c_%05d.jpg', \
                                shuffle=True), \
                            transform_param=dict(
                                crop_size=crop_size, \
                                mirror=True, \
                                multi_scale=True, \
                                max_distort=1, \
                                scale_ratios=[1, 0.875, 0.75, 0.66], \
                                mean_value=128, \
                                is_flow=True), \
                            include=dict(phase=0))

        data_val_params = dict(name='vdata', \
                                ntop=2, \
                                video_data_param=dict(
                                    source="../kinetics_val_list_of.txt", \
                                    batch_size=1, \
                                    new_width=width, \
                                    new_height=height, \
                                    new_length=length, \
                                    num_segments=num_segments, \
                                    modality=1, \
				    length_first=True, \
                                    step=step, \
                                    name_pattern='%c_%05d.jpg'), \
                                transform_param=dict(
                                    crop_size=crop_size, \
                                    mirror=False, \
                                    mean_value=128, \
                                    is_flow=True), \
                                include=dict(phase=1))
        # pdb.set_trace()
        netspec.data, netspec.label = BaseModule('VideoData', data_train_params).attach(netspec, [])
        netspec.vdata, netspec.vlabel = BaseModule('VideoData', data_val_params).attach(netspec, [])
    else:
        data_params = dict(name='data', \
                            dummy_data_param=dict( \
                                shape=dict(\
                                    dim=[10, 2, length, crop_size, crop_size])))
        netspec.data = BaseModule('DummyData', data_params).attach(netspec, [])

    #### (Optional) Reshape Layer ####
    if is_train:
        reshape_params = dict(name='data_reshape', \
                            reshape_param=dict( \
                                shape=dict(dim=[-1, 2, length, crop_size, crop_size])))
        netspec.data_reshape = BaseModule('Reshape', reshape_params).attach(netspec, [netspec.data])

    #### Stage 1 ####
    channels = 3*7*7*3*64/(7*7*3+3*64)
    conv1xdxd_params = dict(name='conv1_1x3x3', \
                            num_output=channels, \
                            kernel_size=[1, 7, 7], \
                            pad=[0, 3, 3], \
                            stride=[1, 2, 2], \
                            engine=2)
    conv1_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(
                            netspec, [netspec.data_reshape if is_train else netspec.data])
    convtx1x1_params = dict(name='conv1_3x1x1', \
                            num_output=64, \
                            kernel_size=[3, 1, 1], \
                            pad=[1, 0, 0], \
                            stride=[2, 1, 1], \
                            engine=2)
    if uni_bn:
        bn_params = dict(frozen=False)
    else:
        bn_params = dict(use_global_stats=use_global_stats)
    stage1 = BNReLUConvModule(name_template='1', \
                            bn_params=bn_params, \
                            conv_params=convtx1x1_params, \
                            sync_bn=sync_bn, \
                            uni_bn=uni_bn).attach(netspec, [conv1_1xdxd])
    num_output = num_output_stage1

    #### Stages 2 - 5 ####
    last = stage1
    for stage in range(4):
        for block in range(blocks[stage]):
            # First block usually projection
            if block == 0:
                shortcut = 'projection'
                stride = 2
                if stage == 0:
                    shortcut = 'identity'
                    stride = 1
            else:
                shortcut = 'identity'
                stride = 1

            name = str(stage+2) + num2letter[int(block)]
            curr_num_output = num_output * (2 ** (stage))

            if uni_bn:
                params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, frozen=False)
            else:
                params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, use_global_stats=use_global_stats)

            last = PreActWiderDecoupSgpAttenPlusTConvBlock(name_template=name, \
            		shortcut=shortcut, \
            		num_output=curr_num_output, \
            		stride=stride, \
            		sync_bn=sync_bn, \
            		uni_bn=uni_bn).attach(netspec, [last])

    #### Last Norm & ReLU ####
    if uni_bn:
        bn_params = dict(frozen=False)
    else:
        bn_params = dict(use_global_stats=use_global_stats)
    last = BNReLUModule(name_template='5b', \
                        bn_params=bn_params, \
                        sync_bn=sync_bn, \
                        uni_bn=uni_bn).attach(netspec, [last])

    #### pool5 ####
    pool_params = dict(global_pooling=True, pool=P.Pooling.AVE, name='pool5')
    pool = BaseModule('Pooling', pool_params).attach(netspec, [last])

    #### pool5_reshape ####
    reshape_params = dict(shape=dict(dim=[-1, num_output_stage1 * 8]), name='pool5_reshape')
    reshape = BaseModule('Reshape', reshape_params).attach(netspec, [pool])

    #### dropout ####
    dropout_params = dict(dropout_ratio=0.2, name='dropout')
    dropout = BaseModule('Dropout', dropout_params).attach(netspec, [reshape])
    
    #### ip ####
    ip_params = dict(name='fc400', num_output=400)
    ip = BaseModule('InnerProduct', ip_params).attach(netspec, [dropout])

    if is_train:

        #### Softmax Loss ####
        smax_params = dict(name='loss')
        smax_loss = BaseModule('SoftmaxWithLoss', smax_params).attach(netspec, [ip, netspec.label])

        #### Top1 Accuracy ####
        top1_params = dict(name='top1', accuracy_param=dict(top_k=1), include=dict(phase=1))
        top1 = BaseModule('Accuracy', top1_params).attach(netspec, [ip, netspec.label])

        #### Top5 Accuracy ####
        top5_params = dict(name='top5', accuracy_param=dict(top_k=5), include=dict(phase=1))
        top5 = BaseModule('Accuracy', top5_params).attach(netspec, [ip, netspec.label])

    filepath = os.path.join(output_folder, filename)
    fp = open(filepath, 'w')
    print >> fp, netspec.to_proto()
    fp.close()

# if __name__ == '__main__':

#### Train Prototxt ####
write_prototxt(True, args.output_folder, \
                args.tr_file, args.main_branch, \
                args.num_output_stage1, \
                args.blocks, args.sync_bn, args.uni_bn)

#### Deploy Prototxt ####
write_prototxt(False, args.output_folder, \
                args.ts_file, args.main_branch, \
                args.num_output_stage1, \
                args.blocks, args.sync_bn, args.uni_bn)
