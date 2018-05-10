from base import BaseModule
from modules import *

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
from copy import deepcopy

class PreActWiderDecoupBlock(BaseModule):
    type='PreActWiderDecoup'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[1, 2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[3, 1, 1], \
                                    pad=[1, 0, 0], \
                                    stride=[2, 1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=self.stride2_3D, \
                                engine=2)

        br2a_tx1x1 = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=convtx1x1_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2_out = BNReLUConvModule(name_template=name, \
                                bn_params=self.bn_params, \
                                conv_params=convtx1x1_params, \
                                sync_bn=self.sync_bn, \
                                uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd])

        #### Eltwise Add ####
        eltadd_params = dict(name='eltadd_'+self.name_template)
        out = BaseModule('Eltwise', eltadd_params).attach(netspec, [shortcut, br2_out])

        return out

class PreActWiderDecoupSgpAttenBlock(BaseModule):
    type='PreActWiderDecoupSgpAtten'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[1, 2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[3, 1, 1], \
                                    pad=[1, 0, 0], \
                                    stride=[2, 1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=self.stride2_3D, \
                                engine=2)

        br2a_tx1x1 = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=convtx1x1_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        out = SgpAttenModule(name_template=name, \
                            bn_params=self.bn_params, \
                            stride=1, \
                            num_output=self.num_output, \
                            sync_bn=self.sync_bn, \
                            uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd], shortcut)

        return out

class PreActWiderDecoupSgpAttenStrongestBlock(BaseModule):
    type='PreActWiderDecoupSgpAtten'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', first_block=False, t_conv=False, sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn
        self.t_conv = t_conv
        self.first_block = first_block

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 1, 1] if first_block else [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[1, 1, 1] if self.first_block else [1, 2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[1, 1, 1], \
                                    pad=[0, 0, 0], \
                                    stride=[2, 1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        br2a_tx1x1 = SgpAttenModule(name_template=name, \
                            bn_params=self.bn_params, \
                            stride=self.stride, \
                            num_output=self.num_output, \
                            t_conv=self.t_conv, \
                            sync_bn=self.sync_bn, \
                            uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        out = SgpAttenModule(name_template=name, \
                            bn_params=self.bn_params, \
                            stride=1, \
                            num_output=self.num_output, \
                            t_conv=self.t_conv, \
                            sync_bn=self.sync_bn, \
                            uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd], shortcut)

        return out

class PreActWiderDecoupSgpAttenPlusTConvBlock(BaseModule):
    type='PreActWiderDecoupSgpAttenPlusTConv'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[1, 2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[3, 1, 1], \
                                    pad=[1, 0, 0], \
                                    stride=[2, 1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=self.stride2_3D, \
                                engine=2)

        br2a_tx1x1 = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=convtx1x1_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        out = SgpAttenPlusTConvModule(name_template=name, \
                            bn_params=self.bn_params, \
                            stride=1, \
                            num_output=self.num_output, \
                            sync_bn=self.sync_bn, \
                            uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd], shortcut)

        return out

class PreActWiderDecoup2dBlock(BaseModule):
    type='PreActWiderDecoup2d'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [3, 3]
        self.pad1 = [1, 1]
        self.kernel2_size = [1, 1]
        self.pad2 = [0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [2, 2]
            self.stride2_3D = [1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1]
            self.stride2_3D = [1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[3, 3], \
                                    pad=[1, 1], \
                                    stride=[2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[1, 1], \
                                    pad=[0, 0], \
                                    stride=[1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=self.stride2_3D, \
                                engine=2)

        br2a_tx1x1 = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=convtx1x1_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=[1, 1], \
                                engine=2)

        br2_out = BNReLUConvModule(name_template=name, \
                                bn_params=self.bn_params, \
                                conv_params=convtx1x1_params, \
                                sync_bn=self.sync_bn, \
                                uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd])

        #### Eltwise Add ####
        eltadd_params = dict(name='eltadd_'+self.name_template)
        out = BaseModule('Eltwise', eltadd_params).attach(netspec, [shortcut, br2_out])

        return out

class PreActDecoup2dBlock(BaseModule):
    type='PreActDecoup2d'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, uni_bn=True):
        self.uni_bn = uni_bn
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [3, 3]
        self.pad1 = [1, 1]
        self.kernel2_size = [1, 1]
        self.pad2 = [0, 0]
        if stride == 2:
            self.stride1_3D = [2, 2]
            self.stride2_3D = [1, 1]
        elif self.stride == 1:
            self.stride1_3D = [1, 1]
            self.stride2_3D = [1, 1]
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.num_output, \
                                    kernel_size=[3, 3], \
                                    pad=[1, 1], \
                                    stride=[2, 2], \
                                    engine=2)
            shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.num_output, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1], \
                                engine=2)

        br2_out = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### Eltwise Add ####
        eltadd_params = dict(name='eltadd_'+self.name_template)
        out = BaseModule('Eltwise', eltadd_params).attach(netspec, [shortcut, br2_out])

        return out

class PyramidDecoupBlock(BaseModule):
    type='PyramidDecoup'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### 1xdxd conv ####
            name = self.name_template + '_branch1_s'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.channels1, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[1, 2, 2], \
                                    engine=2)
            conv1xdxd_shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

            #### BNReLU + tx1x1 conv ####
            name = self.name_template + '_branch1_t'
            convtx1x1_params = dict(num_output=self.num_output, \
                                    kernel_size=[3, 1, 1], \
                                    pad=[1, 0, 0], \
                                    stride=[2, 1, 1], \
                                    engine=2)
            shortcut = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=convtx1x1_params,
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [conv1xdxd_shortcut])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_s'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### BNReLU + tx1x1 convA ####
        name = self.name_template + '_branch2a_t'
        convtx1x1_params = dict(num_output=self.num_output, \
                                kernel_size=self.kernel2_size, \
                                pad=self.pad2, \
                                stride=self.stride2_3D, \
                                engine=2)
        br2a_tx1x1 = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=convtx1x1_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_s'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### BNReLU + tx1x1 convB ####
        name = self.name_template + '_branch2b_t'
        res, br2_out = PyramidTemporalConvv3Module(name_template=name,
                                            bn_params=self.bn_params,
                                            stride=1,
                                            num_output=self.num_output,
                                            sync_bn=self.sync_bn,
                                            uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd])

        #### Eltwise Add ####
        # out = br2_out
        eltadd_params = dict(name='eltadd_'+self.name_template)
        out = BaseModule('Eltwise', eltadd_params).attach(netspec, [shortcut, br2_out])

        return out

class PreActWiderTempConvBlock(BaseModule):
    type='PreActWiderTempConv'
    def __init__(self, name_template, shortcut, num_output, stride, \
                main_branch='normal', sync_bn=False, wider=True, uni_bn=True):
        self.uni_bn = uni_bn
        self.wider = wider
        self.name_template = name_template
        self.shortcut = shortcut
        self.stride = stride
        self.main_branch = main_branch
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

        # set kernel_size & pad
        self.kernel1_size = [1, 3, 3]
        self.pad1 = [0, 1, 1]
        self.kernel2_size = [3, 1, 1]
        self.pad2 = [1, 0, 0]
        self.channels1 = num_output
        if wider:
            self.channels2 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            self.channels2 = num_output
        if stride == 2:
            self.stride1_3D = [1, 2, 2]
            self.stride2_3D = [2, 1, 1]
            if wider:
                self.channels1 = 27*(self.num_output/2)*self.num_output/(9*self.num_output/2+3*self.num_output)
        elif self.stride == 1:
            self.stride1_3D = [1, 1, 1]
            self.stride2_3D = [1, 1, 1]
            if wider:
                self.channels1 = 27*self.num_output*self.num_output/(9*self.num_output+3*self.num_output)
        else:
            raise ValueError('Unexpected stride value: {}'.format(self.stride))

    def attach(self, netspec, bottom):
        ########### Projection Shortcut Needs Pre Norm ###########
        if self.shortcut == 'projection':
            prenorm = BNReLUModule(name_template=self.name_template, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)

        ########### Shortcut: Identity or Projection (Downsample) ###########
        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            #### temporal stride2 1xdxd conv ####
            name = self.name_template + '_branch1_1x3x3'
            conv1xdxd_params = dict(name='conv' + name, \
                                    num_output=self.num_output, \
                                    kernel_size=[1, 3, 3], \
                                    pad=[0, 1, 1], \
                                    stride=[2, 2, 2], \
                                    engine=2)
            shortcut = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])

        ############ Main Branch ############
        assert(self.main_branch == 'normal'), "Only support normal main branch temporarily"
        
        #### (BNReLU + ) 1xdxd convA ####
        name = self.name_template + '_branch2a_1x3x3'
        conv1xdxd_params = dict(name='conv' + name, \
                        num_output=self.channels1, \
                        kernel_size=self.kernel1_size, \
                        pad=self.pad1, \
                        stride=self.stride1_3D, \
                        engine=2)
        if self.shortcut == 'projection':
            br2a_1xdxd = BaseModule('Convolution', conv1xdxd_params).attach(netspec, [prenorm])
        else:
            br2a_1xdxd = BNReLUConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        conv_params=conv1xdxd_params, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, bottom)
        
        #### Temporal convA ####
        name = self.name_template + '_branch2a_3x1x1'
        br2a_tx1x1 = TemporalConvModule(name_template=name, \
                                        bn_params=self.bn_params, \
                                        stride=self.stride, \
                                        num_output=self.num_output, \
                                        sync_bn=self.sync_bn, \
                                        uni_bn=self.uni_bn).attach(netspec, [br2a_1xdxd])

        #### BNReLU + 1xdxd convB ####
        name = self.name_template + '_branch2b_1x3x3'
        conv1xdxd_params = dict(num_output=self.channels2, \
                                kernel_size=self.kernel1_size, \
                                pad=self.pad1, \
                                stride=[1, 1, 1], \
                                engine=2)

        br2b_1xdxd = BNReLUConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    conv_params=conv1xdxd_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2a_tx1x1])

        #### Temporal convB ####
        name = self.name_template + '_branch2b_3x1x1'
        br2_out = TemporalConvModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    stride=1, \
                                    num_output=self.num_output, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, [br2b_1xdxd])

        #### Eltwise Add ####
        eltadd_params = dict(name='eltadd_'+self.name_template)
        out = BaseModule('Eltwise', eltadd_params).attach(netspec, [shortcut, br2_out])

        return out

class CorrAttentionBlock(BaseModule):
    type='CorrAttention'
    def __init__(self, name_template, template_type, 
                num_output, kernel_size, \
                max_displacement, pad, \
                stride_1=1, stride_2=1, template_index=0, \
                sync_bn=False, uni_bn=True):
        self.uni_bn = uni_bn
        self.name_template = name_template
        self.template_type = template_type
        self.template_index = template_index
        self.pad = pad
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.num_output = num_output
        self.sync_bn = sync_bn

        # default BN setting
        if uni_bn:
            self.bn_params = dict(frozen=False)
        else:
            self.bn_params = dict(use_global_stats=False)

    def attach(self, netspec, bottom):

        ########### Shortcut #############
        shortcut = bottom[0]

        ########### BNReLUCorr ############
        name = self.name_template
        corr_params = dict(name='corr_' + name,
                            kernel_size=self.kernel_size,
                            max_displacement=self.max_displacement,
                            template_type=self.template_type,
                            template_index=self.template_index,
                            pad=self.pad,
                            stride_1=self.stride_1,
                            stride_2=self.stride_2
                            )
        corr = BNReLUCorrModule(name_template=self.name_template,
                                bn_params=self.bn_params,
                                corr_params=corr_params).attach(netspec, bottom)

        ########### Convolution + Sigmoid ###########
        conv_params = dict(name='conv_' + name,
                            num_output=self.num_output,
                            kernel_size=[3, 3, 3],
                            pad=[1, 1, 1],
                            stride=[1, 1, 1],
                            engine=2)
        conv_corr = BaseModule('Convolution', conv_params).attach(netspec, [corr])
        sigmoid_params = dict(name='sigmoid_' + name)
        gated_conv_corr = BaseModule('Sigmoid', sigmoid_params).attach(netspec, [conv_corr])

        ########### multiply and add eltwise ############
        mul_params = dict(name='mul_' + name,
                            operation=0)
        attention = BaseModule('Eltwise', mul_params).attach(netspec, [shortcut, gated_conv_corr])
        add_params = dict(name='add_' + name,
                            operation=1)
        output = BaseModule('Eltwise', add_params).attach(netspec, [shortcut, attention])

        return output
