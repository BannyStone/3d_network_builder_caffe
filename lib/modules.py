from base import BaseModule
from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P

class BNReLUModule(BaseModule):
    type='BNReLU'
    def __init__(self, name_template, bn_params, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
            self.bnParams.update(name='bn_' + name_template)
        else:
            self.batchNormParams = bn_params.copy()
            self.batchNormParams.update(name='batchnorm_' + name_template)
            self.scaleParams = scale_params.copy()
            self.scaleParams.update(name='scale_' + name_template)
        self.reluParams = dict(name=name_template + '_relu')

    def attach(self, netspec, bottom):
        if self.uni_bn:
            if self.sync_bn:
                bn = BaseModule('SyncBN', self.bnParams).attach(netspec, bottom)
            else:
                bn = BaseModule('BN', self.bnParams).attach(netspec, bottom)
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [bn])
        else:
            batch_norm = BaseModule('BatchNorm', self.batchNormParams).attach(netspec, bottom)
            scale = BaseModule('Scale', self.scaleParams).attach(netspec, [batch_norm])
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [scale])
        return relu

class BNReLUConvModule(BaseModule):
    type='BNReLUConv'
    def __init__(self, name_template, bn_params, conv_params, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
		# scale_params cannot be None when using BatchNorm
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
            self.bnParams.update(name='bn_' + name_template)
        else:
            self.batchNormParams = bn_params.copy()
            self.batchNormParams.update(name='batchnorm_' + name_template)
            self.scaleParams = scale_params.copy()
            self.scaleParams.update(name='scale_' + name_template)
        self.reluParams = dict(name=name_template + '_relu')
        self.convParams = conv_params.copy()
        self.convParams.update(name='conv' + name_template)

    def attach(self, netspec, bottom):
        if self.uni_bn:
            if self.sync_bn:
                bn = BaseModule('SyncBN', self.bnParams).attach(netspec, bottom)
            else:
                bn = BaseModule('BN', self.bnParams).attach(netspec, bottom)
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [bn])
        else:
            batch_norm = BaseModule('BatchNorm', self.batchNormParams).attach(netspec, bottom)
            scale = BaseModule('Scale', self.scaleParams).attach(netspec, [batch_norm])
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [scale])
        conv = BaseModule('Convolution', self.convParams).attach(netspec, [relu])
        return conv

class BNReLUCorrModule(BaseModule):
    type='BNReLUCorr'
    def __init__(self, name_template, bn_params, corr_params, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        # scale_params cannot be None when using BatchNorm
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
            self.bnParams.update(name='bn_' + name_template)
        else:
            self.batchNormParams = bn_params.copy()
            self.batchNormParams.update(name='batchnorm_' + name_template)
            self.scaleParams = scale_params.copy()
            self.scaleParams.update(name='scale_' + name_template)
        self.reluParams = dict(name=name_template + '_relu')
        self.corrParams = corr_params.copy()
        self.corrParams.update(name='corr' + name_template)

    def attach(self, netspec, bottom):
        if self.uni_bn:
            if self.sync_bn:
                bn = BaseModule('SyncBN', self.bnParams).attach(netspec, bottom)
            else:
                bn = BaseModule('BN', self.bnParams).attach(netspec, bottom)
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [bn])
        else:
            batch_norm = BaseModule('BatchNorm', self.batchNormParams).attach(netspec, bottom)
            scale = BaseModule('Scale', self.scaleParams).attach(netspec, [batch_norm])
            relu = BaseModule('ReLU', self.reluParams).attach(netspec, [scale])
        corr = BaseModule('Corrv1', self.corrParams).attach(netspec, [relu])
        return corr

class TemporalConvModule(BaseModule):
    type='TemporalConv'
    def __init__(self, name_template, bn_params, stride, num_output, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        # scale_params cannot be None when using BatchNorm
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
            self.bnParams.update(name='bn_' + name_template)
        else:
            self.batchNormParams = bn_params.copy()
            self.batchNormParams.update(name='batchnorm_' + name_template)
            self.scaleParams = scale_params.copy()
            self.scaleParams.update(name='scale_' + name_template)
        self.reluParams = dict(name=name_template + '_relu')
        self.conv1x1x1Params = dict(name='conv1x1x1_'+name_template, \
                                    num_output=num_output, \
                                    kernel_size=[1, 1, 1], \
                                    pad=[0, 0, 0], \
                                    stride=[stride, 1, 1])
        self.t_convParams = dict(name='t_conv'+name_template, \
                                num_output=num_output, \
                                kernel_size=[3, 1, 1], \
                                pad=[1, 0, 0], \
                                stride=[stride, 1, 1])
        self.poolingParams = dict(name='spt_pool_'+name_template, \
                                pool=P.Pooling.AVE, \
                                spatial_global_pooling=True)
        self.reshapeParams = dict(name='reshape_'+name_template, \
                                shape=dict(dim=[0, 0, 0]))
        ## support Bias ##
        self.biasParams = dict(name='bias_'+name_template, \
                                axis=0)

    def attach(self, netspec, bottom):

        ######## Pre Norm ########
        prenorm = BNReLUModule(name_template=self.name_template, \
                                bn_params=self.bnParams, \
                                sync_bn=self.sync_bn).attach(netspec, bottom)

        ######## 1x1x1 Shortcut ########
        shortcut = BaseModule('Convolution', self.conv1x1x1Params).attach(netspec, [prenorm])

        ######## Main Branch ########

        #### Spatial Global Pooling ####
        pooling = BaseModule('Pooling', self.poolingParams).attach(netspec, [prenorm])

        #### Temporal Convolution ####
        t_conv = BaseModule('Convolution', self.t_convParams).attach(netspec, [pooling])

        #### Reshape ####
        reshape = BaseModule('Reshape', self.reshapeParams).attach(netspec, [t_conv])

        ######## Bias ########
        bias = BaseModule('Bias', self.biasParams).attach(netspec, [shortcut, reshape])

        return bias

class SgpAttenModule(BaseModule):
    type='SgpAtten'
    def __init__(self, name_template, bn_params, stride, num_output, t_conv=False, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        # scale_params cannot be None when using BatchNorm
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
        else:
            self.batchNormParams = bn_params.copy()
            self.scaleParams = scale_params.copy()
        self.reluParams = dict(name=name_template + '_relu')
        self.convtx1x1Params = dict(name='conv'+name_template, \
                                    num_output=num_output, \
                                    kernel_size=[3, 1, 1] if t_conv else [1, 1, 1], \
                                    pad=[1, 0, 0] if t_conv else [0, 0, 0], \
                                    stride=[stride, 1, 1])
        self.t_convParams = dict(name='conv'+name_template+'_atten', \
                                num_output=num_output, \
                                kernel_size=[3, 1, 1], \
                                pad=[1, 0, 0], \
                                stride=[stride, 1, 1])
        self.poolingParams = dict(name='sgp_'+name_template, \
                                pool=P.Pooling.AVE, \
                                spatial_global_pooling=True)
        self.sigmoidParams = dict(name='sigmoid_'+name_template)
        ## axpxpy params ##
        if 'b_t' not in name_template:
            self.addParams = dict(name='add_'+name_template)
        else:
            self.addParams = dict(name='add_'+name_template[:2])

    def attach(self, netspec, bottom, residual_branch=None):

        ######## Pre Norm ########
        prenorm = BNReLUModule(name_template=self.name_template, \
                                bn_params=self.bnParams, \
                                sync_bn=self.sync_bn).attach(netspec, bottom)

        ######## 1x1x1 Shortcut ########
        shortcut = BaseModule('Convolution', self.convtx1x1Params).attach(netspec, [prenorm])

        ######## Main Branch ########

        #### Spatial Global Pooling ####
        pooling = BaseModule('Pooling', self.poolingParams).attach(netspec, [prenorm])

        #### Temporal Convolution ####
        t_conv = BaseModule('Convolution', self.t_convParams).attach(netspec, [pooling])

        #### Sigmoid ####
        sigmoid = BaseModule('Sigmoid', self.sigmoidParams).attach(netspec, [t_conv])

        ######## add ########
        if residual_branch == None:
            out = BaseModule('Axpx', self.addParams).attach(netspec, [sigmoid, shortcut])
        else:
            out = BaseModule('Axpxpy', self.addParams).attach(netspec, [sigmoid, shortcut, residual_branch])

        return out

class SgpAttenPlusTConvModule(BaseModule):
    type='SgpAttenPlusTConv'
    def __init__(self, name_template, bn_params, stride, num_output, sync_bn=False, uni_bn=True, scale_params=None):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        # scale_params cannot be None when using BatchNorm
        if not uni_bn and scale_params is None:
            scale_params = dict()
        if uni_bn:
            self.bnParams = bn_params.copy()
        else:
            self.batchNormParams = bn_params.copy()
            self.scaleParams = scale_params.copy()
        self.reluParams = dict(name=name_template + '_relu')
        self.conv3x1x1Params = dict(name='conv'+name_template, \
                                    num_output=num_output, \
                                    kernel_size=[3, 1, 1], \
                                    pad=[1, 0, 0], \
                                    stride=[stride, 1, 1])
        self.t_convParams = dict(name='conv'+name_template+'_atten', \
                                num_output=num_output, \
                                kernel_size=[3, 1, 1], \
                                pad=[1, 0, 0], \
                                stride=[stride, 1, 1])
        self.poolingParams = dict(name='sgp_'+name_template, \
                                pool=P.Pooling.AVE, \
                                spatial_global_pooling=True)
        self.sigmoidParams = dict(name='sigmoid_'+name_template)
        ## axpxpy params ##
        self.addParams = dict(name='add_'+name_template[:2])

    def attach(self, netspec, bottom, residual_branch):

        ######## Pre Norm ########
        prenorm = BNReLUModule(name_template=self.name_template, \
                                bn_params=self.bnParams, \
                                sync_bn=self.sync_bn).attach(netspec, bottom)

        ######## 1x1x1 Shortcut ########
        shortcut = BaseModule('Convolution', self.conv3x1x1Params).attach(netspec, [prenorm])

        ######## Main Branch ########

        #### Spatial Global Pooling ####
        pooling = BaseModule('Pooling', self.poolingParams).attach(netspec, [prenorm])

        #### Temporal Convolution ####
        t_conv = BaseModule('Convolution', self.t_convParams).attach(netspec, [pooling])

        #### Sigmoid ####
        sigmoid = BaseModule('Sigmoid', self.sigmoidParams).attach(netspec, [t_conv])

        ######## add ########
        out = BaseModule('Axpxpy', self.addParams).attach(netspec, [sigmoid, shortcut, residual_branch])

        return out

class PyramidTemporalConvModule(BaseModule):
    type='PyramidTemporalConv' # start from here
    def __init__(self, name_template, bn_params, stride, num_output, sync_bn=False, uni_bn=True):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        self.num_output = num_output
        self.bn_params = bn_params

    def attach(self, netspec, bottom, res=None):
        #### BNReLU + tx1x1 convA ####
        name = self.name_template
        prenorm = BNReLUModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[self.stride,1,1], \
                                engine=2)
        br2a_tx1x1 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [prenorm])

        #### pyramid_1 ####
        name = self.name_template + '_p1'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1, 3, 3],
                            pad = [0, 1, 1],
                            stride=[1, 2, 2],
                            pool=0)
        pool1 = BaseModule('Pooling', pool_params).attach(netspec, [br2a_tx1x1])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p1 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [pool1])
        interp_params = dict(name='interp_' + name)
        interp_p1 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p1, br2a_tx1x1])

        #### pyramid_2 ####
        name = self.name_template + '_p2'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1,3,3],
                            pad=[0,1,1],
                            stride=[1,2,2],
                            pool=0)
        pool2 = BaseModule('Pooling', pool_params).attach(netspec, [br2a_tx1x1_p1])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p2 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [pool2])
        interp_params = dict(name='interp_' + name)
        interp_p2 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p2, br2a_tx1x1])

        #### pyramid_extreme ####
        ## Not Added Yet

        #### add ####
        if res is None:
            name = self.name_template + '_add'
            eltwise_params = dict(name=name, operation=1, coeff=[1, 0.5, 0.5]) # [1, 1, 1]
            out = BaseModule('Eltwise', eltwise_params).attach(netspec, [br2a_tx1x1, interp_p1, interp_p2])
        else:
            name = 'eltadd_' + res[0]
            eltwise_params = dict(name=name, operation=1, coeff=[1, 1, 0.5, 0.5])
            out = BaseModule('Eltwise', eltwise_params).attach(netspec, [res[1], br2a_tx1x1, interp_p1, interp_p2])

        return out

class PyramidTemporalConvv2Module(BaseModule):
    type='PyramidTemporalConvv2'
    def __init__(self, name_template, bn_params, stride, num_output, sync_bn=False, uni_bn=True):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        self.num_output = num_output
        self.bn_params = bn_params

    def attach(self, netspec, bottom):
        #### BNReLU + tx1x1 convA ####
        name = self.name_template
        prenorm = BNReLUModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[self.stride,1,1], \
                                engine=2)
        br2a_tx1x1 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [prenorm])

        #### pyramid_1 ####
        name = self.name_template + '_p1'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1, 2, 2],
                            pad = [0, 0, 0],
                            stride=[1, 2, 2],
                            pool=0)
        pool1 = BaseModule('Pooling', pool_params).attach(netspec, [br2a_tx1x1])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output/2, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p1 = BNReLUConvModule(name_template=name,
                        bn_params=self.bn_params,
                        conv_params=convtx1x1_params).attach(netspec, [pool1])
        interp_params = dict(name='interp_' + name)
        interp_p1 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p1, br2a_tx1x1])

        #### pyramid_2 ####
        name = self.name_template + '_p2'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1,2,2],
                            pad=[0,0,0],
                            stride=[1,2,2],
                            pool=0)
        pool2 = BaseModule('Pooling', pool_params).attach(netspec, [br2a_tx1x1_p1])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output/2, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p2 = BNReLUConvModule(name_template=name,
                        bn_params=self.bn_params,
                        conv_params=convtx1x1_params).attach(netspec, [pool2])
        interp_params = dict(name='interp_' + name)
        interp_p2 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p2, br2a_tx1x1])

        #### pyramid_extreme ####
        # Not Implemented

        #### concat ####
        name = self.name_template + '_concat'
        concat_params = dict(name=name) # [1, 1, 1]
        concat = BaseModule('Concat', concat_params).attach(netspec, [br2a_tx1x1, interp_p1, interp_p2])

        #### fusion conv ####
        name = self.name_template + '_fusion'
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        out = BNReLUConvModule(name_template=name,
                        bn_params=self.bn_params,
                        conv_params=convtx1x1_params).attach(netspec, [concat])
        
        return br2a_tx1x1, out

class PyramidTemporalConvv3Module(BaseModule):
    type='PyramidTemporalConvv3'
    def __init__(self, name_template, bn_params, stride, num_output, sync_bn=False, uni_bn=True):
        self.uni_bn = uni_bn
        self.sync_bn = sync_bn
        self.stride = stride
        self.name_template = name_template
        self.num_output = num_output
        self.bn_params = bn_params

    def attach(self, netspec, bottom):
        #### BNReLU + tx1x1 convA ####
        name = self.name_template
        prenorm = BNReLUModule(name_template=name, \
                                    bn_params=self.bn_params, \
                                    sync_bn=self.sync_bn, \
                                    uni_bn=self.uni_bn).attach(netspec, bottom)
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[self.stride,1,1], \
                                engine=2)
        br2a_tx1x1 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [prenorm])

        #### pyramid_1 ####
        name = self.name_template + '_p1'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1, 3, 3],
                            pad = [0, 1, 1],
                            stride=[1, 2, 2],
                            pool=0)
        pool1 = BaseModule('Pooling', pool_params).attach(netspec, [prenorm])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output/2, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p1 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [pool1])
        interp_params = dict(name='interp_' + name)
        interp_p1 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p1, br2a_tx1x1])

        #### pyramid_2 ####
        name = self.name_template + '_p2'
        pool_params = dict(name='pool_' + name,
                            kernel_size=[1,3,3],
                            pad=[0,1,1],
                            stride=[1,4,4],
                            pool=0)
        pool2 = BaseModule('Pooling', pool_params).attach(netspec, [prenorm])
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output/2, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        br2a_tx1x1_p2 = BaseModule('Convolution', convtx1x1_params).attach(netspec, [pool2])
        interp_params = dict(name='interp_' + name)
        interp_p2 = BaseModule('Interp', interp_params).attach(netspec, [br2a_tx1x1_p2, br2a_tx1x1])

        #### pyramid_extreme ####
        # Not Implemented

        #### concat ####
        name = self.name_template + '_concat'
        concat_params = dict(name=name) # [1, 1, 1]
        concat = BaseModule('Concat', concat_params).attach(netspec, [br2a_tx1x1, interp_p1, interp_p2])

        #### fusion conv ####
        name = self.name_template + '_fusion'
        convtx1x1_params = dict(name='conv_' + name, \
                                num_output=self.num_output, \
                                kernel_size=[3,1,1], \
                                pad=[1,0,0], \
                                stride=[1,1,1], \
                                engine=2)
        out = BNReLUConvModule(name_template=name,
                        bn_params=self.bn_params,
                        conv_params=convtx1x1_params).attach(netspec, [concat])
        
        return br2a_tx1x1, out