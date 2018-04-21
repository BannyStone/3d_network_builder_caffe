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
