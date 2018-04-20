from caffe import layers as L
from caffe import params as P

'''
Get and set default parameters
'''
class Config(object):
    _default_params = {}

    @staticmethod
    def get_default_params(layer):
        if len(Config._default_params) == 0:
            execfile("../configs/default.params.py", Config._default_params)

        if layer in Config._default_params:
            return Config._default_params[layer]
        else:
            return {}

class BaseModule(object):
    def __init__(self, type_name, params):
        if '_required' not in self.__dict__:
            self._required = []
        self._type_name = type_name
        self._default = dict()
        self._init_default_params()
        self._check_required_params(params)
        self._params = params.copy()

    def get_required_names(self):
        return self._required

    def get_default_params(self):
        return self._default

    def _init_default_params(self):
        self._default = Config.get_default_params(self._type_name)

    def _construct_params(self):
        params = self._default.copy()
        params.update(self._params.copy())
        # block bias_filler and restrict params number when bias_term set to false
        if 'bias_term' in params and params['bias_term'] == False:
            if 'bias_filler' in params:
                del params['bias_filler']
            if 'param' in params and len(params['param']) == 2:
                params['param'].pop()
        if self._type_name == 'SyncBN':
            if 'frozen' in params:
                del params['frozen']

        return params

    # def override_default_param(self, key, val):
    #     self._default[key] = val

    def _check_required_params(self, required_params):
        for r in self._required:
            if r not in required_params.keys():
                raise KeyError('Please specify %s since it is a required parameter' % r)
        return True

    def attach(self, netspec, bottom):
        '''
        Takes in params and makes the caffe layer object.
        @param caffenet: The caffe network specification object on which modules will be attached
        @param bottom: List of the bottom layers needed
        @return: Caffe Layer object modules inside caffe.layers
        '''
        params = self._construct_params()
        layername = getattr(L, self._type_name)
        layer = layername(*bottom, ** params)
        netspec[params['name']] = layer
        return layer
