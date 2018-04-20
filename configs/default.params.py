Convolution = dict(
				weight_filler=dict(type='xavier'),
				bias_filler=dict(type='constant', value=0),
				engine=0,
				param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
				)
				
BatchNorm = dict(
				param=[dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
				in_place=True
				)

InnerProduct = dict(
				weight_filler= dict(type='xavier'),
				bias_filler= dict(type='constant'),
				param= [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
				)

BN = dict(
		param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
		slope_filler=dict(type='constant', value=1),
		bias_filler=dict(type='constant', value=0),
		engine=2,
		in_place=False
		)

SyncBN = dict(
		param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
		bn_param=dict(
			slope_filler=dict(type='constant', value=1),
			bias_filler=dict(type='constant', value=0)),
		in_place=False
		)

Scale = dict(bias_term=True, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)], in_place=True,)

ReLU = dict(in_place=True)

Dropout = dict(in_place=True)