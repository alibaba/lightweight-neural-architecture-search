import logging
import numpy as np
from .op_profiler import OpProfiler
from .builder import LATENCIES

@LATENCIES.register_module(module_name = 'OpPredictor')
class OpPredictor():
    def __init__(self,  device_name = 'V100', data_type = 'FP32', batch_size = 32, image_size = 224, logger=None, **kwargs):

        self.logger = logger or logging
        self.batch_size = batch_size
        self.image_size = image_size
        self.predictor = OpProfiler(
                device_name = device_name,
                data_type = data_type,
                logger = self.logger)

    def __call__(self, model):
        try:
                net_params = model.get_params_for_trt(self.image_size)
                # remove other params, only conv and convDW
                net_params_conv = []
                for idx, net_param in enumerate(net_params):
                    if net_param[0] in ['Regular', 'Depthwise']:
                        net_params_conv.append(net_param)
                times = [0] * len(net_params_conv)

                # the unit is millisecond with batch_size, so modify it to second
                _, the_latency = self.predictor(net_params_conv, times,
                                                self.batch_size)
                the_latency = the_latency / self.batch_size / 1000
        except Exception as e:
            self.logger.error(str(e))
            the_latency = np.inf

        return the_latency

