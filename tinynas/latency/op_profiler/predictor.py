# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import logging
import math
import os
import sys
from copy import deepcopy

from scipy.interpolate import (CloughTocher2DInterpolator,
                               LinearNDInterpolator, NearestNDInterpolator,
                               griddata, interp1d)

from . import util as util


def buildInterpolator(data):
    points, values = data
    fn1 = LinearNDInterpolator(points, values)
    fn2 = NearestNDInterpolator(points, values, rescale=True)
    return (fn1, fn2)


def interpolate_DW(data, test_points):
    points, values = data
    points_2d = []
    for p in points:
        points_2d.append((p[0], p[1]))
    grid = griddata(points_2d, values, test_points, method='cubic')
    return grid[0]


def interpolate_Reg(data, test_points):
    points, values = data
    points_2d = []
    for p in points:
        points_2d.append((p[0], p[1]))
    grid = griddata(points_2d, values, test_points, method='cubic')
    return grid[0]


def buildAllInterpolators(database, logger=None):
    interps = {}
    for key in database:
        interps[key] = {}
        for batch in database[key]:
            conv_type, stride, elmtFused, K = util.parse_conv_key(key)
            if conv_type == 'Depthwise':
                interps[key][batch] = buildInterpolator(database[key][batch])
            else:
                interps[key][batch] = {}
                for ratio in database[key][batch]:
                    interps[key][batch][ratio] = buildInterpolator(
                        database[key][batch][ratio])
    logger.debug('INTERPOLATORS ARE BUILT SUCCESSFULLY')
    return interps


def predict_batch(funcs, test, x, y, logger=None):
    fn, fallback = funcs
    lat = fn([x], [y])[0]
    if math.isnan(lat) or lat < 0:
        logger.debug('nan fallback %s' % str(test))
        lat = fallback([(x, y)])[0]
    return lat


def predict_depthwise(test, database, interps, logger=None):
    conv_type, stride, elmtFused, K, batch, inputC, inputH, outputC = test
    key = '%s %s %s %s' % (conv_type, stride, elmtFused, K)
    if key not in interps:
        logger.info(test, ' skipped because \'%s\' not supported' % key)
        return -1
    if batch in interps[key]:
        lat = predict_batch(
            interps[key][batch], test, outputC, inputH, logger=logger)
    else:
        batches = []
        lats = []
        for b in interps[key]:
            batches.append(b)
            lat = predict_batch(
                interps[key][b], test, outputC, inputH, logger=logger)
            lats.append(lat)
        f = interp1d(batches, lats, fill_value='extrapolate')
        lat = f([batch])[0]
    return lat


def predict_for_ratio(interps, test, inputC, outputC, inputH, logger=None):
    map_ratios = sorted(interps.keys())
    ratios = []
    values = []
    last_value = -1

    pred_ratio = inputC / outputC
    if pred_ratio in interps:
        lat = predict_batch(
            interps[pred_ratio], test, outputC, inputH, logger=logger)
        return lat
    for r in map_ratios:
        if r == 0:
            # inputC = 3 case, skip
            continue
        value = predict_batch(interps[r], test, outputC, inputH, logger=logger)
        if len(ratios) < 2 or value > last_value:
            # we need to keep values in increasing order
            ratios.append(r)
            values.append(value)
            last_value = value
        else:
            logger.debug('skip ratio %s %s' % (r, value))
    logger.debug('ratio dimension: %s %s %s' %
                 (ratios, values, inputC / outputC))
    f = interp1d(ratios, values, fill_value='extrapolate', kind='linear')
    lat = f([pred_ratio])[0]
    if lat < 0:
        lat = pred_ratio / ratios[0] * values[0]
    return lat


def predict_regular(test, database, interps, logger=None):
    conv_type, stride, elmtFused, K, batch, inputC, inputH, outputC = test
    key = '%s %s %s %s' % (conv_type, stride, elmtFused, K)
    if key not in interps:
        logger.info(test, ' skipped because \'%s\' not supported' % key)
        return -1
    lat = -1
    if batch in interps[key]:
        if inputC == 3:
            # use special ratio "0"
            lat = predict_batch(
                interps[key][batch][0], test, outputC, inputH, logger=logger)
        else:
            lat = predict_for_ratio(
                interps[key][batch],
                test,
                inputC,
                outputC,
                inputH,
                logger=logger)
    else:
        batches = []
        lats = []
        for b in interps[key]:
            batches.append(b)
            if inputC == 3:
                lat = predict_batch(
                    interps[key][b][0], test, outputC, inputH, logger=logger)
            else:
                lat = predict_for_ratio(
                    interps[key][b],
                    test,
                    inputC,
                    outputC,
                    inputH,
                    logger=logger)
            lats.append(lat)
        logger.debug('batch dimension: %s %s' % (batches, lats))
        f = interp1d(batches, lats, fill_value='extrapolate')
        lat = f([batch])[0]
    return lat


def predict(tests,
            real_times,
            p_batch=None,
            database=None,
            interps=None,
            logger=None):
    if database is None:
        database = database_RT
    if interps is None:
        interps = interps_RT
    cmp_ret = []
    total_time = 0.0
    for i in range(len(tests)):
        t = tests[i]
        # print(t)
        if real_times is not None:
            real_time = real_times[i]
        else:
            real_time = None
        conv_type, stride, elmtFused, K, batch, inputC, inputH, outputC = t

        if p_batch is not None:
            t = (conv_type, stride, elmtFused, K, p_batch, inputC, inputH,
                 outputC)

        if (conv_type == 'Depthwise'):
            lat = predict_depthwise(t, database, interps, logger=logger)
        else:
            lat = predict_regular(t, database, interps, logger=logger)

        logger.debug('prediction %s \x1b[6;30;42m%s\x1b[0m %s' %
                     (str(t), lat, real_time))
        total_time += lat
        assert (lat >= 0)
        cmp_ret.append((t, lat, real_time))
    logger.debug('prediction total time is \x1b[6;30;42m%s\x1b[0m\n\n'
                 % total_time)
    return cmp_ret, total_time


def eval_cmp(cmp_data, logger=None):
    errors = []
    max_error = -1
    max_index = -1
    total_alert = 0
    total_error = 0.0

    s = 0.0
    ref = 0.0
    for i in range(len(cmp_data)):
        t, lat, real = cmp_data[i]
        s += lat
        ref += real
        err = lat - real
        rel = err / real
        errors.append(rel)
        total_error += abs(rel)
        if abs(rel) > max_error:
            max_error = abs(rel)
            max_index = i
            logger.info('alert %s %s' % (cmp_data[i], rel))
            total_alert += 1
    logger.info('total samples = %s' % len(errors))
    logger.info('avg error = %s' % (total_error / len(errors)))
    logger.info('max error = %s %s' % (max_error, cmp_data[max_index]))
    logger.info('total alert = %s %s' %
                (total_alert, total_alert / len(errors)))
    print('total time = %s' % s)
    print('ref total time = ', ref)


# filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "V100/conv_data.out")
# assert os.path.isfile(filepath), "Invalid profiler list for TensorRT"
# database_RT = util.readDataBase(filepath)
# interps_RT = buildAllInterpolators(database_RT)


class OpProfiler():

    def __init__(self, device_name='V100', data_type='FP32', logger=None):
        self.device_name = device_name
        self.data_type = data_type
        if data_type == 'FP32':
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '%s/conv_data.out' % (device_name))
        elif data_type == 'FP16':
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '%s/conv_data.out.fp16' % (device_name))
        elif data_type=="FP16_DAMOYOLO":
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                "%s/conv_data.out.fp16.damoyolo" % (device_name))
        elif data_type == 'INT8':
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '%s/conv_data.out.int8' % (device_name))
        elif data_type == 'INT4':
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '%s/conv_data.out.int4' % (device_name))
        assert os.path.isfile(filepath), 'Invalid profiler list for TensorRT'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging
            log_level = logging.DEBUG
            logging.basicConfig(
                stream=sys.stdout,
                level=log_level,
                format='%(levelname)s - %(message)s',
            )
        self.database_RT = util.readDataBase(filepath, logger=self.logger)
        self.interps_RT = buildAllInterpolators(
            self.database_RT, logger=self.logger)

    def revise_params_t40(self, tests):
        """
        [Channels in t40 must be a multiple of 32, Batchsize must be 1]
        """
        self.logger.debug(
            'Revise channels in T40, must be a multiple of 32!!!')
        tests_tmp = []
        for idx, test in enumerate(tests):
            test_tmp = list(deepcopy(test))
            test_tmp[2] = 0  # ElmtFused is equal to 0 for t40, not others
            test_tmp[4] = 1  # Batchsize is equal to 1 for t40, not others
            if test_tmp[5] % 32 != 0:
                test_tmp[5] = 32 * (test_tmp[5] // 32 + 1)
            if test_tmp[7] % 32 != 0:
                test_tmp[7] = 32 * (test_tmp[7] // 32 + 1)
            self.logger.debug('the process of idx-%d:\n%s-->\n%s<--' %
                              (idx, test, test_tmp))
            tests_tmp.append(test_tmp)
        return tests_tmp

    def __call__(self, tests, real_times, p_batch=128):
        if 'T40' in self.device_name:
            if p_batch != 1:
                raise ValueError(
                    'the batchsize for predict latency for T40 must be 1, not %d'
                    % (p_batch))
            tests = self.revise_params_t40(tests)

        cmp_ret, total_time = predict(
            tests,
            real_times,
            p_batch=p_batch,
            database=self.database_RT,
            interps=self.interps_RT,
            logger=self.logger)
        return cmp_ret, total_time


if __name__ == '__main__':
    pass
