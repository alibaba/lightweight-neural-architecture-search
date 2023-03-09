# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import logging
import os

import numpy as np


def parse_data_point(line):
    # example: {Regular,1,16,7,7,16,1,1,0} 0.123
    conv_config, time = line.split()
    conv_config = conv_config.strip('{')
    conv_config = conv_config.strip('}')
    conv_type, batch, inputC, inputH, inputW, outputC, K, stride, elmtFused = conv_config.split(
        ',')
    conv_key = conv_type + ' ' + stride + ' ' + elmtFused + ' ' + K
    batch = int(batch)
    inputC = int(inputC)
    outputC = int(outputC)
    inputH = int(inputH)
    time = float(time)
    return conv_key, batch, inputC, outputC, inputH, time


def parse_conv_key(conv_key):
    conv_type, stride, elmtFused, K = conv_key.split()
    return conv_type, int(stride), int(elmtFused), int(K)


def preprocess_regular_conv_data(database):
    for key in database:
        conv_type, stride, elmtFused, K = parse_conv_key(key)
        if conv_type != 'Regular':
            continue
        for batch in database[key]:
            grid = {}
            points = database[key][batch][0]
            values = database[key][batch][1]
            for i in range(len(points)):
                inputC, outputC, inputH = points[i]
                value = values[i]
                ratio = inputC / outputC
                # special case: group inputC == 3 datapoints together
                if inputC == 3:
                    ratio = 0

                ratio = int(ratio * 100) / 100
                if ratio not in grid:
                    grid[ratio] = ([], [])
                grid[ratio][0].append((outputC, inputH))
                grid[ratio][1].append(value)
            database[key][batch] = grid


def readDataBase(filepath, logger=None):
    file = open(filepath)
    lines = file.readlines()
    file.close()
    database = {}
    for line in lines:
        line = line.strip('\n')
        conv_key, batch, inputC, outputC, inputH, time = parse_data_point(line)
        conv_type, stride, elmtFused, K = parse_conv_key(conv_key)

        if inputH == 4:
            continue
        if time < 0:
            # logging.debug('skip %s because profiling failed' % line)
            continue
        if conv_key not in database:
            database[conv_key] = {}
        if batch not in database[conv_key]:
            database[conv_key][batch] = ([], [])
        if conv_type == 'Regular':
            database[conv_key][batch][0].append((inputC, outputC, inputH))
        else:
            # b/c inputC == outputC
            database[conv_key][batch][0].append((outputC, inputH))
        database[conv_key][batch][1].append(time)

    logger.debug('DATA LOAD SUCCESSFULLY')
    preprocess_regular_conv_data(database)
    return database


def filter(test, realtime):
    conv_type, stride, elmt, K, batch, inputC, inputH, outputC = test
    if realtime is not None:
        if realtime > 15:
            return False
    if (conv_type == 'Depthwise'):
        if K > 5:
            return False
    return True


def readTestFile(file_name):
    file = open(file_name)
    lines = file.readlines()
    tests = []
    times = []
    for line in lines:
        conv_key, batch, inputC, outputC, inputH, time = parse_data_point(line)
        conv_type, stride, elmt, K = parse_conv_key(conv_key)
        test = (conv_type, stride, elmt, K, batch, inputC, inputH, outputC)
        if time < 0:
            continue
        tests.append(test)
        times.append(time)
    return tests, times


def cmp_fusion(database):
    for key in database:
        conv_type, stride, elmtFused, K = parse_conv_key(key)
        if conv_type != 'Regular' or elmtFused == 1:
            continue
        cmp_key = conv_type + ' ' + str(stride) + ' ' + str(1) + ' ' + str(K)
        for batch in database[key]:
            for i in range(len(database[key][batch][0])):
                time1 = database[key][batch][1][i]
                inputC, outputC, inputH = database[key][batch][0][i]
                time2 = None
                for j in range(len(database[cmp_key][batch][0])):
                    cmp_inputC, cmp_outputC, cmp_inputH = database[cmp_key][
                        batch][0][j]
                    if (cmp_inputC == inputC) and (cmp_outputC
                                                   == outputC) and (cmp_inputH
                                                                    == inputH):
                        time2 = database[cmp_key][batch][1][j]
                        break
                print(key, batch, database[key][batch][0][i], time1, time2)
