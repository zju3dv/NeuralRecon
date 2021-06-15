# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import numpy as np
import os


def visualize(fname):
    key_names = ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete', 'dist1', 'dist2',
                 'prec', 'recal', 'fscore']

    metrics = json.load(open(fname, 'r'))
    metrics = sorted([(scene, metric) for scene, metric in metrics.items()], key=lambda x: x[0])
    scenes = [m[0] for m in metrics]
    metrics = [m[1] for m in metrics]

    keys = metrics[0].keys()
    metrics1 = {m: [] for m in keys}
    for m in metrics:
        for k in keys:
            metrics1[k].append(m[k])

    for k in key_names:
        if k in metrics1:
            v = np.nanmean(np.array(metrics1[k]))
        else:
            v = np.nan
        print('%10s %0.3f' % (k, v))


def main():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to metrics file")
    args = parser.parse_args()

    rslt_file = os.path.join(args.model, 'metrics.json')
    visualize(rslt_file)


if __name__ == "__main__":
    main()
