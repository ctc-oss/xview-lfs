from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from xview import wv_util as wv

from lfs import checkout


# from tensorflow.python.util.tf_export import keras_export
# @keras_export('keras.datasets.mnist.load_data')
def load_train_data(iids, url, ref='master', chipsz=(300, 300)):
    """
    Loads an xview tile.

    Returns:
        Dictionary mapping id -> image, boxes, classes
        That is image_id to a tuple of
          - chip images
          - chip coords
          - chip classes
    """
    iids = iids if hasattr(iids, "__iter__") else [iids]

    include = []
    for iid in iids:
        include.extend(['%s.tif' % iid, '%s.geojson' % iid])

    wd = checkout(url, ref, include, [])

    if not iids:
        iids = map(lambda n: os.path.splitext(os.path.basename(n))[0], glob.glob(os.path.join(wd, "train/*.tif")))

    splits = {}
    for iid in iids:
        tif = 'train/%s.tif' % iid
        json = 'train/%s.geojson' % iid

        tif = os.path.join(wd, tif)
        json = os.path.join(wd, json)

        arr = wv.get_image(tif)
        coords, chips, classes = wv.get_labels(json)
        splits[iid] = wv.chip_image(arr, coords, classes, chipsz)

    # wd, {id -> image, boxes, classes}
    return wd, splits


def load_classes():
    with open(os.path.join(os.path.dirname(__file__), 'xview_class_labels.txt')) as f:
        res = {}
        for l in f:
            k, v = l.strip().split(':')
            res[int(k)] = v
        return res


def fill_in_gaps_and_background(class_map):
    values = class_map.keys()
    max_class_id = max(values)

    ret = class_map.copy()
    if len(class_map) != max_class_id + 1:
        for value in range(1, max_class_id):
            if value not in values:
                ret[value] = f'class_{value}'
            else:
                ret[value] = class_map[value]

    return ret
