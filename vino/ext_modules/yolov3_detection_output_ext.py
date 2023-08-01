"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

import logging as log

from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.front.caffe.yolov3detectionoutput import YoloV3DetectionOutput


class YoloV3DetectionOutputExtractor(FrontExtractorOp):
    op = 'Yolov3DetectionOutput'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.yolov3_detection_output_param

        update_attrs = {
            'num_classes': param.num_classes,
            'confidence_threshold': param.confidence_threshold,
            'num_boxes': param.num_boxes,
            'biases': np.array(param.biases),
            'masks': np.array(param.masks),
            'anchors_scale': np.array(param.anchors_scale),
        }

        mapping_rule = merge_attrs(param, update_attrs)

        # update the attributes of the node
        YoloV3DetectionOutput.update_node_stat(node, mapping_rule)
        return True

