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

import traceback
import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.extractor import attr_getter
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.graph.graph import Node, Graph

# Number of entries: (imageId, label, conf, xmin, ymin, xmax, ymax)
_numEntries = 7


class YoloV3DetectionOutput(Op):
    op = 'Yolov3DetectionOutput'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
                'op': self.op,
                'type': self.op,
                'infer': self.infer,
                'version': '2',
            }, attrs)

    # def supported_attrs(self):
    #     print("HEREEEEEEEE 2")
    #     return [
    #         'confidence_threshold',
    #         'classes',
    #         'num_classes',
    #         'num_boxes',
    #         'biases',
    #         'masks',
    #         'anchors_scale',
    #     ]

    # def backend_attrs(self):
    #     print("HEREEEEEEEE 3")
    #     return [
    #         'confidence_threshold',
    #         'classes',
    #         'num_classes',
    #         'num_boxes',
    #         ('biases', lambda node: attr_getter(node, 'biases')),
    #         ('masks', lambda node: attr_getter(node, 'masks')),
    #         ('anchors_scale', lambda node: attr_getter(node, 'anchors_scale')),
    #     ]

    @staticmethod
    def infer(node):
        try:
            numInputs = len(node.in_nodes())

            maxNumBoxes = 0
            for n in range(numInputs):
                currentShape = node.in_node(n).shape
                maxNumBoxes += currentShape[2] * currentShape[3] * node['num_boxes']

            # Note: This only handles batch size of 1.
            # node.out_node().shape = np.array([1, 1, maxNumBoxes, _numEntries])
            node.out_port(0).data.set_shape([1, 1, maxNumBoxes, _numEntries])
        except:
            print(f"I'm in YoloV3DetectionOutput_infer and things went south .... <=========================== {traceback.format_exc()}")

