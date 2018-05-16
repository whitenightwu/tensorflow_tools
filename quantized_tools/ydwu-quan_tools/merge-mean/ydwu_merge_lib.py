# pylint: disable=g-bad-file-header
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Folding batch normalization ops into the pre-calculated weights.

 - Fusing common operations into unified versions.

This script takes a frozen GraphDef file (where the weight variables have been
converted into constants by the freeze_graph script) and outputs a new GraphDef
with the optimizations applied.

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input_graph=some_graph_def.pb \
--output_graph=/tmp/optimized_graph.pb \
--input_names=Mul \
--output_names=softmax

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import re
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib

flags = flags_lib
FLAGS = flags.FLAGS

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Support folding two types of batch norm ops:
# BatchNormWithGlobalNormalization and FusedBatchNorm.  The two types only
# differ in input order and attribute names, so we've collected their
# differences up front.
INPUT_ORDER = {
    # Order of inputs for BatchNormWithGlobalNormalization.
    "BatchNormWithGlobalNormalization": [
        "conv_op", "mean_op", "var_op", "beta_op", "gamma_op"
    ],
    # Order of inputs for FusedBatchNorm.
    "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"]
}
# Name of the attribute epsilon value is stored in.
EPSILON_ATTR = {
    "BatchNormWithGlobalNormalization": "variance_epsilon",
    "FusedBatchNorm": "epsilon"
}


def optimize_for_inference(input_graph_def, input_node_names, output_node_names,
                           placeholder_type_enum, 
                           merge_bn, merge_bn_mode, 
                           merge_pre, input_mean, input_std):
  """Applies a series of inference optimizations on the input graph.

  Args:
    input_graph_def: A GraphDef containing a training model.
    input_node_names: A list of names of the nodes that are fed inputs during
      inference.
    output_node_names: A list of names of the nodes that produce the final
      results.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.

  Returns:
    An optimized version of the input graph.
  """
  ensure_graph_is_valid(input_graph_def)
  optimized_graph_def = input_graph_def
  optimized_graph_def = strip_unused_lib.strip_unused(
      optimized_graph_def, input_node_names, output_node_names,
      placeholder_type_enum)
  optimized_graph_def = graph_util.remove_training_nodes(
      optimized_graph_def, output_node_names)

  if merge_bn:
    if merge_bn_mode == "origine":
      optimized_graph_def = fold_batch_norms(optimized_graph_def)
    if merge_bn_mode == "ydwu1":
      optimized_graph_def = ydwu1_fold_batch_norms(optimized_graph_def)
    if merge_bn_mode == "ydwu2":
      optimized_graph_def = ydwu2_fold_batch_norms(optimized_graph_def)

  if merge_pre:
    optimized_graph_def = ydwu_merge_pre(optimized_graph_def, input_node_names, 
                                         input_mean, input_std)
  # optimized_graph_def = fuse_resize_and_conv(optimized_graph_def,
  #                                            output_node_names)
  ensure_graph_is_valid(optimized_graph_def)
  return optimized_graph_def


def ensure_graph_is_valid(graph_def):
  """Makes sure that the graph is internally consistent.

  Checks basic properties of the graph def and raises an exception if there are
  input references to missing nodes, duplicated names, or other logic errors.

  Args:
    graph_def: Definition of a graph to be checked.

  Raises:
    ValueError: If the graph is incorrectly constructed.
  """
  node_map = {}
  for node in graph_def.node:
    if node.name not in node_map.keys():
      node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)
  for node in graph_def.node:
    for input_name in node.input:
      input_node_name = node_name_from_input(input_name)
      if input_node_name not in node_map.keys():
        raise ValueError("Input for ", node.name, " not found: ", input_name)


def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name


def node_from_map(node_map, name):
  """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
  stripped_name = node_name_from_input(name)
  if stripped_name not in node_map:
    raise ValueError("No node named '%s' found in map." % name)
  return node_map[stripped_name]


def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Node named '%s' should be a Const op for values_from_const." %
        node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  # shape = tensor_value.shape
  # print(shape)

  return tensor_value


# Whether to scale by gamma after normalization.
def scale_after_normalization(node):
  if node.op == "BatchNormWithGlobalNormalization":
    return node.attr["scale_after_normalization"].b
  return True


def fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only spots batch
  normalization that's performed by the BatchNormWithGlobalNormalization op, and
  will need to be extended in the future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  nodes_to_skip = {}
  new_ops = []
  for node in input_graph_def.node:
    if node.op not in ("BatchNormWithGlobalNormalization", "FusedBatchNorm"):
      continue

    # "conv_op"
    conv_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("conv_op")])
    print(conv_op.input[1])
    if conv_op.op != "Conv2D":
      tf_logging.warning(
          "Didn't find expected Conv2D input to '%s'" % node.name)
      continue
################## error
    print("ydwu === conv-ok")

    weights_op = node_from_map(input_node_map, conv_op.input[1])
    if weights_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (conv_op.name, weights_op))
      continue
    weights = values_from_const(weights_op)
    channel_count = weights.shape[3]

    # "mean_op"
    mean_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("mean_op")])
    if mean_op.op != "Const":
      tf_logging.warning("Didn't find expected mean Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, mean_op))
      continue
    mean_value = values_from_const(mean_op)
    if mean_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for mean, found %s, expected %s,"
                         " for node %s" % (str(mean_value.shape), str(
                             (channel_count,)), node.name))
      continue

    # "var_op"
    var_op = node_from_map(input_node_map,
                           node.input[INPUT_ORDER[node.op].index("var_op")])
    if var_op.op != "Const":
      tf_logging.warning("Didn't find expected var Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, var_op))
      continue
    var_value = values_from_const(var_op)
    if var_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for var, found %s, expected %s,"
                         " for node %s" % (str(var_value.shape), str(
                             (channel_count,)), node.name))
      continue

    # "beta_op"
    beta_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("beta_op")])
    if beta_op.op != "Const":
      tf_logging.warning("Didn't find expected beta Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, beta_op))
      continue
    beta_value = values_from_const(beta_op)
    if beta_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for beta, found %s, expected %s,"
                         " for node %s" % (str(beta_value.shape), str(
                             (channel_count,)), node.name))
      continue

    # "gamma_op"
    gamma_op = node_from_map(input_node_map,
                             node.input[INPUT_ORDER[node.op].index("gamma_op")])
    if gamma_op.op != "Const":
      tf_logging.warning("Didn't find expected gamma Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, gamma_op))
      continue
    gamma_value = values_from_const(gamma_op)
    if gamma_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for gamma, found %s, expected %s,"
                         " for node %s" % (str(gamma_value.shape), str(
                             (channel_count,)), node.name))
      continue

    variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
    nodes_to_skip[node.name] = True
    nodes_to_skip[weights_op.name] = True
    nodes_to_skip[mean_op.name] = True
    nodes_to_skip[var_op.name] = True
    nodes_to_skip[beta_op.name] = True
    nodes_to_skip[gamma_op.name] = True
    nodes_to_skip[conv_op.name] = True

    if scale_after_normalization(node):
      scale_value = (
          (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
          gamma_value)
    else:
      scale_value = (
          1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value))
    offset_value = (-mean_value * scale_value) + beta_value
    scaled_weights = np.copy(weights)
    it = np.nditer(
        scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
      current_scale = scale_value[it.multi_index[3]]
      it[0] *= current_scale
      it.iternext()

    scaled_weights_op = node_def_pb2.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = weights_op.name
    scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            scaled_weights, weights.dtype.type, weights.shape)))
    new_conv_op = node_def_pb2.NodeDef()
    new_conv_op.CopyFrom(conv_op)
    offset_op = node_def_pb2.NodeDef()
    offset_op.op = "Const"
    offset_op.name = conv_op.name + "_bn_offset"
    offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
    offset_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            offset_value, mean_value.dtype.type, offset_value.shape)))
    bias_add_op = node_def_pb2.NodeDef()
    bias_add_op.op = "BiasAdd"
    bias_add_op.name = node.name
    bias_add_op.attr["T"].CopyFrom(conv_op.attr["T"])
    bias_add_op.input.extend([new_conv_op.name, offset_op.name])
    new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def


def fuse_resize_and_conv(input_graph_def, output_node_names):
  """Merges preceding resize and mirror pad ops into a specialized convolution.

  There's a common pattern of enlarging the input to a convolution using a
  resize operation, and also using MirrorPad to extend the boundaries to that
  zero edge pixels don't bleed inwards when convolving. This routine looks for
  that pattern of operations, and fuses them together into a Conv2DWithResizeOp.

  Args:
    input_graph_def: A GraphDef containing a model.
    output_node_names: A list of names of the nodes that produce the final
      results.

  Returns:
    Modified graph with resize and pad ops merged.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """

  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  node_reference_count = collections.defaultdict(int)
  for node in input_graph_def.node:
    for input_name in node.input:
      stripped_name = node_name_from_input(input_name)
      node_reference_count[stripped_name] += 1
  for output_name in output_node_names:
    node_reference_count[output_name] += 1

  new_ops = []
  for node in input_graph_def.node:

    if node.op != "Conv2D":
      continue
    conv_op = node

    input_op = node_from_map(input_node_map, conv_op.input[0])
    if input_op.op == "MirrorPad":
      mirror_pad_op = input_op
      resize_op = node_from_map(input_node_map, mirror_pad_op.input[0])
      if resize_op.op != "ResizeBilinear":
        resize_op = None
    else:
      mirror_pad_op = None
      if input_op.op == "ResizeBilinear":
        resize_op = input_op
      else:
        resize_op = None

    # There are no ops to be fused into the conv, so skip replacing this one.
    if not mirror_pad_op and not resize_op:
      continue

    # We're replacing this node, so make sure the old one is removed.
    node_reference_count[conv_op.name] = 0
    if mirror_pad_op:
      node_reference_count[mirror_pad_op.name] -= 1
    if resize_op:
      node_reference_count[resize_op.name] -= 1

    fused_conv_op = node_def_pb2.NodeDef()
    if resize_op:
      fused_conv_op.op = "FusedResizeAndPadConv2D"
    else:
      fused_conv_op.op = "FusedPadConv2D"
    fused_conv_op.name = conv_op.name
    if mirror_pad_op:
      mirror_paddings_name = mirror_pad_op.input[1]
      mirror_paddings_mode = mirror_pad_op.attr["mode"]
    else:
      # If there was no MirrorPad op, then create settings that make the padding
      # stage of the fused operation a no-op.
      paddings_op = node_def_pb2.NodeDef()
      paddings_op.op = "Const"
      paddings_op.name = conv_op.name + "_dummy_paddings"
      paddings_op.attr["dtype"].CopyFrom(
          attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
      paddings_op.attr["value"].CopyFrom(
          attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
              [0, 0, 0, 0, 0, 0, 0, 0], dtypes.int32, [4, 2])))
      new_ops.extend([paddings_op])
      mirror_paddings_name = paddings_op.name
      mirror_paddings_mode = attr_value_pb2.AttrValue(s=b"REFLECT")
    if resize_op:
      fused_conv_op.input.extend([
          resize_op.input[0], resize_op.input[1], mirror_paddings_name,
          conv_op.input[1]
      ])
      fused_conv_op.attr["resize_align_corners"].CopyFrom(
          resize_op.attr["align_corners"])
    else:
      fused_conv_op.input.extend(
          [mirror_pad_op.input[0], mirror_paddings_name, conv_op.input[1]])
    fused_conv_op.attr["T"].CopyFrom(conv_op.attr["T"])
    fused_conv_op.attr["mode"].CopyFrom(mirror_paddings_mode)
    fused_conv_op.attr["strides"].CopyFrom(conv_op.attr["strides"])
    fused_conv_op.attr["padding"].CopyFrom(conv_op.attr["padding"])
    new_ops.extend([fused_conv_op])

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node_reference_count[node.name] < 1:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def



def ydwu2_fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only spots batch
  normalization that's performed by the BatchNormWithGlobalNormalization op, and
  will need to be extended in the future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """

  print("ydwu === ready for merge BN with conv! ")
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  channel_count = {}
  nodes_to_skip = {}
  new_ops = []
  value_consts = {}
  old_ops = {}
  is_relu = False
  int32_type = None
  for node in input_graph_def.node:
    if "ExpandDims/dim" in node.name:
        int32_type = node.attr["dtype"]
    split_pattern = re.compile(r'/')
    # print(split_pattern)
    name_list = split_pattern.split(node.name)
    # print(name_list)

    if node.op == "Placeholder":
      print(node.name)
      continue
    if node.op == "ExpandDims":
      print(node.name)
      continue
    if node.op == "Reshape":
      print(node.name)
      continue
    if node.op == "ArgMax":
      print(node.name)
      continue
    if node.op == "Shape":
      print(node.name)
      continue
    if node.op == "Softmax":
      print(node.name)
      continue
    if node.op == "Squeeze":
      print(node.name)
      continue

    if name_list[-2] == "batchnorm":
      nodes_to_skip[node.name] = True
    if name_list[-2] == "BatchNorm":
      nodes_to_skip[node.name] = True

    # if name_list[-1] == "convolution":
    #     nodes_to_skip[node.name] = True
    # if name_list[-1] == "depthwise":
    #     nodes_to_skip[node.name] = True
    if name_list[-1] == "Relu" or name_list[-1] == "Relu6":
        old_ops["relu"] = node
        is_relu = True
        # nodes_to_skip[node.name] = True

    print(node.name)

    # "y"
    if name_list[-1] == "y":
      # print("ydwu ==== y")
      y_value = values_from_const(node)
      old_ops["y_op"] = node
      # nodes_to_skip[node.name] = True
      value_consts["y_value"] = y_value

    # "gamma"
    if name_list[-1] == "gamma":
      # print("ydwu ==== gamma")
      gamma = values_from_const(node)
      old_ops["gamma_op"] = node
      # nodes_to_skip[node.name] = True
      value_consts["gamma"] = gamma

    # "conv"
    if node.op in ("Conv2D" ,"DepthwiseConv2dNative"):
      old_ops["conv_op"] = node
      # print("old_ops = ", old_ops.keys())
      # print("old_Conv = ", old_ops["conv_op"].name)
      weights_op = node_from_map(input_node_map, node.input[1])
      # print(conv_op.input[1])
      old_ops["weights_op"] = weights_op
      if weights_op.op != "Const":
        tf_logging.warning("Didn't find expected bias Constant input to '%s',"
                           "found %s instead. Maybe because freeze_graph wasn't"
                           " run first?" % (conv_op, weights_op))
        continue
      weights = values_from_const(weights_op)
      value_consts["weights"] = weights
      if node.op == 'DepthwiseConv2dNative':
        value_consts["channel"] = 2
        channel_count = weights.shape[2]
      else:
        value_consts["channel"] = 3
        channel_count = weights.shape[3]    
      # nodes_to_skip[node.name] = True
      # nodes_to_skip[weights_op.name] = True

    # "BiasAdd"
    if node.op == "BiasAdd":
      # if old_ops.has_key('conv_op'):
      old_ops["biasadd_op"] = node

      # print("old_ops = ", old_ops.keys())
      # print("old_BiasAdd = ", old_ops["old_BiasAdd"].name)
      bias_op = node_from_map(input_node_map, node.input[1])
      old_ops["bias_op"] = bias_op
      if bias_op.op != "Const":
        tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                           " found %s instead. Maybe because freeze_graph wasn't"
                           " run first?" % (old_BiasAdd_op.name, bias_op))
        continue
      bias_value = values_from_const(bias_op)
      value_consts["bias"] = bias_value
      # nodes_to_skip[node.name] = True
      # nodes_to_skip[bias_op.name] = True


    # "Const"
    if node.op == "Const":
        const_split_pattern = re.compile(r'/')
        const_name_list = const_split_pattern.split(node.name)
        if const_name_list[-1] == "moving_variance":
            # print("ydwu ==== variance")
            moving_variance = values_from_const(node)
            old_ops["variance_op"] = node
            value_consts["moving_variance"] = moving_variance
            nodes_to_skip[node.name] = True
        if const_name_list[-1] == "moving_mean":
            # print("ydwu ==== mean")
            moving_mean = values_from_const(node)
            old_ops["mean_op"] = node
            value_consts["moving_mean"] = moving_mean
            nodes_to_skip[node.name] = True
        if const_name_list[-1] == "beta":
            # print("ydwu ==== beta")
            beta = values_from_const(node)
            old_ops["beta_op"] = node
            nodes_to_skip[node.name] = True
            value_consts["beta"] = beta


    if not is_relu:
        continue
    else:
        print("ydwu ========= relu")
        is_relu = False
        scale_value = (1.0 / np.vectorize(math.sqrt)(value_consts["moving_variance"] + value_consts["y_value"])) * value_consts["gamma"]
        if value_consts.has_key('bias'):
          offset_value = (-value_consts["moving_mean"] * scale_value) + value_consts["beta"] + scale_value * value_consts["bias"]
        else:
          offset_value = (-value_consts["moving_mean"] * scale_value) + value_consts["beta"]
        scaled_weights = np.copy(value_consts["weights"])
        it = np.nditer(scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            current_scale = scale_value[it.multi_index[value_consts["channel"]]]
            it[0] *= current_scale
            it.iternext()


### add op
        new_weights_op = node_def_pb2.NodeDef()
        new_weights_op.op = "Const"
        new_weights_op.name = old_ops["weights_op"].name
        new_weights_op.attr["dtype"].CopyFrom(old_ops["weights_op"].attr["dtype"])
        new_weights_op.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                scaled_weights, value_consts["weights"].dtype.type, value_consts["weights"].shape)))

        new_conv_op = node_def_pb2.NodeDef()
        new_conv_op.CopyFrom(old_ops["conv_op"])

        new_bias_op = node_def_pb2.NodeDef()
        new_bias_op.op = "Const"
        if value_consts.has_key('bias'):
          new_bias_op.name = old_ops["bias_op"].name
          new_bias_op.attr["dtype"].CopyFrom(old_ops["bias_op"].attr["dtype"])
          new_bias_op.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
              offset_value, value_consts["bias_op"].dtype.type, offset_value.shape)))
        else:
          offset_split_pattern = re.compile(r'/')
          offset_name_list = offset_split_pattern.split(old_ops["weights_op"].name)
          offset_name_list.remove(offset_name_list[-1])
          offset_name_list.append('bias')
          # print(offset_name_list)
          # print("offset=====", offset_name_list[-2])
          # print('/'.join(offset_name_list))

          new_bias_op.name = '/'.join(offset_name_list)
          new_bias_op.attr["dtype"].CopyFrom(old_ops["weights_op"].attr["dtype"])
          new_bias_op.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
              offset_value, value_consts["weights"].dtype.type, offset_value.shape)))


        new_biasadd_op = node_def_pb2.NodeDef()
        new_biasadd_op.op = "BiasAdd"
        if value_consts.has_key('biasadd_op'):
          new_biasadd_op.name = old_ops["biasadd_op"].name
          new_biasadd_op.attr["T"].CopyFrom(old_ops["biasadd_op"].attr["T"])
          new_biasadd_op.input.extend([new_conv_op.name, new_bias_op.name])
        else:
          biasadd_split_pattern = re.compile(r'/')
          biasadd_name_list = biasadd_split_pattern.split(old_ops["conv_op"].name)
          # print(biasadd_name_list)
          biasadd_name_list.remove(biasadd_name_list[-1])
          biasadd_name_list.append('biasadd')
          # print('/'.join(biasadd_name_list))

          new_biasadd_op.name = '/'.join(biasadd_name_list)
          new_biasadd_op.attr["T"].CopyFrom(old_ops["conv_op"].attr["T"])
          new_biasadd_op.input.extend([new_conv_op.name, new_bias_op.name])          

        new_relu_op = node_def_pb2.NodeDef()
        new_relu_op.op = old_ops["relu"].op
        new_relu_op.name = old_ops["relu"].name
        new_relu_op.attr["T"].CopyFrom(old_ops["relu"].attr["T"])
        new_relu_op.input.extend([new_biasadd_op.name])

        new_ops.extend([new_weights_op, new_conv_op, new_bias_op, new_biasadd_op, new_relu_op])

        for yy_key in old_ops.keys():
          # print(old_ops[yy_key].name)
          nodes_to_skip[old_ops[yy_key].name] = True
        value_consts = {}
        old_ops.clear()


  print("ydwu === finally result")
  print("================================================")
  for aaa in new_ops[:]:
    print(aaa.op)
  print("================================================")
  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      # print(node.name)
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    print(new_node.name)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def




# Support folding two types of batch norm ops:
# BatchNormWithGlobalNormalization and FusedBatchNorm.  The two types only
# differ in input order and attribute names, so we've collected their
# differences up front.
YDWU_INPUT_ORDER = {
    # Order of inputs for BatchNormWithGlobalNormalization.
    "BatchNormWithGlobalNormalization": [
        "BN_input_op", "mean_op", "var_op", "beta_op", "gamma_op"
    ],
    # Order of inputs for FusedBatchNorm.
    "FusedBatchNorm": ["BN_input_op", "gamma_op", "beta_op", "mean_op", "var_op"]
}
# Name of the attribute epsilon value is stored in.
YDWU_EPSILON_ATTR = {
    "BatchNormWithGlobalNormalization": "variance_epsilon",
    "FusedBatchNorm": "epsilon"
}


# Whether to scale by gamma after normalization.
def ydwu_merge_BN_with_conv(node, value_consts, nodes_to_skip, new_ops, old_ops, input_node_map):
  # "input_op"
  input_op = node_from_map(input_node_map,
                           node.input[YDWU_INPUT_ORDER[node.op].index("BN_input_op")])
  # if input_op.op != "Const":
  #   tf_logging.warning("Didn't find expected mean Constant input to '%s',"
  #                      " found %s instead. Maybe because freeze_graph wasn't"
  #                      " run first?" % (node.name, input_op))
  #   continue
  #   mean_value = values_from_const(input_op)
  #   if mean_value.shape != (channel_count,):
  #     tf_logging.warning("Incorrect shape for mean, found %s, expected %s,"
  #                        " for node %s" % (str(mean_value.shape), str(
  #                          (channel_count,)), node.name))
  #     continue
  channel_count = {}

  # "old_Conv_op"
  if old_ops.has_key('old_Conv'):
    old_Conv_op = old_ops["old_Conv"]        
    weights_op = node_from_map(input_node_map, old_Conv_op.input[1])
    if weights_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (old_Conv_op.name, weights_op))
      return False
    weights = values_from_const(weights_op)
    if old_Conv_op.op == 'DepthwiseConv2dNative':
      value_consts["channel"] = 2
      channel_count = weights.shape[2]
    else:
      value_consts["channel"] = 3
      channel_count = weights.shape[3]    

  # "old_BiasAdd_op"
  if old_ops.has_key('old_BiasAdd'):
    old_BiasAdd_op = old_ops['old_BiasAdd']
    bias_op = node_from_map(input_node_map, old_BiasAdd_op.input[1])
    if bias_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (old_BiasAdd_op.name, bias_op))
      return False
    bias_value = values_from_const(bias_op)

  # "mean_op"
  mean_op = node_from_map(input_node_map,
                          node.input[YDWU_INPUT_ORDER[node.op].index("mean_op")])
  if mean_op.op != "Const":
    tf_logging.warning("Didn't find expected mean Constant input to '%s',"
                       " found %s instead. Maybe because freeze_graph wasn't"
                       " run first?" % (node.name, mean_op))
    return False
  mean_value = values_from_const(mean_op)
  if mean_value.shape != (channel_count,):
    tf_logging.warning("Incorrect shape for mean, found %s, expected %s,"
                       " for node %s" % (str(mean_value.shape), str(
                         (channel_count,)), node.name))
    return False

  # "var_op"
  var_op = node_from_map(input_node_map,
                         node.input[YDWU_INPUT_ORDER[node.op].index("var_op")])
  if var_op.op != "Const":
    tf_logging.warning("Didn't find expected var Constant input to '%s',"
                       " found %s instead. Maybe because freeze_graph wasn't"
                       " run first?" % (node.name, var_op))
    return False
  var_value = values_from_const(var_op)
  if var_value.shape != (channel_count,):
    tf_logging.warning("Incorrect shape for var, found %s, expected %s,"
                       " for node %s" % (str(var_value.shape), str(
                         (channel_count,)), node.name))
    return False

  # "beta_op"
  beta_op = node_from_map(input_node_map,
                          node.input[YDWU_INPUT_ORDER[node.op].index("beta_op")])
  if beta_op.op != "Const":
    tf_logging.warning("Didn't find expected beta Constant input to '%s',"
                       " found %s instead. Maybe because freeze_graph wasn't"
                       " run first?" % (node.name, beta_op))
    return False
  beta_value = values_from_const(beta_op)
  if beta_value.shape != (channel_count,):
    tf_logging.warning("Incorrect shape for beta, found %s, expected %s,"
                       " for node %s" % (str(beta_value.shape), str(
                         (channel_count,)), node.name))
    return False

  # "gamma_op"
  gamma_op = node_from_map(input_node_map,
                           node.input[YDWU_INPUT_ORDER[node.op].index("gamma_op")])
  if gamma_op.op != "Const":
    tf_logging.warning("Didn't find expected gamma Constant input to '%s',"
                       " found %s instead. Maybe because freeze_graph wasn't"
                       " run first?" % (node.name, gamma_op))
    return False
  gamma_value = values_from_const(gamma_op)
  if gamma_value.shape != (channel_count,):
    tf_logging.warning("Incorrect shape for gamma, found %s, expected %s,"
                       " for node %s" % (str(gamma_value.shape), str(
                         (channel_count,)), node.name))
    return False

  variance_epsilon_value = node.attr[YDWU_EPSILON_ATTR[node.op]].f
  nodes_to_skip[node.name] = True
  nodes_to_skip[weights_op.name] = True
  nodes_to_skip[bias_op.name] = True
  nodes_to_skip[mean_op.name] = True
  nodes_to_skip[var_op.name] = True
  nodes_to_skip[beta_op.name] = True
  nodes_to_skip[gamma_op.name] = True

  if scale_after_normalization(node):
    # FusedBatchNorm
    scale_value = (
      (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
      gamma_value)
  else:
    # BatchNormWithGlobalNormalization
    scale_value = (
      1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value))
  offset_value = (bias_value - mean_value) * scale_value + beta_value
  scaled_weights = np.copy(weights)
  it = np.nditer(
    scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
  while not it.finished:
    current_scale = scale_value[it.multi_index[value_consts["channel"]]]
    it[0] *= current_scale
    it.iternext()

  ## add new op.
  scaled_weights_op = node_def_pb2.NodeDef()
  scaled_weights_op.op = "Const"
  scaled_weights_op.name = weights_op.name
  scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
  scaled_weights_op.attr["value"].CopyFrom(
    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
      scaled_weights, weights.dtype.type, weights.shape)))

  new_conv_op = node_def_pb2.NodeDef()
  new_conv_op.CopyFrom(old_Conv_op)

  offset_op = node_def_pb2.NodeDef()
  offset_op.op = "Const"
  offset_op.name = bias_op.name # + "_bn_offset"
  offset_op.attr["dtype"].CopyFrom(bias_op.attr["dtype"])
  offset_op.attr["value"].CopyFrom(
    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
      offset_value, mean_value.dtype.type, offset_value.shape)))
  # print(offset_value.shape)

  new_bias_add_op = node_def_pb2.NodeDef()
  new_bias_add_op.op = "BiasAdd"
  new_bias_add_op.name = old_BiasAdd_op.name
  new_bias_add_op.attr["T"].CopyFrom(old_BiasAdd_op.attr["T"])
  new_bias_add_op.input.extend([new_conv_op.name, offset_op.name])

  new_ops.extend([scaled_weights_op, new_conv_op, offset_op, new_bias_add_op])

  return True




def ydwu1_fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only spots batch
  normalization that's performed by the BatchNormWithGlobalNormalization op, and
  will need to be extended in the future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)
  
  value_consts = {}
  nodes_to_skip = {}
  new_ops = []
  old_ops = {}
  for node in input_graph_def.node:
    # print("======== current node = ", node.name)

    if node.op == "Relu" or node.op == "Relu6":
      print("Relu")
      new_relu_op = node_def_pb2.NodeDef()
      new_relu_op.op = node.op
      new_relu_op.name = node.name
      new_relu_op.attr["T"].CopyFrom(node.attr["T"])
      print(old_ops["old_BiasAdd"].name)
      new_relu_op.input.extend([old_ops["old_BiasAdd"].name])

      old_ops.clear()
      print("Relu = ", old_ops.keys())
      nodes_to_skip[node.name] = True
      new_ops.extend([new_relu_op])
      continue

    if node.op == "BiasAdd":
      if old_ops.has_key('old_Conv'):
        old_ops["old_BiasAdd"] = node
        nodes_to_skip[node.name] = True
        print("old_ops = ", old_ops.keys())
        print("old_BiasAdd = ", old_ops["old_BiasAdd"].name)
        continue
      # else:
      #   continue

    if node.op in ("Conv2D" ,"DepthwiseConv2dNative"):
    # if node.op in ("Conv2D" ,"DepthwiseConv2dNative", "MatMul"):
      old_ops["old_Conv"] = node
      nodes_to_skip[node.name] = True
      print("old_ops = ", old_ops.keys())
      print("old_Conv = ", old_ops["old_Conv"].name)
      continue

    if node.op in ("BatchNormWithGlobalNormalization", "FusedBatchNorm"):      
      ydwu_merge_BN_with_conv(node, value_consts, nodes_to_skip, new_ops, old_ops, input_node_map)
      continue

### TODO: not "mulmat",

    # if node.op == "Reshape":
    #   print("Reshape")
    #   new_Reshape_op = node_def_pb2.NodeDef()
    #   new_Reshape_op.op = node.op
    #   new_Reshape_op.name = node.name
    #   new_Reshape_op.attr["T"].CopyFrom(node.attr["T"])
    #   print(old_ops["old_BiasAdd"].name)
    #   new_Reshape_op.input.extend([old_ops["old_BiasAdd"].name])

    #   old_ops.clear()
    #   # print("Reshape = ", old_ops.keys())
    #   nodes_to_skip[node.name] = True
    #   new_ops.extend([new_Reshape_op])
    #   continue
      
    

  print("==================================================")
  # print("result_graph_def")

  # print("copy_ops")
  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    # print(new_node.op)
    result_graph_def.node.extend([new_node])

  # print("==================================================")
  # print("new_ops")
  # for aaa in new_ops[:]:
  #   print(aaa.op)

  result_graph_def.node.extend(new_ops)
  return result_graph_def












def merge_mean_and_std(input_node_map, nodes_to_skip, old_ops, new_ops, input_mean, input_std):

  # "old_Conv_op"
  if old_ops.has_key('old_Conv'):
    old_Conv_op = old_ops["old_Conv"]        
    weights_op = node_from_map(input_node_map, old_Conv_op.input[1])
    if weights_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (old_Conv_op.name, weights_op))
      return False
    weights = values_from_const(weights_op)
    print(weights.shape[3])

  # "old_BiasAdd_op"
  if old_ops.has_key('old_BiasAdd'):
    old_BiasAdd_op = old_ops["old_BiasAdd"]
    bias_op = node_from_map(input_node_map, old_BiasAdd_op.input[1])
    if bias_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (old_BiasAdd_op.name, bias_op))
      return False
    bias_value = values_from_const(bias_op)
    # print("bias_value=", bias_value)

  nodes_to_skip[weights_op.name] = True
  nodes_to_skip[bias_op.name] = True


  ### comput new-weights and bias
  ## new-weight
  scaled_weights = np.copy(weights)
  # print(scaled_weights[0])
  scaled_weights /= input_std
  # print(scaled_weights[0])

  ## new-bias
  weights_sum = np.copy(weights)
  for i in range(0, 2):
    # print(i, "=", weights)
    weights_sum = weights_sum.sum(axis=0)

  # print(np.array(input_mean))
  input_mean = np.tile(input_mean, (weights.shape[3], 1))
  input_mean = input_mean.transpose()
  print(input_mean.shape)
  print(input_mean)

  print(weights_sum.shape)
  print("axis32 = ", weights_sum)
  weights_sum = weights_sum * input_mean
  print("axis32 = ", weights_sum)
  weights_sum = weights_sum.sum(axis=0)
  print(weights_sum)
  offset_value = bias_value - (weights_sum / input_std)
  # offset_value = bias_value - (weights_sum * input_mean / input_std)
  # print("offset_value=", offset_value)


  ## add new op.
  new_weights_op = node_def_pb2.NodeDef()
  new_weights_op.op = "Const"
  new_weights_op.name = weights_op.name
  new_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
  new_weights_op.attr["value"].CopyFrom(
    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
      scaled_weights, weights.dtype.type, weights.shape)))

  new_bias_op = node_def_pb2.NodeDef()
  new_bias_op.op = "Const"
  new_bias_op.name = bias_op.name
  new_bias_op.attr["dtype"].CopyFrom(bias_op.attr["dtype"])
  new_bias_op.attr["value"].CopyFrom(
    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
      offset_value, bias_value.dtype.type, bias_value.shape)))

  new_ops.extend([new_weights_op, new_bias_op])

  return new_ops



def ydwu_merge_pre(input_graph_def, input_node_names, input_mean, input_std):
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  print("==================================================")
  print("ydwu_merge_pre")
  print("==================================================")
  print("input_mean = ", input_mean)
  print("input_std = ", input_std)
  input_mean = map(eval, input_mean)  
  print(input_mean)

  print(input_node_names)
  input_node_names = ''.join(input_node_names)
  print(input_node_names)

  # input_mean = [127, 127, 127]
  # input_std = 127

  value_consts = {}
  nodes_to_skip = {}
  new_ops = []
  old_ops = {}
  for node in input_graph_def.node:
    print("======== current node = ", node.name)

    if node.op in ("Conv2D" ,"DepthwiseConv2dNative") and not old_ops.has_key('old_Conv') and node.input[0] == input_node_names:
      old_ops["old_Conv"] = node
      print("old_Conv = ", old_ops["old_Conv"].name)
      print(node.input[1])
      continue

    if node.op == "BiasAdd":
      if old_ops.has_key('old_Conv'):
        old_ops["old_BiasAdd"] = node
        print("old_BiasAdd = ", old_ops["old_BiasAdd"].name)
        new_ops = merge_mean_and_std(input_node_map, nodes_to_skip, old_ops, new_ops, input_mean, input_std)
        break

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  print("==================================================")
  for aaa in new_ops[:]:
    print(aaa.op)

  result_graph_def.node.extend(new_ops)
  return result_graph_def
