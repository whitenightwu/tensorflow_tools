/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

  // white
      // {"QuantizedBiasAdd",
      //   {
      // 	  {"Requantize"},
      // 	  {"*"},
      // 	  {"*"},
      // 	  {"*"},
      // 	  {"Const"},
      // 	  {"Const"},
      //   }
      // },  // clang-format on

  /************************/  
Status BiasAdd2Add(const GraphDef& input_graph_def,
                                    const TransformFuncContext& context,
                                    GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"QuantizedAdd",
        {
      	  {"Requantize"},
      	  {"*"},
      	  {"*"},
      	  {"*"},
      	  {"Const"},
      	  {"Const"},
        }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {

        const NodeDef& old_bias_add_node = match.node;
	
	std::cout << "=============================================" << std::endl;
	std::cout << "old_bias_add_node = " << old_bias_add_node.name() << std::endl;
	std::cout << "match.inputs[0].node.name() = " << match.inputs[0].node.name() << std::endl;
	std::cout << "match.inputs[1].node.name() = " << match.inputs[1].node.name() << std::endl;
	std::cout << "match.inputs[2].node.name() = " << match.inputs[2].node.name() << std::endl;
	std::cout << "match.inputs[3].node.name() = " << match.inputs[3].node.name() << std::endl;
	std::cout << "match.inputs[4].node.name() = " << match.inputs[4].node.name() << std::endl;
	std::cout << "match.inputs[5].node.name() = " << match.inputs[5].node.name() << std::endl;

        new_nodes->push_back(match.inputs[0].node);
        new_nodes->push_back(match.inputs[1].node);
        // new_nodes->push_back(match.inputs[2].node:1);
        // new_nodes->push_back(match.inputs[3].node:2);
        new_nodes->push_back(match.inputs[4].node);
        new_nodes->push_back(match.inputs[5].node);

	
        NodeDef new_add_node;
        new_add_node.set_op("QuantizedBiasAdd");
        new_add_node.set_name(old_bias_add_node.name());
        SetNodeAttr("T1", DT_QUINT8, &new_add_node);
	SetNodeAttr("T2", DT_QUINT8, &new_add_node);
        SetNodeAttr("Toutput", DT_QINT32, &new_add_node);
        AddNodeInput(match.inputs[0].node.name() + ":0", &new_add_node);
	AddNodeInput(match.inputs[1].node.name(), &new_add_node);
	AddNodeInput(match.inputs[2].node.name() + ":1", &new_add_node);
	AddNodeInput(match.inputs[3].node.name() + ":2", &new_add_node);
        AddNodeInput(match.inputs[4].node.name(), &new_add_node);
	AddNodeInput(match.inputs[5].node.name(), &new_add_node);
        new_nodes->push_back(new_add_node);


	std::cout << "=============================================" << std::endl;	
        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

  /************************/

REGISTER_GRAPH_TRANSFORM("biasadd_to_add", BiasAdd2Add);
  
}  // namespace graph_transforms
}  // namespace tensorflow
