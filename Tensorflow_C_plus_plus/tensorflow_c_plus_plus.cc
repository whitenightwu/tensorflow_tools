/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : tensorflow_c_plus_plus.cc
 * Authors    : whitewu@whitewu-ubuntu
 * Create Time: 2019-07-22:10:22:11
 * Description:
 * 
 */

#include <iostream>
#include "tensorflow_c_plus_plus.h"

using namespace std;


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"

int main(int argc, char* argv[]) {


    ////////////////////////// c++
    // loading graph proto

  ////////////////////////// c++
  // first
    tensorflow::GraphDef graph_def;
    std::ifstream f_in(argv[1]);
    bool ctrl = graph_def.ParseFromIstream(&f_in);
    f_in.close();
    if (ctrl) std::cout << "deserialization successful\n\n";

    std::cout << "Graph contents:\n";
    std::cout << tensorflow::SummarizeGraphDef(graph_def) << "\n";

    ////////////////////////// c++
    // second
    tensorflow::GraphDef graph_def;
    std::ifstream graph_path(argv[1]);
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    
    ////////////////////////// c++
    // third
    tensorflow::GraphDef graph_def;
    std::string facenet_string;
    facenet_string = facenet_pb;
    facenet_pb = mtcnn_pb;
    // ParseFromString(const string& data):            解析给定的string
    bool ctrl = graph_def.ParseFromString(facenet_string);

    

    ////////////////////////// c
    // loading graph_def as TF_Graph
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_Status* status = TF_NewStatus();
    TF_Graph* tf_graph = TF_NewGraph();
    std::string gd_as_string = std::string();
    graph_def.SerializeToString(&gd_as_string);
    TF_Buffer* gd_as_buffer = TF_NewBufferFromString(gd_as_string.c_str(), gd_as_string.length());
    TF_GraphImportGraphDef(tf_graph, gd_as_buffer, opts, status);

    if (TF_GetCode(status) == 0)
      std::cout << "graph_def succesfully loaded into tf_graph\n";
    else
      std::cout << TF_Message(status) << '\n';

    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteStatus(status);
    TF_DeleteGraph(tf_graph);

    return 0;
}
