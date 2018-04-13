#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : quant_weight.py
## Authors    : slwang@taurus
## Create Time: 2018-02-05:17:10:29
## Description:
## 
##
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#caffe_root = "/home/lzlu/work/git/caffe-jacinto"
caffe_root = "/home/lzlu/work/git/tmp/caffe_ingenic"
##sys.path.insert(0, caffe_root + "python")
print sys.path
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
caffe.set_mode_cpu()

QUANT_BIT = 8

def parse_args():
    '''Parse input parameter'''
    parser = argparse.ArgumentParser(description = 'quantize weight')
    parser.add_argument('--weights_src', dest = "weights_src", help = "input model path", default = " ", type = str)
    parser.add_argument("--model_src", dest = "model_src", help = "input net protxt", default = " ", type = str)
    parser.add_argument('--weights_dst', dest = "weights_dst", help = "output model path", default = " ", type = str)
    parser.add_argument("--model_dst", dest = "model_dst", help = "output net protxt", default = " ", type = str)
    parser.add_argument("--op", dest = "operation", help = "operation:[mergeB,mergeM,quantW]", default = " ", type = str)
    args = parser.parse_args()
    return args


def merge_BatchNormalAndScale(weights_src, model_src, weights_dst, model_dst):
    net_src = caffe.Net(model_src, weights_src, caffe.TEST)
    net_dst = caffe.Net(model_dst, weights_src, caffe.TEST)
    params_src = net_src.params.keys()

    #for layer_name, blob in net_src.blobs.iteritems():
    #    print layer_name + '\t' + str(blob.data.shape)
    
    netParam = caffe_pb2.NetParameter()
    f = open(weights_src, 'rb')
    netParam.ParseFromString(f.read())
    f.close()
    layers = netParam.layer
    layer_type={}
    batchNormal_eps={}
    for layer in layers:
        layer_type[layer.name] = layer.type
        if layer.type == "BatchNorm":
            batchNormal_eps[layer.name] = layer.batch_norm_param.eps if layer.batch_norm_param.HasField('eps') else 0.00001
        
    weights_MaxMin=[{},{}]
    last_convfc_name=""
    for param_name in params_src:
        if (layer_type[param_name] == "Convolution" or layer_type[param_name] == "InnerProduct"):
            last_convfc_name = param_name

        num_blobs=len(net_src.params[param_name])
        ##print("num_blobs:",num_blobs)
        if layer_type[param_name] == "BatchNorm" :
            num = net_src.params[last_convfc_name][0].data.shape[0]
            ##print("Output_Channel:",num)
            blobs_num = len(net_src.params[param_name])
            for n in range(num):
                net_dst.params[last_convfc_name][0].data[n, :, : ,:] = net_dst.params[last_convfc_name][0].data[n, :, : ,:] \
                                                                               / np.sqrt(net_src.params[param_name][1].data[n] + batchNormal_eps[param_name])## default 0.00001
                net_dst.params[last_convfc_name][1].data[n] = (net_dst.params[last_convfc_name][1].data[n] - net_src.params[param_name][0].data[n]) \
                                                                               / np.sqrt(net_src.params[param_name][1].data[n] + batchNormal_eps[param_name])## default 0.00001
                if blobs_num == 5:
                    net_dst.params[last_convfc_name][0].data[n, :, : ,:] = net_dst.params[last_convfc_name][0].data[n, :, : ,:] * net_src.params[param_name][3].data[n]
                    net_dst.params[last_convfc_name][1].data[n] = net_dst.params[last_convfc_name][1].data[n] * net_src.params[param_name][3].data[n] \
                                                                  + net_src.params[param_name][4].data[n]
        elif layer_type[param_name] == "Scale" :
            num = net_src.params[last_convfc_name][0].data.shape[0]
            ##print("Output_Channel:",num)
            for n in range(num):
                net_dst.params[last_convfc_name][0].data[n, :, : ,:] = net_dst.params[last_convfc_name][0].data[n, :, : ,:] * net_src.params[param_name][0].data[n]
                net_dst.params[last_convfc_name][1].data[n] = net_dst.params[last_convfc_name][1].data[n] * net_src.params[param_name][0].data[n] \
                                                                      + net_src.params[param_name][1].data[n]
        else:
            for i in range(0,num_blobs,1):
                net_dst.params[param_name][i].data[...] = net_src.params[param_name][i].data[...]

    ### save weight max and min after mergeBN&scale
    for param_name in params_src:
        if (layer_type[param_name] == "Convolution" or layer_type[param_name] == "InnerProduct"):
            weight_min = net_dst.params[param_name][0].data.min()
            weight_max = net_dst.params[param_name][0].data.max()
            weights_MaxMin[0][param_name] = weight_max
            weights_MaxMin[1][param_name] = weight_min
    ###
        
    net_dst.save(weights_dst)
    copy_MaxAndMin(weights_src, weights_dst)
    save_ScaleParamMN(weights_MaxMin,weights_dst)
    ##For test
    netParam_dst = caffe_pb2.NetParameter()
    f1 = open(weights_dst, 'rb')
    netParam_dst.ParseFromString(f1.read())
    f1.close()
    layers_dst = netParam_dst.layer
    for layer_dst in layers_dst:
        print("this is second time:")
        print(layer_dst.quantization_param)
    
def merge_MeanValue(weights_src, model_src, weights_dst, model_dst):
        
    net_src = caffe.Net(model_src, weights_src, caffe.TEST)
    net_dst = caffe.Net(model_dst, weights_src, caffe.TEST)
    params_src = net_src.params.keys()
    #blobs_src  = net_src.blobs.keys()
    print("params_src:",params_src)
    
    netParam = caffe_pb2.NetParameter()
    f = open(weights_src, 'rb')
    netParam.ParseFromString(f.read())
    f.close()
    layers = netParam.layer

    #mean_value = [103.94, 116.78, 123.68]
    #scale = 0.017
    mean_value = [128, 128, 128]
    first_conv_name=""
    for layer in layers:
        if layer.type == "Data":
            #_mean_value[layer.name] = layer.transform_param.mean_value if layer.transform_param.HasField('mean_value') else 0
            print("dir(layer.transform_param)",dir(layer.transform_param))
            if len(layer.transform_param.mean_value):
                index = 0
                print "len(layer.transform_param.mean_value):",len(layer.transform_param.mean_value)
                for mean in layer.transform_param.mean_value:
                    mean_value[index] = mean
                    index += 1
            print("layer.transform_param.HasField('scale'):",layer.transform_param.HasField('scale'))
            scale = layer.transform_param.scale if layer.transform_param.HasField('scale') else 1
        if layer.type == "Convolution" :
            first_conv_name = layer.name
            break

    first_conv = net_src.params[first_conv_name]
    out_channel,in_channel,kernel_h,kernel_w = first_conv[0].data.shape
    print("%s:out_channel:%s,in_channel:%s,kernel_h:%s,kernel_w:%s"%(first_conv_name,out_channel,in_channel,kernel_h,kernel_w))

    print("mean_valueM:",mean_value)
    print("scaleM:",scale)
    if len(first_conv) >= 2 :
        print("bias_shape: ", first_conv[1].data.shape[0])
        print("output_num {}, input_channel {}".format(out_channel, in_channel))
        for n in range(out_channel):
            summ = 0.0
            for c in range(in_channel):
                for h in range(kernel_h):
                    for w in range(kernel_w):
                        first_conv[0].data[n, c, h, w] = first_conv[0].data[n, c, h, w] * scale
                        summ += first_conv[0].data[n, c, h, w] * mean_value[c]
                        
            first_conv[1].data[n]= first_conv[1].data[n] - summ
    else:
        print("Error: Bias is not Exist!")
        print("len(first_conv):",len(first_conv))

    net_dst.save(weights_dst)
    copy_MaxAndMin(weights_src, weights_dst)
        
def _get_scale(Nbit, maxVal, minVal, isRELU=False):
    half_quan_num = (1 << (Nbit-1))
    if isRELU:
        return ((1 << Nbit) - 1)/(abs(maxVal)+0.00001)
    else:
        scale_max = (half_quan_num - 0.5)/(abs(maxVal)+0.00001)
        scale_min = (half_quan_num + 0.5)/(abs(minVal)+0.00001)
        return min(scale_max,scale_min)
    
    
def quant_Weight(weights_src, model_src, weights_dst, model_dst):
    net_src = caffe.Net(model_src, weights_src, caffe.TEST)
    net_dst = caffe.Net(model_dst, weights_src, caffe.TEST)
    params_src = net_src.params.keys()
    print("params_src:",params_src)
    
    netParam = caffe_pb2.NetParameter()
    f = open(weights_src, 'rb')
    netParam.ParseFromString(f.read())
    f.close()
    layers = netParam.layer

    quan_bit = 8
    quan_num = ((1 << 8 )-1)
    isFirstConv = True
    for layer in layers:
        print ' layer_name: "%s", layer_type: "%s"'%(layer.name, layer.type)
        if (layer.type == "Convolution" or layer.type == "InnerProduct"):
            ##weight
            weight_min = net_src.params[layer.name][0].data.min()
            weight_max = net_src.params[layer.name][0].data.max()
            ###max_val = max(np.abs(weight_max), np.abs(weight_min))
            ####print "weight_max: %f, weight_min: %f, max_val: %f"%(weight_max, weight_min, max_val)
            ###weight_scale = quan_num  / (2 * max_val)
            print("WHY:")
            weight_scale = _get_scale(QUANT_BIT, weight_max, weight_min)
            net_dst.params[layer.name][0].data[...] = np.round(weight_scale * net_src.params[layer.name][0].data[...])
            ###net_dst.params[layer.name][0].data[...] =  net_dst.params[layer.name][0].data[...] / weight_scale ## for reverse quant

            ##bias
            first_qparam_in=''
            for qparam_in in layer.quantization_param.qparam_in:
                first_qparam_in = qparam_in
                break
            input_min = first_qparam_in.min
            input_max = first_qparam_in.max
            ###input_max_val = max(np.abs(input_max), np.abs(input_min))
            ###input_scale  = quan_num  / (2 * input_max_val)
            input_scale = _get_scale(QUANT_BIT, input_max, input_min,True)
            if layer.type == "Convolution" and isFirstConv :
                isFirstConv = False
                input_scale = 1.0

            print "weight_max, weight_min:",weight_max, weight_min
            print "weight_scale:",weight_scale ,"input_scale:",input_scale
            #net_dst.params[layer.name][1].data[...] = net_src.params[layer.name][1].data[...]
            net_dst.params[layer.name][1].data[...] = np.round(input_scale * weight_scale * net_src.params[layer.name][1].data[...])
            ###net_dst.params[layer.name][1].data[...] =  net_dst.params[layer.name][1].data[...] / input_scale / weight_scale ## for reverse quant
            # #if layer.type == "Convolution":
            # #    print "one step"
            # #    break
        # #print "two step"    
    net_dst.save(weights_dst)
    copy_MaxAndMin(weights_src, weights_dst)
    
    netParam_dst = caffe_pb2.NetParameter()
    f1 = open(weights_dst, 'rb')
    netParam_dst.ParseFromString(f1.read())
    f1.close()
    layers_dst = netParam_dst.layer
    for layer_dst in layers_dst:
        print("this is second timeXXXX:")
        print(layer_dst.quantization_param)

def compute_ScaleParamMN(input_scale, output_scale, weight_scale):
    ###max_input_real = max(abs(input_min), abs(input_max))
    ###max_output_real = max(abs(output_min), abs(output_max))
    ###max_weights_real = max(abs(weights_min), abs(max_weights_max))
    ###scale = max_weights_real * max_input_real / max_output_real / 127.0
    scale = output_scale / input_scale / weight_scale
    
    min_loss = 100.0;
    best_n = -1;
    for n in range(-32,32,1):
        m = int(scale * pow(2,n)+0.5);    
        if(m>0 and m < 256):
            loss = abs(scale - m/(pow(2,n)+0.0));
            if(min_loss>loss):
	        min_loss = loss;
	        best_n = n;
                                
    best_m = int(scale * pow(2,best_n)+0.5);
    return best_m, best_n

def save_ScaleParamMN(weights_MaxMin,weights_dst):
    dirname = os.path.dirname(weights_dst)
    tmpFile = "tmp.xxxx.caffemodel"
    weights_tmp = dirname+"/"+tmpFile
    if os.path.isfile(weights_tmp):
        os.remove(weights_tmp)
        print "remove tmp file:",weights_tmp
    else:
        print "there is no tmp file:",weights_tmp
    os.rename(weights_dst,weights_tmp)
    
    netParam_dst = caffe_pb2.NetParameter()
    with open(weights_tmp,'rb') as fr:
        netParam_dst.ParseFromString(fr.read())
    layers_dst = netParam_dst.layer

    isFirstConv = True
    for layer_dst in layers_dst:
        if (layer_dst.type == "Convolution" or layer_dst.type == "InnerProduct"):
            qparam_out = layer_dst.quantization_param.qparam_out
            qparam_w_max = weights_MaxMin[0][layer_dst.name]
            qparam_w_min = weights_MaxMin[1][layer_dst.name]
            
            count = 0
            for qparam_in in layer_dst.quantization_param.qparam_in:
                output_scale = _get_scale(QUANT_BIT, qparam_out.max, qparam_out.min, True)
                weight_scale = _get_scale(QUANT_BIT, qparam_w_max, qparam_w_min)
                input_scale  = _get_scale(QUANT_BIT, qparam_in.max, qparam_in.min, True)
                if layer_dst.type == "Convolution" and isFirstConv :
                    isFirstConv = False
                    input_scale = 1.0

                    # #output_scale = 1.0
                    
                if layer_dst.type == "InnerProduct" :
                    output_scale = 1.0

                print "qparam_w_max:",qparam_w_max,"qparam_w_min:",qparam_w_min
                print("input_scale, output_scale, weight_scale:",input_scale, output_scale, weight_scale)
                best_m, best_n = compute_ScaleParamMN(input_scale, output_scale, weight_scale)
                layer_dst.quantization_param.scaleparam_m = best_m
                layer_dst.quantization_param.scaleparam_n = best_n
                count += 1
                
                # #if layer_dst.type == "Convolution":
                # #    break
                    
    with open(weights_dst,'wb') as fr:
        fr.write(netParam_dst.SerializeToString())
    os.remove(weights_tmp)

def copy_MaxAndMin(weights_src, weights_dst):

    dirname = os.path.dirname(weights_dst)
    tmpFile = "tmp.xxxx.caffemodel"
    weights_tmp = dirname+"/"+tmpFile
    if os.path.isfile(weights_tmp):
        os.remove(weights_tmp)
        print "remove tmp file:",weights_tmp
    else:
        print "there is no tmp file:",weights_tmp
    os.rename(weights_dst,weights_tmp)
        
    netParam_src = caffe_pb2.NetParameter()
    with open(weights_src,'rb') as fr:
        netParam_src.ParseFromString(fr.read())
    layers_src = netParam_src.layer

    netParam_dst = caffe_pb2.NetParameter()
    with open(weights_tmp,'rb') as fr:
        netParam_dst.ParseFromString(fr.read())
    layers_dst = netParam_dst.layer
    
    mean_value=[0,0,0]
    for layer_dst in layers_dst:
        for layer_src in layers_src:
            if layer_dst.name  == layer_src.name:
                #mean_value&scale
                if layer_src.type == "Data":
                    for index in range(len(layer_src.transform_param.mean_value)):
                        if len(layer_src.transform_param.mean_value) < index+1:
                            layer_dst.transform_param.mean_value.append(layer_src.transform_param.mean_value[index])
                        else:
                            layer_dst.transform_param.mean_value[index] = layer_src.transform_param.mean_value[index]

                    scale = layer_src.transform_param.scale if layer_src.transform_param.HasField('scale') else 1
                    if layer_dst.transform_param.HasField('scale'):
                        layer_dst.transform_param.scale = scale
                    else:
                        layer_dst.transform_param.append(scale)
                    #print("layer_dst.transform_param.mean_value:",layer_dst.transform_param.mean_value)
                    #print("layer_dst.transform_param.scale:",layer_dst.transform_param.scale)
                #out
                layer_dst.quantization_param.qparam_out.max = layer_src.quantization_param.qparam_out.max
                layer_dst.quantization_param.qparam_out.min = layer_src.quantization_param.qparam_out.min
                
                #w
                layer_dst.quantization_param.qparam_w.max = layer_src.quantization_param.qparam_w.max
                layer_dst.quantization_param.qparam_w.min = layer_src.quantization_param.qparam_w.min
                
                #in 
                count = 0
                for qparam_in_src in layer_src.quantization_param.qparam_in:
                    if count >= len(layer_dst.quantization_param.qparam_in):
                        qparam_in_dst = layer_dst.quantization_param.qparam_in.add()
                        qparam_in_dst.max = qparam_in_src.max
                        qparam_in_dst.min = qparam_in_src.min
                    count += 1

                # m n  
                layer_dst.quantization_param.scaleparam_m = layer_src.quantization_param.scaleparam_m
                layer_dst.quantization_param.scaleparam_n = layer_src.quantization_param.scaleparam_n
                    
        print("this is first time:","layer_dst.name:",layer_dst.name)
        print(layer_dst.quantization_param)
        if (layer_dst.type == "Convolution" or layer_dst.type == "InnerProduct"):
            print("layer_dst.quantization_param.scaleparam_m:",layer_dst.quantization_param.scaleparam_m)
        
        
    with open(weights_dst,'wb') as fr:
        fr.write(netParam_dst.SerializeToString())

    '''
    ##For test
    netParam_dst = caffe_pb2.NetParameter()
    f1 = open(weights_dst, 'rb')
    netParam_dst.ParseFromString(f1.read())
    f1.close()
    layers_dst = netParam_dst.layer
    for layer_dst in layers_dst:
        print("this is second time:")
        print(layer_dst.quantization_param)
    '''
    os.remove(weights_tmp)
        
if __name__ == "__main__":
    """Parse argument"""
    args = parse_args()
    weights_src = args.weights_src
    model_src   = args.model_src
    weights_dst = args.weights_dst
    model_dst   = args.model_dst
    operation   = args.operation
    
    #weights_src = "/home/lzlu/work/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2017-12-25_11-38-03/initial/imagenet_mobilenet-1.0_iter_80000.caffemodel"
    #model_src   = "/home/lzlu/work/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2017-12-25_11-38-03/initial/deploy.prototxt"
    #weights_dst = "./new_mobilenet_iter_1.caffemodel"
    #model_dst   = "/home/lzlu/work/git/caffe-jacinto/quan_tools/new_deploy.prototxt"
    if operation == "mergeB":
        merge_BatchNormalAndScale(weights_src, model_src, weights_dst, model_dst)
    elif operation == "mergeM":
        merge_MeanValue(weights_src, model_src, weights_dst, model_dst)
    elif operation == "quantW":
        quant_Weight(weights_src, model_src, weights_dst, model_dst)
    else:
        print "Use Right Op!"
        
    print("Done!")
