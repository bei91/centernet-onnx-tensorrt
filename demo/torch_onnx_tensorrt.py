#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    : torch_onnx_tensorrt.py
@Autor   : jiabb
@Time    : 2019/9/2
"""
import torch
import tensorrt as trt
import pycuda.driver as cuda
from network.dlav0 import DLASeg
# import calibrator

def saveONNX(model, filepath):
    '''
    save onnx
    param model: pytorch model
    param filepath: onnx save path
    '''
    model1 = DLASeg('dla{}'.format(34), {'hm': 80, 'wh': 2, 'reg': 2},
                 pretrained=False,
                 down_ratio=4,
                 head_conv=256)
    checkpoint = torch.load(model)
    model1.load_state_dict(checkpoint['state_dict'])
    model1.cuda()

    #input
    dummy_input = torch.randn(1, 3, 512, 512, device='cuda')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model1, (dummy_input), filepath, verbose=True, input_names=input_names, output_names=output_names, export_params=True)

#tensorrt 7
def get_engine_trt7(onnx_model_name, trt_model_name):
  batch_size = 1  #4 or other
  G_LOGGER = trt.Logger(trt.Logger.WARNING)
  explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)#trt7
  with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
    builder.max_batch_size = batch_size
    builder.max_workspace_size = 1 << 20
    print('Loading ONNX file from path {}...'.format(onnx_model_name))
    with open(onnx_model_name, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    print('Completed parsing of ONNX file')
    print('Building an engine from file {}; this may take a while...'.format(onnx_model_name))
    #builder.int8_mode = True
    #builder.int8_calibrator = calib
    builder.fp16_mode = True
    print("num layers:", network.num_layers)
    #last_layer = network.get_layer(network.num_layers - 1)
    #if not last_layer.get_output(0):
    network.get_input(0).shape = [batch_size, 3, 512, 512]
    engine = builder.build_cuda_engine(network)
    print("engine:", engine)
    with open(trt_model_name, "wb") as f:
        f.write(engine.serialize())
    return engine
    print("Completed creating Engine")
#tensorrt 6
def get_engine_trt6(onnx_file_path, engine_file_path):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    '''
    :param onnx_file_path: onnx path
    :return: engine
    '''
    # 打印日志
    with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                               G_LOGGER) as parser:
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 20  # 1024

        builder.fp16_mode = True  # convert to fp16 tensorrt

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        print(network.num_layers)
        for i in range(network.num_layers):
            print(network.get_layer(i).name)
        print("input shape:", network.get_layer(2).get_input(0).shape,
              "input name:", network.get_layer(2).get_input(0).name)
        print("output shape:", network.get_layer(network.num_layers - 1).get_output(0).shape,
              "output name:", network.get_layer(network.num_layers - 1).get_output(0).name)

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        #save
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

def get_engine_int8(onnx_file_path, calib, engine_int8_file_path):
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = 1  # calib.get_batch_size()
        builder.max_workspace_size = 1 << 10
        builder.int8_mode = True
        builder.int8_calibrator = calib
        with open(onnx_file_path, 'rb') as model:
           parser.parse(model.read())   # , dtype=trt.float32
        #return builder.build_cuda_engine(network)
        engine = builder.build_cuda_engine(network)
        #print(engine)
        print("Completed creating Engine")
    #        # 保存计划文件
        with open(engine_int8_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__ == '__main__':
    # import glob
    # from random import shuffle
    #
    # CALIBRATION_DATASET_LOC = '../cal/*'
    #
    # def create_calibration_dataset():
    #     # Create list of calibration images (filename)
    #     # This sample code picks 1000 images at random from training set
    #     calibration_files = glob.glob(CALIBRATION_DATASET_LOC)  # 全部图片
    #     shuffle(calibration_files)
    #     print(calibration_files)
    #     return calibration_files[:1170]
    #     #return calibration_files[:10]
    #
    #
    # def sub_mean_chw(data):
    #     data = data.transpose((1, 2, 0))  # CHW -> HWC
    #     # data -= np.array(MEAN)
    #     # data = data - np.array(MEAN)  # Broadcast subtract
    #     # print(type(data),type(np.array(MEAN)))
    #     #data = ((data/255. - np.array([0.408, 0.447, 0.470]) / np.array([0.289, 0.274, 0.278])) * 255)
    #     #data = data - np.array([104, 114, 120])
    #
    #     data = data.transpose((2, 0, 1))  # HWC -> CHW
    #     return data

    # calibration_files = create_calibration_dataset()  # 1000
    # #Process 5 images at a time for calibration
    # #This batch size can be different from MaxBatchSize (1 in this example)
    # batchstream = calibrator.ImageBatchStream(5, calibration_files, sub_mean_chw)
    # calib = calibrator.PythonEntropyCalibrator(["data"], batchstream)
    # #onnx model convert int8 engine
    # save_int8_engine_path = 'model_best_all_dete_int8.trt'
    # get_engine_int8('model_best_all_dete.onnx', calib, save_int8_engine_path)
    saveONNX('coco_80.pth', 'coco_80.onnx')
    get_engine_trt7('coco_80.onnx', 'coco_80.trt')

