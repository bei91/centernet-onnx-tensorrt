## yolov5-onnx-tensorrt 
This Repos contains how to run CenterNet model using TensorRT.  
The Pytorch implementation is [xingyizhou/CenterNet ](https://github.com/xingyizhou/CenterNet).  
Convert pytorch to onnx and tensorrt model to run on a Jetson AGX Xavier.  
Support to infer an image.  
Support to infer multi images simultaneously.

## Requirements 
Please use torch>=1.6.0 + onnx==1.8.0 + TensorRT >=6.0 to run the code

## Code structure 
`networks` code is network  
`demo` code runs tensorrt implementation on Jetson AGX Xavier
```
├── networks
├── utils
├── models
├── demo
│   ├── centernet_inference.py
|   ├── centernet_inference_batch.py
|   ├── trt_inference.py
|   ├── trt_function.py
│   └── torch_onnx_tensorrt.py
```

- [x] convert CenterNet pytorch model to onnx
- [x] convert CenterNet onnx model to tensorrt
- [x] pre-process image 
- [x] run inference against input using tensorrt engine
- [x] post process output (forward pass)
- [x] apply nms thresholding on candidate boxes
- [ ] visualize results
- [ ] demo.py

___
## Compile pytorch model to onnx and onnx to tensorrt
```
python torch_onnx_tensorrt.py

```


## Run demo to infer 
```
python demo.py

```
___

## Thanks
* [https://github.com/xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)