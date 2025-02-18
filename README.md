 
  
# BEVFormer ONNX conversion and inference

This repository is for converting BEVFormer to onnx and performing inference based on the onnx graph. It is not intended for training BEVFormer.

## Dataset

Download and prepare the nuscenes dataset as normal for [BEVFormer preparation](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md).

## Building and Running in Docker

Build the dockerfile
Clone the repo and use the bevonnx branch
```
git clone -b bevonnx git@github.com:rod409/BEVFormer.git
cd ./BEVFormer
```

Build the docker container
```
docker build -t bevonnx .
```

Run the container and mount the directory containing the data along with the pth and onnx checkpoints
```
docker run -v <path to NuScenes dataset>:/BEVFormer/data/nuscenes --ipc=host -it bevonnx
```

## Export to ONNX and ONNX inference
Export the model to ONNX. This will generate a bevformer.onnx file. We will need to disable any CUDA usage for the export.
```
export CUDA_VISIBLE_DEVICES=""
python tools/export_onnx.py projects/configs/bevformer/bevformer_tiny.py ./data/nuscenes/bevformer_tiny_epoch_24.pth --eval bbox
```
Perform inference with the onnx graph.
```
python tools/test_onnx.py projects/configs/bevformer/bevformer_tiny.py ./bevformer.onnx --eval bbox
```

# Acknowledgements

Based on the source code of BEVFormer and BEVFormer_tensorrt
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) 
- [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) 



