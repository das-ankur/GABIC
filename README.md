# GABIC: GRAPH-BASED ATTENTION BLOCK FOR IMAGE COMPRESSION

This repository contains the official implementation of **GABIC: GRAPH-BASED ATTENTION BLOCK FOR IMAGE COMPRESSION**, published at ICIP 2024.

[Paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10647413)

## To Train a Model

```
cd src

python train.py \
--save-dir /path/to/save \
--model wgrcnn_cw \
--lambda 0.0035 \
--epochs 200 \
--num-workers 4 \
--cuda \
--project-name name \
--dataset /path/to/openimages/ \
--test-pt /path/to/kodak \
--save \
--knn 9 \
--graph-conv transf_custom \
--local-graph-heads 8 \
--use-edge-attr \
--seed 42
```

## To Train a Model
- Download our pretrained model in the GABIC directory from [here](TODO).
- Extract model_results.zip
- Run:
```
cd src

python -m evaluate.eval --dataset /path/to/kodak
```