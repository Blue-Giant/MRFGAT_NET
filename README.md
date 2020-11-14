# Multi-scale Receptive Fields Graph Attention Network for Point Cloud Classification
created by Xi-An Li, Lei Zhang, Li-Yan Wang, Jian Lu

[[Paper]](https://arxiv.org/abs/2009.13289)

# Overview
Understanding the implication of point cloud is still challenging to achieve the goal of classification or segmentation due to the irregular and sparse structure of point cloud. As we have known, PointNet architecture as a ground-breaking work for point cloud which can learn efficiently shape features directly on unordered 3D point cloud and have achieved favorable performance. However, this model fail to consider the fine-grained semantic information of local structure for point cloud. Afterwards, many valuable works are proposed to enhance the performance of PointNet by means of semantic features of local patch for point cloud. In this paper, a multi-scale receptive fields graph attention network (named after MRFGAT) for point cloud classification is proposed. By focusing on the local fine features of point cloud and applying multi attention modules based on channel affinity, the learned feature map for our network can well capture the abundant features information of point cloud. The proposed MRFGAT architecture is tested on ModelNet10 and ModelNet40 datasets, and results show it achieves state-of-the-art performance in shape classification tasks.

# Requirement
* [TensorFlow](https://www.tensorflow.org/)

# Point Cloud Classification
* Run the training script:
``` bash
python train.py
```
* Run the evaluation script after training finished:
``` bash
python evaluate.py --model=network --model_path=log/epoch_185_model.ckpt
```

# Point Cloud Part Segmentation
* Run the training script:
``` bash
python train_multi_gpu.py
```
* Run the evaluation script after training finished:
``` bash
python test.py --model_path train_results/trained_models/epoch_130.ckpt
```

# Citation
Please cite this paper if you want to use it in your work.

``` bash
@article{chen2019gapnet,
  title={GAPNet: Graph Attention based Point Neural Network for Exploiting Local Feature of Point Cloud},
  author={Chen, Can and Fragonara, Luca Zanotti and Tsourdos, Antonios},
  journal={arXiv preprint arXiv:1905.08705},
  year={2019}
}
```

# License
MIT License

