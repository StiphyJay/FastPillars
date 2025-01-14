# FastPillars

[**FastPillars:
A Deployment-friendly Pillar-based 3D Detector**](https://arxiv.org/abs/2302.02367)\
*Sifan Zhou, Zhi Tian, Xiangxiang Chu, Xinyu Zhang, Bo Zhang, Xiaobo Lu, Chengjian Feng, Zequn Jie, Patrick Yin Chiang and Lin Ma*\
Southeast University, Meituan, Fudan University\
*Paper ([arXiv 2302.02367](https://arxiv.org/abs/2302.02367))*



## ToDo List
- [ ] release the implementation code for KITTI dataset
- [ ] release the TensorRT implementation code
- [ ] release the training code and details
- [x] release the model weights on nuScenes val set
- [x] release the inference code on nuScenes dataset

## Contact
Any questions or suggestions are welcome! Sifan Zhou [sifanjay@gmail.com](mailto:sifanjay@gmail.com)

## Abstract
The deployment of 3D detectors strikes one of the major challenges in real-world self-driving scenarios. Existing BEV-based (i.e., Bird Eye View) detectors favor sparse convolutions (known as SPConv) to speed up training and inference, which puts a hard barrier for deployment, especially for on-device applications. In this paper, to tackle the challenge of efficient 3D object detection from an industry perspective, we devise a deployment-friendly pillar-based 3D detector, termed FastPillars. First, we introduce a novel lightweight Max-and-Attention Pillar Encoding (MAPE) module specially for enhancing small 3D objects. Second, we propose a simple yet effective principle for designing a backbone in pillar-based 3D detection. We construct FastPillars based on these designs, achieving high performance and low latency without SPConv. Extensive experiments on two large-scale datasets demonstrate the effectiveness and efficiency of FastPillars for on-device 3D detection regarding both performance and speed. Specifically, FastPillars delivers state-of-the-art accuracy on Waymo Open Dataset with 1.8X speed up and 3.8 mAPH/L2 improvement over CenterPoint (SPConv-based). 

**FastPillars Framework**
<div>
  <img width="100%" alt="FastPillars architecture" src="figs/pipeline.png">
</div>
<div>
  <img width="100%" alt="FastPillars architecture" src="figs/pillar-encoding.png">
</div>
<div>
<img width="47%" src="figs/blocks_number.png"> <img width="40%" src="figs/Backbone_pipeline.png"> 
</div>
<div>

# Highlights

- **Simple and SPConv-free:** Two sentences method summary: We employ a  Max-and-Attention Pillar Encoding (MAPE) module to enhance pillar feature extraction and a computation reallocation principle to improve the bev feature representation. The SPConv-free design enables our method to be seamlessly accelerated using TensorRT.

- **Fast and Accurate**: Our best single model achieves *73.3* mAPH on Waymo val set and *71.8* NDS on nuScenes test set while running at 20FPS+. 

## Main results

#### 3D detection on Waymo val set

|         |  #Frame | Veh_L2 (mAP/mAPH) | Ped_L2 (mAP/mAPH) | Cyc_L2 (mAP/mAPH)  | Mean_L2 mAPH   |  FPS  |
|---------|---------|--------|--------|---------|--------|-------|
|FastPillars | 1       |  71.5 / 71.1     |  73.2 / 67.2      |  70.5 / 69.5       |  69.3     |   27.4    | 
|FastPillars | 2       |  72.5 / 72.0 | 75.5 / 72.4 |73.9 / 73.0       |   72.5     |  24.3     |
|FastPillars | 3       |  73.2 / 72.8 | 76.3 / 73.2 | 74.6 / 73.8       |   73.3      |  21.7     |

#### 3D detection on Waymo test set (mAPH)

|         |  #Frame | Veh_L2 (mAP/mAPH) | Ped_L2 (mAP/mAPH) | Cyc_L2 (mAP/mAPH)  |
|---------|---------|--------|--------|---------|
|FastPillars | 1       |  75.4 / 75.0     |  75.0 / 69.2      |  70.3 / 69.2       |
|FastPillars | 2       |  76.5 / 76.1 | 77.2 / 73.9 |74.1 / 73.1       |
|FastPillars | 3       |  77.1 / 76.7 | 77.8 / 74.6 | 74.2 / 73.2      |


#### 3D detection on nuScenes val set  

|         | Model Weights|  mAP | NDS  |car  | truck |bus  | trailer | CV  | Ped |Motor | Bic |TC  | barrier |
|---------|---------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|--------|--------|
|FastPillars |[Model Weights](https://drive.google.com/file/d/1OmCO_E-YFr9AXjqGwj6_5Dp4JirBV7Ml/view?usp=sharing)| 61.3   | 68.2 |   87.3   | 58.7 |71.9   | 39.8 |20.2   | 86.6 |63.3   | 45.8 |72.8   | 66.7 | 


#### 3D detection on nuScenes test set 

|         |  mAP | NDS  |car  | truck |bus  | trailer | CV  | Ped |Motor | Bic |TC  | barrier |
|---------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|--------|--------|
|FastPillars | 66.8   | 71.8 |   87.3 | 58.0 | 66.0 | 62.3 | 34.5 | 87.4 |70.3| 47.9 | 81.5 | 72.8| 

All results are tested on a Titan RTX GPU with batch size 1.

## Use FastPillars

### Installation

Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

### Benchmark Evaluation and Training 

Please refer to [GETTING_START](docs/GETTING_START.md) to prepare the data. Then follow the instruction there to reproduce our detection and tracking results. All detection configurations are included in [configs](configs).

### Develop

If you are interested in training CenterPoint on a new dataset, use CenterPoint in a new task, or use a new network architecture for CenterPoint, please refer to [DEVELOP](docs/DEVELOP.md). Feel free to send us an email for discussions or suggestions. 
  
## License
FastPillars is release under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07). We also incorperate a large amount of code from [CenterPoint](https://github.com/tianweiy/CenterPoint). See the [NOTICE](docs/NOTICE) for details. Note that both nuScenes and Waymo datasets are under non-commercial licenses. 

## Citation
If you think our paper or code is helpful, please consider citing our work.
```
@article{zhou2023fastpillars,
  title={FastPillars: A Deployment-friendly Pillar-based 3D Detector},
  author={Zhou, Sifan and Tian, Zhi and Chu, Xiangxiang and Zhang, Xinyu and Zhang, Bo and Lu, Xiaobo and Feng, Chengjian and Jie, Zequn and Chiang, Patrick Yin and Ma, Lin},
  journal={arXiv preprint arXiv:2302.02367},
  year={2023}
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=StiphyJay/FastPillars&type=Date)](https://star-history.com/#StiphyJay/FastPillars&Date)


## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [mmcv](https://github.com/open-mmlab/mmcv)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [PillarNet](https://github.com/VISION-SJTU/PillarNet-LTS)
* [RepVGG](https://github.com/DingXiaoH/RepVGG)