## Response to Cons 1
The goal of this paper is to provide a study to address the distribution/spatial misalignment among datasets. In addition to intrinsic, we also study extrinsic and the ego car center. We intend to provide a simple solution/baseline to the misalignment problem. 

## Response to Cons 2

We use the sampled point location as a heuristic to decide on an optimal receptive field. E.g., if a sampled point (determined by its x coordinate) is far from the ego vehicle, this module adjusts the receptive field to a small value. Vice versa, nearby points have large receptive fields. According to our experiments, simple extrinsic augmentation improves extrinsic robustness, but limits the performance, while the proposed Extrinsic Aware Module does not hurt performance while improving robustness. Here is the result:


| methods  | None | on y | on x | on x,y |
|----------|------|------|------|--------|
| Direct   | 36.3 | 34.7 | 27.9 | 26.9   |
| PoseAug  | 33.7 | -    | -    | 32.9   |
| With EAM | 36.9 | 35.5 | 35.4 | 34.9   |

**Table 1**: Experiments on nuScenes using DETR3D. We randomly shift the ego frame by a range within [-2m,2m] to simulate variation in extrinsic during testint. "PoseAug" means we also do random jittering on x,y during training.
## Response to Cons 3

In autonomous driving, intrinsic and extrinsic are readily available when sensors are calibrated, and they are given as prior knowledge. In fact, current automotive companies (such as Tesla) provide auto/re- calibration of sensors to deal with sensor alteration over time. 

## Response to Cons 4

From a high-level perspective, we aim to shift multiple datasets into the same distribution. The novelty of this paper is to study how to manipulate sensor parameters with simple solutions to achieve this goal. Our method does generalize to other methods, such as BEVDet, which is a depth-based method. 
In fact, recent SOTA 3D object detectors are based on DETR3D/BEVDet with little modification. DETR3D can achieve SOTA performance with a better backbone and carefully selected hyper-parameters. Image-based detector (such as FCOS3D) has certain limitations in 3D object detection: 
1) depth prediction is not perfect; 
2) rotation is hard to estimate in the 2D space; 
3) a monocular detector cannot leverage overlapped region of multi-view images, further hampering its depth estimation; 
4) a monocular method requires post-processing to remove redundant predictions as studied in DETR3D.
