## Problem 1
Thank you for pointing this out! We meant to use weather conditions as a motivating example to explain why current object detection models do not generalize: weather conditions covered by a single dataset are similar, which prevents trained models from generalizing to data in the wild.

For the augmentation, we intended not to use heavy data augmentation to avoid introducing more geometric bias for BEVDet. However, after careful examination, we found that we could avoid the bias by incorporating an intrinsic decoupled depth module, so we conducted additional experiments and updated BEVDet results with data augmentations. They are now available in the updated supplement with detailed data augmentation settings, and our alignment techniques are still effective under these settings.

## Problem 2

We have examined the data statistics and the performance of each class, shown in Table 2 below. We found the correlation between class distribution and model performance is weak. As shown in Table 1 below, Waymo has fewer cyclists, while the cyclist’s mAP is the highest among all datasets. Of course, our method achieves a much higher mAP in cyclists and/or pedestrians when using all datasets for training so that the potential data imbalance (if there is any) problem is mitigated. 

In KITTI-360, 'pedestrian' is labelled as 'person' and 'cyclist' is labelled as 'bicycle' and 'rider'. In the original text, we mentioned "lack of pedestrian and bicycle annotations in KITTI-360", where we meant to argue that pedestrians and bicycles are too small to detect in KITTI-360 since the focal length is very small. We have updated the statement in the main paper to avoid confusion.  

| Dataset | Vehicle | Pedestrian | Cyclist |
|---------|---------|------------|---------|
| N       | 70.3\%  | 26.6\%     | 3.1\%   |
| A       | 71.9\%  | 26.9\%     | 1.2\%   |
| L       | 93.4\%  | 3.7\%      | 2.9\%   |
| K       | 84.8\%  | 11.4\%     | 3.8\%   |
| K360    | 89.9\%  | 4.9\%      | 5.1\%   |
| W       | 64.3\%  | 34.7\%     | 1.0\%   |

**Table 1**: Percentage of each class in each dataset.

| Dataset | Single                | Multiple              |
|---------|-----------------------|-----------------------|
| N       | 36.3 (56.7/36.7/15.7) | 46.2 (66.6/42.7/29.2) |
| A       | 48.0 (73.8/38.7/31.7) | 53.7 (79.8/47.9/33.5) |
| L       | 37.3 (70.5/16.6/24.9) | 49.4 (76.7/33.0/38.4) |
| K       | 24.5 (40.2/25.1/8.3)  | 39.5 (54.2/36.1/28.0) |
| K360    | 26.1(60.2/4.5/13.7)   | 29.7 (60.6/9.4/19.2)  |
| W       | 58.8 (78.1/50.3/47.9) | 61.9 (82.0/55.4/48.2) |

**Table 2**: Evaluation results per dataset in the format: all(vehicles/pedestrians/bicycles). "Single" stands for training on each single dataset. "Multiple" stands for training on all six datasets.

## Problem 3
Per your suggestion, we have claimed S, s_pixel and p0 as scalars in the updated main paper. We provide step-by-step derivation of Eq. 1 and Eq. 3 **at the end of your attachment.**

## Problem 4
We assume that there is an optimal ego center position for every dataset, which will make the data distribution more aligned. We also assume that the cross-dataset performance would improve when the distribution is aligned, and vice versa. So we choose Waymo as a reference for alignment and shift the ego center position of every dataset to find the best performance in a manner of grid-searching. We then pick the shifting translation value of the best performance (the lightest grid points in Figure 4) as the optimal ego center position w.r.t. Waymo's ego center.

## Problem 5

The "camera rotation" in the limitation section means the rotation between a camera and the ego center. Manipulating such parameters requires a physical-based re-rendering of images, which is not easily done. We believe the rotation you asked is the absolute camera pose, which defines the transformation between a camera and the world center. The data used in this paper is collected on unconstrained roads with turns. 