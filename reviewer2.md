# Response to Weakness 1

DETR3D (query-based detectors) and BEVDet (depth estimation-based detectors) are the most popular and widely-used 3D object detectors. Recent works are built on these two methods and share many similarities to them. 
The detector architecture you mentioned is essentially a variant of FCOS3D. FCOS3D is a baseline studied in DETR3D and BEVDet and has been outperformed by these 3D-based detectors due to its several limitations:
1) Depth prediction is not perfect.
2) Rotation is hard to estimate in the 2D space.
3) A monocular detector cannot leverage overlapped regions of multi-view images, further hampering its depth estimation.
4) A monocular method requires post-processing to remove redundant predictions, as studied in DETR3D.
5) Depth prediction is coupled with camera intrinsics, as mentioned in other papers [1,2] and studied in our updated supplement.

We're glad that you agree the task studied in this paper is novel, and our method provides a strong improvement over popular works such as DETR3D and BEVDet. We will incorporate the discussion into our final version.  


[1] Park, Dennis, et al. "Is pseudo-lidar needed for monocular 3d object detection?."  ICCV 2021.

[2] Wang, Shuo, et al. "Towards Domain Generalization for Multi-view 3D Object Detection in Bird-Eye-View." CVPR 2023.


# Response to Weakness 2
We discussed such experiments in Sec 3.3 and provided results in Table 3. Indeed, adding more data improves model performance on a held-old dataset. We also include results below. The 'Direct' rows are performances of the models trained on multiple datasets with no cross-dataset techniques applied. The mAP improves with more data to some extent, but it improves more with cross-dataset alignments, such as Intrinsic Synchronization(Ksync), Extrinsic Aware Module(EAM) and Ego Frame Alignment(EgoSync).

| Setting           | src\ dst | N    | A    | L    | K    | K360 | W    | avg.S | avg.T |
|-------------------|-----------------------|------|------|------|------|------|------|-------|-------|
| Direct            | N                     | 36.3 | 0.8  | 1.8  | 0.0  | 0.0  | 1.1  | 36.3  | 0.7   |
| Direct            | +A                    | 40.5 | 49.2 | 0.5  | 0.0  | 0.0  | 5.2  | 44.9  | 1.4   |
| Direct            | +L                    | 41.6 | 50.5 | 43.7 | 0.0  | 0.0  | 3.8  | 45.3  | 1.3   |
| Direct            | +K                    | 41.5 | 49.7 | 46.0 | 41.4 | 1.1  | 3.6  | 44.6  | 2.4   |
| Direct            | +K360                 | 42.6 | 54.3 | 46.8 | 36.3 | 29.7 | 3.3  | 41.9  | 3.3   |
| Direct            | +W                    | 46.2 | 53.7 | 49.4 | 39.5 | 29.7 | 61.9 | 46.7  | -     |
| Ksync             | N                     | 40.8 | 25.5 | 18.6 | 29.7 | 18.0 | 23.4 | 40.8  | 23.0  |
| Ksync             | +A                    | 45.5 | 50.0 | 25.1 | 35.8 | 21.3 | 44.2 | 47.8  | 31.6  |
| Ksync             | +L                    | 46.8 | 53.2 | 55.1 | 37.8 | 23.1 | 45.3 | 51.7  | 35.4  |
| Ksync             | +K                    | 47.4 | 53.5 | 53.6 | 57.8 | 21.8 | 44.4 | 53.1  | 33.1  |
| Ksync             | +K360                 | 50.2 | 54.4 | 54.0 | 60.2 | 39.6 | 44.7 | 51.7  | 44.7  |
| Ksync             | +W                    | 51.8 | 55.3 | 56.6 | 61.9 | 40.7 | 63.7 | 55.0  | -     |
| Ksync+EAM+EgoSync | N                     | 43.1 | 33.6 | 32.8 | 33.0 | 18.4 | 33.0 | 43.1  | 30.2  |
| Ksync+EAM+EgoSync | +A                    | 52.1 | 52.7 | 38.4 | 42.2 | 23.2 | 40.7 | 52.4  | 36.1  |
| Ksync+EAM+EgoSync | +L                    | 52.6 | 53.2 | 59.5 | 46.1 | 26.1 | 43.6 | 55.1  | 38.6  |
| Ksync+EAM+EgoSync | +K                    | 51.0 | 54.7 | 60.2 | 63.9 | 28.4 | 44.6 | 57.5  | 36.5  |
| Ksync+EAM+EgoSync | +K360                 | 50.0 | 55.0 | 59.8 | 65.0 | 42.7 | 45.2 | 54.5  | 45.2  |
| Ksync+EAM+EgoSync | +W                    | 54.8 | 56.4 | 60.5 | 66.8 | 43.4 | 62.7 | 57.4  | -     |

**Table 1:** Performance of DETR3D trained on **multiple** datasets.  "Direct" means direct merge for training and direct transfer for testing. "avg.T" stands for the average in target domains. "avg.S" stands for the average in source domains. "Ksync", "EAM" and "EgoSync" stand for Intrinsic Synchronization, Extrinsic Aware Module and Ego Frame Alignment.


