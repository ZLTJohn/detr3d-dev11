# Response to question 1: AV task
>*... on other vision tasks for AV datasets ... do the authors hypothesize that similar sensor alignments are helpful there too? Why/why not? Have others done that already perhaps?*

AV perception usually involves object recognition (studied in this paper) and map prediction. We hypothesize that map prediction benefits from the same alignment method where accurate geometry estimation is required. There are fewer datasets containing mapping data and other papers have not studied such problems in mapping. We leave this as a future direction to study due to time limits and the efforts required to re-implement related map prediction models.   

# Response to question 2: clarifications

* Sorry for the typo related to Waymo/Argo2 results. We have corrected all the results and updated the main paper. 

* We updated the mAP-related statement in the introduction to avoid confusion.

* For data augmentation, we intended not to use heavy data augmentation to avoid introducing more geometric bias for BEVDet. However, after careful examination, we found that we could avoid the bias by incorporating an intrinsic decoupled depth module, so we conducted additional experiments and updated BEVDet results with data augmentations. They are now available in the updated supplement with detailed data augmentation settings, and our alignment techniques are still effective under these settings.

    For the detail of the intrinsic decoupled module, please refer to "Analysis and Results of BEVDet" in the updated supplement.

* MMDetection3D is a widely used 3D object detection codebase: https://github.com/open-mmlab/mmdetection3d. We added reference in the paper.

* For "thresholds at 05, 0.3 and 0.3", this might be a bug with openreview and we have verified that the text is correct on our end. 

* For the order of tables, we tried our best to re-arrange them in the updated main paper.

* We updated the text and captions to clarify the terminologies in the tables.

* Per your suggestion, we split the last column into "avg.S" and "avg-T", which means average performance on source domains ("avg-same") and target domains ("avg-not-same"). 

* The differences in intrinsic synchronization results for nuScenes comes from the different reference focal length for synchronization.

    In fact, we meant to state that our method works with different focal lengths. This has been analyzed in the supplement. We update the results with the same focal length (therefore, the same results) to avoid confusion.

* There are few cameras poses because most calibrated AV datasets use the same cars or the same type of car in each of them, so the camera poses (with reference to the ego center) in different frames are almost the same.

# Response to question 3: Evaluations
### a. More categories

> *... it would also have been good to add some results which include more than just those 3 categories ...*

AV perception focuses on dynamic objects, which usually are vehicles, bicycles, and pedestrians; they are all included in the study. We agree evaluating more categories is interesting; however, this requires a careful re-collection of data and is beyond the scope of this paper. We will leave it for future exploration.  

### b. Inclusion of intrinsic synchronization

> *It was not clear to me whether "Sync Extrinsic and Ego center" also includes intrinsic synchronization.*

Yes, intrinsic synchronization is used in the "Sync Extrinsic and Ego center". Performing "Sync Extrinsic and Ego center" without intrinsic synchronization fails in our experiments. We make it clear in the updated main paper.

### c. BEVDet Results

> *For the other BEVDet detector (SM), why did you only use the "Direct" approach for its evaluations, i.e. why did you not showcase your sensor adjustment strategy here?*

We have conducted additional experiments using BEVDet. In summary, our strategy is still effective in domain transfer and joint training with BEVDet. We achieve up to 10 mAP boost in domain transfer, while a 6.7 mAP gain in joint training compared to direct transfer or merging. Even though our method achieves strong performance, we still find that the metric is not fair to evaluate our strategy on BEVDet, since BEVDet omits altitude information and produces wrong heights. We use BEV-AP instead of 3D-AP to further demonstrate that our strategy corrects BEVDet's prediction in the x-y plane. BEVDet achieves 10.8 mAP gains when transferring nuScenes to other datasets. For more details, please refer to the updated supplement.

# Response to question 4: typos

Thank you for your comments. We have corrected all of them in the updated version of our paper. The updated main paper and supplement are available **in the attachment**.