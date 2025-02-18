---
library_name: transformers
tags:
- pneumonia
- chest_x_ray
- medical_imaging
- radiology
base_model:
- timm/tf_efficientnetv2_s.in21k_ft_in1k
pipeline_tag: image-classification
---

This model performs binary classification and segmentation for pneumonia (lung opacity) in frontal chest radiographs. 
It is a `tf_efficientnetv2_s` backbone with a U-Net decoder and linear classification head.
The model was trained on the [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) and the [SIIM-FISABIO-RSNA COVID-19 Detection dataset](https://www.kaggle.com/c/siim-covid19-detection).
Both of these datasets were annotated with bounding boxes, which were converted to ellipsoid segmentation masks. 

Classification performance on a holdout test set of 1,334 images from the RSNA dataset and 317 images from the SIIM-FISABIO-RSNA dataset:
```
RSNA + SIIM-FISABIO-RSNA (n=1,651): AUC 0.900
                    RSNA (n=1,334): AUC 0.885
       SIIM-FISABIO-RSNA (n=317)  : AUC 0.914
```