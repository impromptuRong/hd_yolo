# Histological Based Multi-level Detection and Segmentation with Yolo and Hierarchical Loss


## Introduction
This repo provided a multi-level nuclei/nodule and 2nd structure detection and panoptic segmentation framework for digital patholgy slides. When given objects and masks at different amplification level, model will first assign different tasks into different detection and segmentation header and then refine the results based on a hierarchical confliction loss. 

For basic nuclei detection and segmentation, use the model defined in folder `model/metayolo`. For the hierarchical panoptic segmentation use`model/hnet`.
