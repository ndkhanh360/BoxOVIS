<div align="center">

# Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2512.19088-b31b1b.svg)](https://arxiv.org/abs/2512.19088)
[![Workshop](https://img.shields.io/badge/AAAI_2026-Frontier_IR-blue)](https://frontier-ir-workshop.github.io/)

**Accepted to [AAAI 2026 Workshop on New Frontiers in Information Retrieval](https://frontier-ir-workshop.github.io/)**

</div>

## Abstract

Locating and retrieving objects from scene-level point clouds is a challenging problem with broad applications in robotics and augmented reality. This task is commonly formulated as open-vocabulary 3D instance segmentation. Although recent methods demonstrate strong performance, they depend heavily on SAM and CLIP to generate and classify 3D instance masks from images accompanying the point cloud, leading to substantial computational overhead and slow processing that limit their deployment in real-world settings. Open-YOLO 3D alleviates this issue by using a real-time 2D detector to classify class-agnostic masks produced directly from the point cloud by a pretrained 3D segmenter, eliminating the need for SAM and CLIP and significantly reducing inference time. However, Open-YOLO 3D often fails to generalize to object categories that appear infrequently in the 3D training data. In this paper, we propose a method that generates 3D instance masks for novel objects from RGB images guided by a 2D open-vocabulary detector. Our approach inherits the 2D detector's ability to recognize novel objects while maintaining efficient classification, enabling fast and accurate retrieval of rare instances from open-ended text queries.

## Code

The code will be made available here soon. Stay tuned!
