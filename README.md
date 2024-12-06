<!--
 * @Author       : Yuanting Ma
 * @Github       : https://github.com/YuantingMaSC
 * @LastEditors  : yuanting 
 * @Date         : 2024-12-06 09:16:50
 * @LastEditTime : 2024-12-06 09:17:30
 * @FilePath     : /JaunENet/README.md
 * @Description  : 
 * Copyright (c) 2024 by Yuanting_Ma@163.com, All Rights Reserved. 
-->
# JaunENet: An effective non-invasive detection of multi-class jaundice deep learning method with limited labeled data

Jaundice, caused by elevated bilirubin levels, manifests as yellow discoloration of the eyes, mucous membranes, and skin, often indicating conditions such as hepatitis or liver cancer. This study proposes a non-invasive, multi-class jaundice detection framework using weakly supervised pre-training on large-scale medical images, followed by transfer learning and fine-tuning on 450 collected jaundice cases. The model achieves exceptional performance on an independent test set (accuracy: 98.9%, sensitivity: 0.991, specificity: 0.999, AUC: 0.999, F1-score: 0.990) while requiring only 0.128 GFLOPs per image, making it efficient for mobile deployment. SHAP-based interpretability confirms its reliability in identifying subtle pathological features. Notably, weakly supervised pre-training outperforms detailed annotation-based approaches, providing valuable insights for small-sample deep learning in medical imaging.