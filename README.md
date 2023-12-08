# OQIF
AI-based Object-level Quantitative inversion for Zanthoxylum Rust Disease Index

For dataset construction, please refer to：
Dataset_Construction_for_CoCo_format.ipynb
CoCo_format_anno_viewer.ipynb

In order to train the object detection model, please refer to MMDetection：https://github.com/open-mmlab/mmdetection
This project only provides paper-related parameters in the mmdet_config folder.

Training and validation Regression_Branch, please refer to the documentation：
Train_Regression_Branch_ResNet_18.py
Train_Regression_Branch_ResNet_34.py
Train_Regression_Branch_ResNet_50.py
Train_Regression_Branch_ResNet_101.py
Test_Result_Train_Regression_Branch_ResNet.ipynb

For integrated object detection and image regression algorithms for holistic inference, please refer to：
Full_Map_Predict.ipynb
