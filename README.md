# Infrared_image_detection_via_apg
The detection is formulated as an optimization problem of recovering low_rank and sparse matrices via apg algorithm.  
# 整体过程
1. 通过对图像进行滑窗处理得到其patch image D.
2. 使用APG算法估计低秩的背景patch image B 和稀疏的目标patch image T.
3. 使用1D median filter ，从patch image 中重建出背景图像fb 和目标图像 ft.
4. 使用一个简单的分割方法去自适应地分割目标图像（去除虚警）.
5. 通过后处理，改进分割结果，得到最终的检测结果。
