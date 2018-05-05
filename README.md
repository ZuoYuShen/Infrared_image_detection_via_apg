# Infrared_image_detection_via_apg
The detection is formulated as an optimization problem of recovering low_rank and sparse matrices via apg algorithm.  

# 参考文献
[Infrared Patch-Image Model for Small Target Detection in a Single Image](https://ieeexplore.ieee.org/document/6595533/)
[APG(Accelerated Proximal Gradient)算法推导](http://www.docin.com/p-2104320979.html)

# 整体过程
1. 通过对图像进行滑窗处理得到其patch image D.
2. 使用APG算法估计低秩的背景patch image B 和稀疏的目标patch image T.
3. 使用1D median filter ，从patch image 中重建出背景图像fb 和目标图像 ft.
4. 使用一个简单的分割方法去自适应地分割目标图像（去除虚警）.
5. 通过后处理，改进分割结果，得到最终的检测结果。

# 步骤详解
程序中只涉及到第1到4步。
1. 对于滑窗处理，文章中指出滑窗的大小比步长更影响处理的结果，所以根据论文所述，选择80×80的滑窗进行处理，为了处理方便，取步长为8，这样滑窗到边缘后就不会出现溢出情况。
2. 对于APG算法的参数选择，论文选择lamda = 1/sqrt(max(m,n)), eta = 0.99, mu_0 = s2, mu_bar = 0.05s4. s2,s4是D中第二大和第四大的奇异值。但是在实际应用中，发现eta参数取0.9的影响不大，但是可以加速收敛，权衡之下，选择使用eta = 0.9。
3. 对于重建，将patch image 中的每一列按照patch在原图像中的位置，将patch image投影到一个三维矩阵中，判断三维矩阵的第三维度，若都为零，则该重建图像该位置的像素值就为0，若不全为0，则将第三维中像素为0的点去除，剩下的点中取中值作为该点的像素值。
4. 对于第4步，文章中根据重建图像的均值和方差来自适应选择阈值，但是观察发现，重建后的图像中的虚警为负数，所以，在算法中，我选择了以0为阈值，大于0即为目标。

# 使用
1. 程序中的矩阵都在GPU上定义的，在使用之前，先在命令行窗口下输入命令：gpuDevice查看是否支持该类型的矩阵运算。
2. 一张图片的处理为20s左右。
