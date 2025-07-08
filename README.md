# braintumor_classify


--------------DATASET----------------------------

CT数据从kaggle下载

CT data from kaggle (https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri/data)
filename: Brain Tumor CT scan Images
          - healthy(2300)
          - tumor(2318)
if need data, download via kaggle


-------------MODEL------------------------------

自主搭建了一个简易的残差卷积网络：五层卷积，五次最大池化下采样224->7，其中1/2层和3/5层引进残差连接，保留底层特征，做到特征融合。
每一层卷积后进行BN，对每一批次的特征图进行标准化，稳定数据分布。之后使用常规的relu激活。
最后一层做全连接分类输出。二分类中选择输出大的值对应的属性作为类别结果。
采用Adam优化器，自适应调整每个参数学习率，经测试，在此过程中比SGD效果好。
Build a simple ResNet: five layers convolution, five times Maxpooling, initial size 224*224, downsample to 7*7.
Among them, residual connections are introduced in the 1/2 and 3/5 layers to retain the underlying features and achieve feature fusion.
After each layer of convolution, BN is performed to normalize the feature maps of each batch and stabilize the data distribution.Then activate it using relu. 
The last layer uses fully connected classification. In the binary classification, select the attribute corresponding to the larger output value as the category result: healthy or tumor.
The Adam optimizer is adopted to adaptively adjust the learning rate. It has been proved that it performs better then SGD in this model.


--------------RESULT------------------------------------

测试集准确率>97%，生成模型文件.pth和.onnx，进行C++部署，在网上和医院里找了几个图片，经验证均测试正确。
文件中还展示了混淆矩阵，模型结构。
Accuracy>97%. Generate Model with .pth and .onnx. Deploye the .onnx in C++.
The document also displays the confusion matrix and model structure.
