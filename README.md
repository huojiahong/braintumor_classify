# braintumorCT_classify
--------------DATASET----------------------------
CT数据从kaggle下载
CT data from kaggle (https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri/data)
filename: Brain Tumor CT scan Images
          - healthy(2300)
          - tumor(2318)
if need data, download via kaggle

-------------MODEL------------------------------
自主搭建了一个简易的残差卷积网络：五层卷积，五次最大池化下采样224->7，其中1/2层和3/5层引进残差连接

