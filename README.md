|Author|YunYang1994|
|---|---
|E-mail|dreameryangyun@sjtu.edu.cn


SphereFace
===========================
This is a quick implementation for [Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1805.09298).This paper proposed the angular softmax loss that enables convolutional neural networks(CNNs) to learn angularly discriminative features. The main content I replicated contains: <br>

- **1**. mathematical comparison among original softmax, modified softmax and angular softmax;
- **2**. show the accuracy and loss comparison of different softmax in the experiment;
- **3**. 2D and 3D visualization of embeddings learned with differnet softmax loss on MNIST dataset;

### softmax loss
|softmax|formula|
|---|:---:|
|original softmax|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/original_softmax.png)|
|modified softmax|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/modified_softmax.png)|
|angular softmax|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/angular_softmax.png)|

### 2D visualization
|original softmax|modified softmax|angular softmax|
|---|---|:---:|
|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/2D_Original_Softmax_Loss_embeddings.jpg)|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/2D_Modified_Softmax_Loss_embeddings.jpg)|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/2D_Angular_Softmax_Loss_embeddings.jpg)|

### 3D visualization
|original softmax|modified softmax|angular softmax|
|---|---|:---:|
|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/3D_Original_Softmax_Loss_embeddings.jpg)|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/3D_Modified_Softmax_Loss_embeddings.jpg)|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/3D_Angular_Softmax_Loss_embeddings.jpg)|
