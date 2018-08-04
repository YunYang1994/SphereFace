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
Let's revisit the original softmax loss:
<p align="center">
    <img width="70%" src="https://github.com/YunYang1994/SphereFace/blob/master/image/softmax.jpg" style="max-width:90%;">
    </a>
</p>


