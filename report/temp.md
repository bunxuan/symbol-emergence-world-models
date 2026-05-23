要从一开始吗，首先trajectory 到WorldModel nn.Linear(1, 32)  nn.ReLU(),将数据线性变换，分段函数 为了可以更好“猜出”函数 把高维特征（32 维）压缩到低维 latent网络学习数据的核心特征 用pred 算loss让函数学习，最后保留latent 后面处理

The world model begins by lifting the 1D trajectory into a higher-dimensional feature space using a 
linear layer (1 → 32). This expansion allows the encoder to represent different predictive regimes 
through distinct ReLU activation patterns. Because the encoder is piecewise-linear, each activation 
pattern corresponds to a different local linearization of the environment’s dynamics.

The prediction loss forces the model to organize the latent space according to the structure of the 
future trajectory. As a result, the latent representation becomes a compressed summary of the 
environment’s predictive structure. This latent space is then used for downstream mechanistic 
analysis, including PCA geometry, Jacobian discontinuities, flow-based invertible transformations, 
and diffusion-based generative reconstruction.

SVD本质是任意矩阵——因为它找到了那组特殊的输入方向 ，使得输出仍然正交且沿坐标轴拉伸 单位圆 → 椭圆。h2 = pca.fit_transform(h) 就是 中心化后以数据为点，拟合椭圆，这个椭圆本质就是一个圆线性变换来的，PCA 要找一条方差最大的方向，等价于找 XTX, 的最大特征向量。以X 数据描述为一个单位圆 的线性变换  而 SVD 直接，最大方向。如果 
v指向点云最伸展的方向（长轴方向），那么投影后的点会分散得很开，就会很大。如果 
v 指向点云最扁的方向（短轴方向），投影后的点会挤在一起，平方和就会很小。也就是 SVD 的第一右奇异向量，同时是 PCA 的第一主成分方向。为什么要中心化 因为 单位圆 的线性变换 是默认原点开始的，

PCA can be understood geometrically as fitting an ellipsoid to the centered data cloud. 
Centering is essential because PCA assumes that the linear transformation acts around the origin; 
without centering, the mean shift would be interpreted as variance.

SVD provides a direct geometric interpretation: any linear transformation maps the unit circle 
(or sphere) into an ellipse whose axes correspond to the singular vectors. The longest axis of 
this ellipse is the direction in which the transformation stretches the most. This direction is 
precisely the first right singular vector, which is also the first principal component.

Thus, PCA’s first component is the direction that maximizes the variance of the projected data, 
equivalently the direction that maximizes the quadratic form vᵀ(XᵀX)v. This is why PCA reduces to 
finding the dominant eigenvector of XᵀX, and why SVD provides the same result directly.


训练模型的过程 是算出loss，然后 求各各参数对其的导数，也就是变化一点loss会 怎么变化，因为要让loss逼近0最好，所以要反向减去，乘学习率，最后能慢慢逼近猜出来

Training proceeds by minimizing the prediction loss using gradient-based optimization. 
For each parameter, backpropagation computes the derivative of the loss with respect to 
that parameter, indicating how a small change would affect predictive error. The optimizer 
then updates parameters in the direction that most reduces the loss, scaled by a learning rate. 
Through repeated updates, the model gradually discovers a parameter configuration that best 
captures the predictive structure of the environment.


Jacobian 就是算 每个latent在x细微变化下的反应，当状态改变的时候，每个维度 不同 
x 处的梯度会不同，甚至出现跳变热力图会显示哪些 latent 维在哪些 
x 区间被激活。

The encoder Jacobian measures how each latent dimension responds to an infinitesimal change in the 
input x. Because the encoder is piecewise-linear, each ReLU activation pattern defines a distinct 
local linear map. When the environment undergoes an event boundary—such as a bounce—the activation 
pattern changes abruptly, causing a discontinuous jump in the Jacobian.

These Jacobian discontinuities reveal which latent dimensions become active in which regions of the 
input space. In the heatmap, this appears as sharp transitions where different subsets of latent 
units switch on or off. These transitions align precisely with the environment’s event boundaries, 
indicating that symbolic boundaries emerge from changes in the model’s predictive regime.

symbol_clustering 其实就是 以latent kmeans，分出不同状态，再用不同颜色画出来，这些就是不同state_machine

Symbol clustering is performed by applying k-means directly to the latent trajectory. Because the 
world model organizes the latent space into piecewise-linear segments, k-means naturally separates 
these segments into distinct predictive regimes. Each cluster corresponds to a symbolic state, and 
coloring the latent trajectory by cluster assignment reveals the underlying symbolic structure.

By examining the sequence of cluster assignments over time, we obtain a discrete symbolic 
state-transition sequence. Aggregating these transitions yields a symbolic state machine that 
captures the environment’s predictive dynamics.

Diffusion 我说一下我的理解 先是训练的时候 生成随机噪声，用q_sample一步算出实际，用eps_model预测 来训练模型，输入 t 和连续噪声 后的最终噪声。sample里 p_sample 还原出上一步？  

Because the reverse diffusion process reconstructs the data manifold by iteratively removing noise along the manifold’s principal directions, it reproduces the same piecewise‑smooth geometry as the original latent trajectory. At event boundaries, where the manifold’s tangent direction changes abruptly, the predicted noise also changes direction, causing the reverse trajectory to exhibit the same segmentation. Thus, symbolic boundaries emerge even in the stochastic reverse process.

flow 就是 把数据进去 通过 mask 变化 方便计算det，可以逆向 生成 z 特征, log_det 面积变化 。原本base_log_prob 和 缩放和z 算出保证密度一样。 概率模型逐渐把数据分布（复杂形状）映射到标准正态分布（圆盘

Flow-Based Models and Preservation of Symbolic Boundaries
RealNVP flow models consist of a sequence of invertible coupling layers. Each layer transforms the input by modifying only part of the variables while keeping the rest fixed, producing a triangular Jacobian whose log‑determinant is easy to compute. This structure allows the model to map a complex data distribution to a simple base distribution such as a standard Gaussian while preserving exact density through the change‑of‑variables formula.

Because each coupling layer is invertible, the overall transformation cannot merge or split regions of the latent space. As a result, the flow model preserves the topological structure of the latent manifold. When the original latent trajectory contains piecewise‑smooth segments separated by directional discontinuities—corresponding to different predictive regimes—the flow transformation maps these segments smoothly into the Gaussian space without creating new boundaries. Consequently, the symbolic segmentation observed in the flow latent space reflects the same underlying structure as in the original world model. 

















