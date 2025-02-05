- [Different types of differentation](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation): Talks about different types of differentiation and dives deep into backward and forward AD with its implementation in Python.
- [Really nice example of why use Taylor Series in ML/DL models](https://ainxt.co.in/importance-of-taylor-series-in-deep-learning-machine-learning-models/). It will be an easy read if you are already familiar with limit definition of partial derivatives, Linear Approximation (a special case of Taylor Series), Gradients, and Directional derivatives.
- [Quick refresher on Probabilities](https://blog.dailydoseofds.com/p/a-visual-guide-to-joint-marginal)
- [Nice resource for Expectations of r.v and functions](https://www.stat.auckland.ac.nz/~fewster/325/notes/ch3.pdf): This resource is a goldmine. Very intuitive explanations along with robust examples.
- [Reference for more theoretical statistics and probability theory](https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php): Very dense and heavy in theory
- [Resources for MLflow](https://levelup.gitconnected.com/mlflow-made-easy-your-beginners-guide-bf63f8fed915)
- [Resources for MLflow](https://dzone.com/articles/from-novice-to-advanced-in-mlflow-a-comprehensive)
- [Resources for MLflow](https://github.com/amesar/mlflow-examples/tree/master)
- [Resource for DVC](https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial)
- [Univariate statistics](https://www.jbstatistics.com/)
- [How does Pickle work](https://rushter.com/blog/pickle-serialization-internals/)
- Basics of CNN [Refresher if you have forgotten the concepts](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
- Upsampling before CV? [You are doing it wrong](https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html)
- [Bayesian Optimization](https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization/)
- What to do if negative predictions does not make sense to your modeling? [Use this](https://scikit-learn.org/1.5/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
  ```python
  # An example:
  from sklearn.compose import TransformedTargetRegressor
  from sklearn.linear_model import LinearRegression
  
  # Model pipeline with log transform and inverse exp automatically applied
  model = TransformedTargetRegressor(
      estimator=LinearRegression(),
      transformer=np.log,
      inverse_transform=np.exp
  )
  
  # Train your model with your data
  model.fit(X_train, y_train)
  ```
- Backpropagation in CNNs [Here](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
- Kernel methods [Here](https://alex.smola.org/papers/2002/SchSmo02b.pdf)
- [BatchNorm](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
- [Residual Blocks](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)
- [Revision lecture notes](https://cs231n.github.io/convolutional-networks/)
- [DL using SVM-paper](https://arxiv.org/pdf/1306.0239)
- [Crazy paper regarding CNN architecture Design](https://arxiv.org/pdf/2003.13678)
- [Multiblock PLS](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3618)
- [Loved this video on Pseudo-inverse](https://www.youtube.com/watch?v=DysbzsiBAdg&list=LL&index=6&t=1899s&ab_channel=AdamDhalla)
- [Power method for eigenvalue-eigenvector calculation](https://ergodic.ugr.es/cphys/lecciones/fortran/power_method.pdf)
- [Power method with code](https://lemesurierb.people.charleston.edu/introduction-to-numerical-methods-and-analysis-python.pdf)
  - The [power method](https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html) is quite insightful. Classical Chemometric methods like PCA and PLS often use power method (SVD would be quite expensive) and in chemometrics, since we just care about the first few components, power method is efficient in terms of both memory and speed. Even though, its used to find the dominant eigenvector, chemometric methods use deflation (removing the component corresponding to the first eigenvector) and repeating the power method. Interestingly, sklearn uses power method. [Check here](https://github.com/scikit-learn/scikit-learn/blob/6e9039160/sklearn/cross_decomposition/_pls.py#L58)
  - This thing is relevant in RNN BPTT, where we assume that the $W_{HH}$is diagonalizable (its square obviously). During backprop, $W^kx$appears which using the power method approximates for $\lim_{k \to \inf}$to $\lambda ^kc_1v_1$and if the eigenvalue is greater than 1, the gradient calculation diverges and if less than 1 it vanishes. Hence, there's an issue of vanishing and exploding gradients in RNN. Randomized truncation (concept similar to dropout, where the expected value is equal to the actual value) or time series truncation (however introduces inductive bias) can be used.
- [Intro to LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Were RNNs all we needed?](https://arxiv.org/pdf/2410.01201)
- [Good lecture on kernel methods](https://www.youtube.com/watch?v=XUj5JbQihlU&t=1553s)
. [Only Vapnik could do better than this (hopefully)](https://www.youtube.com/watch?v=eHsErlPJWUU)
- [Choosing the right K](https://cran.r-project.org/web/packages/cvms/vignettes/picking_the_number_of_folds_for_cross-validation.html)
- [Practical tips on using SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
- [Nice blog on GPs, I recommend after finishing GP section from d2l book](https://yugeten.github.io/posts/2019/09/GP/)
- [Gaussian Process Classifier using NumPy](https://krasserm.github.io/2020/11/04/gaussian-processes-classification/)
- [Just a refresher on polar coordinates](https://web.ma.utexas.edu/users/m408m/Display15-4-2.shtml)
- [Geometric intuition of Determinant](https://www.youtube.com/watch?v=xX7qBVa9cQU)
  - When you apply a transformation matrix 𝐴A to the unit cube (in 3D) or unit square (in 2D), the actual volume (or area) change is given by the absolute value of the determinant of the matrix. Remember The Jacobian determinant represents how much the area (or volume, in higher dimensions) changes when you transform from one coordinate system to another. Example: In multivariable calculus $dxdy = rdrd\theta$where r is the determinant of the Jacobian. 
- [Some matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)
- [GP and inversion of kernel matrix](https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion)
  Kernel matrix is positive-semidefinite (which means [cholesky](https://zief0002.github.io/matrix-algebra/cholesky-decompostion.html) is not valid when one of the eigenvalue is zero) but we add some noise(not a problem for noise-added GP but a problem for noise-free GP).
- [MMD and kernels](https://www.youtube.com/watch?v=zFffYuDGslg)
- [Entire book on GP](https://gaussianprocess.org/gpml/chapters/RW.pdf): Make sure you have skimmed through chapter 6 of PRML before leaping into this new venture
- [Linear Algebra primer](https://pabloinsente.github.io/intro-linear-algebra#vector-null-space)
- PCA (on A.T  !!!) vs Gram-Schimdt (on A !!!) (or let's say QR decomposition): If your goal is to find the set of orthonormal basis vectors, all of them will work for you. Well, although they might be different because there are infintely many but they will do the job (i.e. spanning the column space of a matrix, let's say A). But PCA comes with a overhead of finding the set of orthonormal basis vectors in the order of decreasing magnitude of projected vector onto that vector (a.k.a less variance explained as we proceed further), so it's a slower compared to G-S (or QR). You might also find some similarities between how PCA and G-S works (I am talking about the NIPALS here and yeah, it's the deflation step). So PCA (original on A) basis vectors of the row space whereas for the GS and QR its the basis vectors of the column space.
- [Model Evaluation, Model Selection, and Algorithm
Selection in Machine Learning: Paper by our own Sebastian Raschka](https://arxiv.org/pdf/1811.12808)
  - [Sklearn docs regarding model comparison](https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_stats.html)
- [Absolutely beautiful approach to underfitting and overfitting](https://youtu.be/Yan7y3N4V3c?si=faudgmMIhkVatrj8)
- Stuffs on Bias-Variance:
  - (https://www.bradyneal.com/bias-variance-tradeoff-textbooks-update)
  - (https://mlu-explain.github.io/double-descent/)
  - (https://arxiv.org/pdf/1812.11118)
  - (http://doursat.free.fr/docs/Geman_Bienenstock_Doursat_1992_bv_NeurComp.pdf)
  - (https://www.casact.org/sites/default/files/2021-07/Bias-Variance-Tradeoff-Brady-Brockmeier.pdf)
- [Boosting](https://arxiv.org/pdf/1403.1452)
- [InstanceHardnessCV](https://fritshermans.github.io/posts/hard_to_classify_datapoints.html)
- [PLS-DA class weights and multiclass](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3030)
- [Algorithmic details with Linear Algebra in depth. Super compact](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=87b7455b4a1ebd6f354cf23233435f81215f25d1)
- [Notes on Rayleigh quotient](https://www.sjsu.edu/faculty/guangliang.chen/Math250/lec4RQ.pdf)
----------
### Explanation of True Risk, Empirical Risk, Estimation Error, and Approximation Error

We aim to find a function that minimizes the **true risk** (the Bayes predictor, $f^\ast$). However, since we only have access to a finite dataset, we are forced to minimize the **empirical risk**, which serves as an unbiased estimator of the true risk. To understand this, consider the bias:  
$$\text{Bias} = \mathbb{E}[\hat{\theta}] - \theta$$ 
where $\hat{\theta}$ is the estimator derived from empirical risk minimization, and $\theta$ is the true parameter.

Given the dataset and a specific algorithm, we obtain a function $f_N$ that minimizes the empirical risk. While we want $f_N $to minimize the true risk and match $f^\ast$ , the function $\hat{f}$ found by the algorithm may differ from $f^*$ due to model limitations and the dataset used.

#### Estimation and Approximation Errors:
- The difference in true risk between $f_N$(the function found via empirical risk minimization) and $\hat{f}$ (the best possible function the algorithm can find) is called the **estimation error**.
- The difference in true risk between $\hat{f}$ and $f^\ast$ is called the **approximation error**.

#### Hypothesis Class and Function Space:
Consider a big circle representing the set of all possible functions generated by different algorithms ($\mathcal{F}$). The hypothesis class $\mathcal{H}$ is a subset of $\mathcal{F}$, representing all possible functions the chosen algorithm and model can generate. This smaller inner circle depends on the model's structure and parameters.

The hypothesis class is not restricted to functions that fit the data but encompasses all functions the model can represent. You can define the hypothesis class:
1. By fixing specific hyperparameters (e.g., the degree of a polynomial or number of neurons in a neural network), resulting in a smaller, more focused $\mathcal{H}$.
2. By allowing all possible combinations of parameters and hyperparameters, which creates a larger $\mathcal{H}$.

#### Relationship Between Hypothesis Class, Approximation Error, and Estimation Error:
- If $f^\ast$ lies **outside the hypothesis class** (e.g., using a simple model with a small hypothesis class), the approximation error will be large because no function in $\mathcal{H}$ can approximate $f^\ast $ well.
- However, with a small $\mathcal{H}$, the estimation error tends to be small. Functions minimizing empirical risk across different datasets are less variable because the hypothesis class is restricted. This means $f_N$ is closer to $\hat{f}$.

- Conversely, with a **complex model**, the hypothesis class $\mathcal{H}$ is large, allowing it to approximate $f^*$ more closely. The approximation error decreases because the larger hypothesis class increases the likelihood of containing a function close to $f^\ast$. 
- However, the estimation error is larger because the choice of $f_N$ is more sensitive to variations in the dataset. For example, in polynomial regression with many parameters and few data points, infinitely many curves may pass through the points, resulting in high variability.

#### Effect of Increasing the Dataset:
As the dataset size $N$ increases:
- The empirical risk $R_{\text{emp}}(f)$ approaches the true risk $R(f)$.
- $f_N$, the function minimizing empirical risk, becomes closer to $\hat{f}$, reducing the **estimation error**. The larger dataset ensures that minimizing empirical risk aligns more closely with minimizing true risk.
- The hypothesis class $\mathcal{H}$ remains unchanged; the inner circle does not shrink. However, the larger dataset helps identify functions in $\mathcal{H}$ that are closer to $\hat{f}$, improving generalization.

### Key Clarifications and Adjustments:
1. **Approximation and Estimation Errors**:
   - Approximation error depends on the hypothesis class and whether it can express functions close to $f^*$.
   - Estimation error depends on the size of the dataset and the complexity of the hypothesis class.

2. **Dataset Size and $\mathcal{H}$**:
   - Increasing the dataset size improves the alignment between $f_N $and $\hat{f}$, reducing estimation error.
   - The size of $\mathcal{H}$ (the hypothesis class) is fixed for a given model and algorithm. Larger datasets do not change $\mathcal{H}$, but they improve the selection of $f_N$ within $\mathcal{H}$.

### Visualization:
- The larger hypothesis class allows approximation closer to $f^*$, but it also leads to higher variability (estimation error).
- A small hypothesis class has less variability but struggles to approximate $f^*$ if it lies outside $\mathcal{H}$.
 ![image](https://github.com/user-attachments/assets/960ed639-7957-4de8-81ec-9cd1575937b8)
--------------------------
