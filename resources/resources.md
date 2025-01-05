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
  - This thing is relevant in RNN BPTT, where we assume that the $W_{HH}$ is diagonalizable (its square obviously). During backprop, $W^kx$ appears which using the power method approximates for $\lim_{k \to \inf}$ to $\lambda ^kc_1v_1$ and if the eigenvalue is greater than 1, the gradient calculation diverges and if less than 1 it vanishes. Hence, there's an issue of vanishing and exploding gradients in RNN. Randomized truncation (concept similar to dropout, where the expected value is equal to the actual value) or time series truncation (however introduces inductive bias) can be used.
- [Intro to LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Good lecture on kernel methods](https://www.youtube.com/watch?v=XUj5JbQihlU&t=1553s)
. [Only Vapnik could do better than this (hopefully)](https://www.youtube.com/watch?v=eHsErlPJWUU)
- [Choosing the right K](https://cran.r-project.org/web/packages/cvms/vignettes/picking_the_number_of_folds_for_cross-validation.html)
