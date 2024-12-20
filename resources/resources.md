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
