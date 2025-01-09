## Kernel methods
- Idea: Use large (possibly infinite) set of fixed non-linear basis functions
- Normally, complexity depends on number of basis functions, but by a "dual trick", complexity depends on the amount of data.
- Examples:
    - Gaussian Processes
    - SVM
    - Kernel Perceptron
    - Kernel PCA

### Kernel Function
- Let $\phi(x)$ be a set of basis functions that map inputs $x$ to a feature space.
- In many algorithms, this feature space only appears in the dot product $\phi(x)^T\phi(x')$ of input pairs $x, x'$.
- Define the kernel function $k(x, x') = \phi(x)^T\phi(x')$ to be dot product of an pair $x, x'$ in the new space.
    - We only need to know $k(x, x')$, not $\phi(x)$.
- Let $\Phi(x)$ = $[\phi(x_1), \phi(x_2), ..., \phi(x_N)]$. 
- Let $K = \Phi(x)^T \Phi(x)$ be the **Gram matrix**. It is essentially the matrix of pairwise dot products of data points in the transformed feature space defined by the kernel function.

### Constructing Kernels
- Two possibilites: 
    - Finding mapping $\Phi(x)$ to feature space and let $K = \Phi(x)^T \Phi(x)$. 
    - Or, directly specify $K$.
- Can any function that takes two arguments serve as a kernel?
     - **No**, a valid kernel must be positive semi-definite, which means all the eigenvalues should be $\geq 0$. *(Don't confuse this with kernels (filters) in CNN. Those kernels are learned from the data and are not constrained to be positive semi-definite)*
     - In other words, $K$ must factor into product of a transposed matrix by itself (e.g, $K = \Phi(x)^T \Phi(x)$)

- Example:
    $$\text{Let } k(x, z) = (x^Tz)^2 \text{ with } x = \begin{bmatrix}x_1 \\ x_2 \end{bmatrix} \text{ and } z = \begin{bmatrix}z_1 \\ z_2 \end{bmatrix}$$
- Basically, we are going to take dot product in the original space and square it. So the question, is this a valid kernel function?
- In other words, is there another mapping that would allow me to go to another space and take a dot product in that space and it correspond to doing this? 
    $$ k(x, z) = (x^Tz)^2 $$
    $$ = (x_1z_1 + x_2z_2)^2 $$
    $$ = (x_1^2z_1^2 + x_2^2z_2^2 + 2x_1x_2z_1z_2) $$
    $$ = \begin{bmatrix} x_1^2,\ \sqrt{2}x_1x_2,\ x_2^2 \end{bmatrix}
     \begin{bmatrix} z_1^2\\ \sqrt{2}z_1z_2\\ z_2^2 \end{bmatrix} $$
- Now let, $\phi_1(x) = x_1^2$, $\phi_2(x) = \sqrt{2}x_1x_2$, and $\phi_3(x) = x_2^2$
    $$ = \phi(x)^T \phi(z) $$
- Now this is really nice. Why?
    - *Because in reality we are just taking the dot product in the original space and raising by power 2. But, implicity, we are creating 3 different basis functions and taking the dot product of those function values. So the power is that with kernels, we make it as if we are working in the original space but implicity, it's mapping the inputs into higher dimension allowing us to capture non-linear relationships.*
- This is basically called the kernel trick, where computations in the original space implicity correspond to operations in a higher-dimensional space.
- It’s called a higher dimension because the kernel function implicitly maps the input data points from their original space (input space) to a new space (feature space) that has more dimensions than the original. For example, $x$ originally has dimension 2, whereas, the $\phi(x)$ has dimension of 3.
- Imagine points on a 2D plane being projected into a 3D space. In the new 3D space, those points can have relationships (e.g., separability) that were not possible in the original 2D space.
- Just like dot product in original space has a notion of similarity between two vectors, the kernel computation carries the same meaning.
- We showed that the kernel function $k$ can be written as a dot product of a vector and its transpose. But will $K = \Phi(x)^T \Phi(x)$ hold?
    - Always holds in theory if the kernel functions corresponds to a valid feature map $\phi(x)$.
    - Consider a set of $n$ points $X = {x_1, x_2, ..., x_n}$, now the kernel matrix is defined as $K_{ij} = k(x_i,x_j)$.
    - If we assemble the feature vectors $\phi(x_1), \phi(x_2), ..., \phi(x_n)$ into a feature map $\Phi(x)$, then:
    $$ K = \Phi(x)^T\Phi(x) $$
    - The Kernel matrix $K$ is a mathematical representation of the implicit feature space relationships, but we compute it directly using the kernel function $k(x, x')$, without ever explicity constructing $\Phi(x)$. 
- The most common kernel is Gaussian kernel:
    $$k(x, x') = \exp\left(-\frac{||x-x'||^2}{2\sigma^2}\right)$$

- [About the notation in Latex](https://tex.stackexchange.com/questions/254785/e-vs-exp-in-display-mode)

- Why is Gaussian kernel a valid kernel?
    $$k(x, x') = \exp\left(-\frac{||x-x'||^2}{2\sigma^2}\right)$$
    $$ = \exp\left(\frac{-x^Tx}{2\sigma^2}\right)\exp\left(\frac{x^Tx'}{\sigma^2}\right)\exp\left(\frac{-x'^Tx'}{2\sigma^2}\right) $$
    - So, as you can see a kernel is valid if $f(x)(k_1(x, x'))f(x')$, where $f(x)$ is any function. Also $\exp(k_1(x, x'))$ is a valid kernel. So we just need to show that $x^Tx$ is a valid kernel, because multiplying a kernel by a constant is a kernel as well.
    Now, $x^Tx'$ is a valid kernel because it satisfies the $x^TAx'$ where A will be identity matrix in that case.
    - Regarding the rules, [consult here, Page 296](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- If we perform similar computation to this kernel, we will end up with having infinite number of basis functions. And the beautiful thing is, we will be able to do this without paying the price computationally.
- Let's see a simple proof of this for the simplest case where $x$ and $x'$ **are scalers**. Also let's assume $\sigma = 1$.
    $$ = K(x, x') = \exp(-(x-x')^2) $$
    $$ = \exp(-x^2)\exp(-x'^2) \exp(2x^Tx')$$
    $$ = \exp(-x^2)\exp(-x'^2)\sum_{0}^{\infty}\frac{2^k(x)^k(x')^k}{k!}$$
- The final part is the taylor expansion.
- We can see that its basically the dot product of two vectors. The first vector is $(x)^k \exp(-x^2)$ where $k=0,..., \infty$ and the second vector is $(x')^k \exp(-x'^2)$ where $k=0,...,\infty$. Hence, the gaussian kernel is basically the dot product of infinte vectors.
- Again, this is for scalars, in case $x$ is a vector, it will include the interaction terms. The infinite-dimensional feature expansion for vectors involves interactions between the components of $x$ and $x'$, so each term in the Taylor expansion will have products of powers of the components of $x$ and $x'$ weighted by the appropriate Gaussian factor. This adds interaction terms between the features of the input vectors, leading to a much richer feature space.

## Gaussian process

Formally, A **Gaussian process** is defined as a collection of random variables, any finite number of which have a joint Gaussian distribution. If a function $f(x)$ is a Gaussian process, with mean function $m(x)$ and covariance function or kernel $k(x, x'), f(x) \sim \mathcal{GP}(m, k)$, then any collection of function values queried at any collection of input points $x$ has a joint multivariate Gaussian distribution with mean vector $\mu$ and covariance matrix $K: f(x_1), ..., f(x_n) \sim \mathcal{N}(\mu, K)$, where $u_i = E[f(x_i)] = m(x_i)$ and $K_{ij} = \mathrm{Cov}(f(x_i), f(x_j)) = k(x_i, x_j)$.

Informally, A gaussian process ($\mathcal{GP}$) is a way to model a distribution over functions. For any input point $x$, the function value $f(x)$ is a random variable following a Gaussian distribution. For any set of input points $x_1, x_2, ..., x_n$, the corresponding function values $f(x_1), f(x_2), ..., f(x_n)$ joinlty follow a multivariate Gaussian distribution, with a mean vector $\mu$ and a covariance matrix $K$ determined by the mean function $m(x)$ and covariance function $k(x, x')$.


**So, Is multivariate Gaussian distribution a Gaussian process then?**

**Well, No.** A gaussian process is a 'generalization' of the Multivariate Gaussian Distribution (MGD). In MGD the number of random variables is finite. In other words, fixed number of $x$'s. However, in $\mathcal{GP}$, the number of random variables are infinite. It models a function $f(x)$, where $x$ can take on any value in a continuous input space (infinite possibilites).

In other words, in MGD, the random variables are the finite $x$'s themselves. We deal with finite-dimensional vector of random variables $[x_1, x_2, ..., x_n].$
Whereas, in $\mathcal{GP}$, the random variables are the function values $f(x)$, not $x$. For any given $x$, $f(x)$ is a random variable, and the $\mathcal{GP}$ specifies the joint distribution of $f(x)$ values over all $x$. The $\mathcal{GP}$ is trying to model any function that could fit the observed data, and for that, it assumes that there are infinitely many possible input points where the function could be evaluated. So, even though we only have a finite set of data points (in practice), the GP imagines that the function exists for any $x$ in the input space. Now, if we limit ourselves to a finite number of x's, we end up with finite $f(x)'s$ each with a Gaussian distribution and jointly multivariate gaussian distribution.

Any function 

$$ f(x) = w^T \phi(x) $$
with  $w$ drawn from a Gaussian distribution, and $\phi$ being any vector of basis functions, for example $\phi(x) = (1, x, x^2, ..., x^d)^T$, is a Gaussian process. Moreover, any Gaussian process $f(x)$ can be expressed in the form above.


### A simple Gaussian process
Suppose $f(x) = w_0 + w_1x$, and $w_0, w_1 \sim \mathcal{N}(0, 1)$, with $w_0, w_1, x$ all in one dimension. It can be written as the inner product $f(x) = (w_0, w_1)(1, x)^T$. So it can be written in the form $ f(x) = w^T \phi(x) $ where $w=(w_0, w_1)^T$ and $\phi(x) = (1, x)^T$.

For any $x$, $f(x)$ is a sum of two Gaussian random variables. Since Gaussians are closed under addition, $f(x)$ is also a Gaussian random variable for any $x$. 

When we say that a distribution is closed under addition, we are referring to a property of a set, function, or distribution where the addition of any two elements within that set (or distribution) results in another element that is still within the same set (or distribution).

We can compute **for any particular $x$** that $f(x)$ is $\mathcal{N}(0, 1+x^2)$. 

**Proof:**

$$ \mathbb{E}[f(x)] = \mathbb{E}[w_0] + \mathbb{E}[w_1x] $$
$$ \mathbb{E}[f(x)] = \mathbb{E}[w_0] + x\cdot\mathbb{E}[w_1] $$
$$ \mathbb{E}[f(x)] = 0 + x\cdot0 $$
$$ \mathbb{E}[f(x)] = 0 $$

*Similarly,* 
$$ \mathbb{V}[f(x)] = \mathbb{V}[w_0] + \mathbb{V}[w_1x] $$
$$ \mathbb{V}[f(x)] = \mathbb{V}[w_0] + x^2\cdot \mathbb{V}[w_1] $$
$$ \mathbb{V}[f(x)] = 1 + x^2\cdot 1 $$
$$ \mathbb{V}[f(x)] = 1 + x^2 $$


And like said previously, the joint distribution for any collection of function values, $(f(x_1), ..., f(x_n))$, for any **finite** collection of inputs $x_1, ..., x_n$, is a multivariate Gaussian distribution. Therefore $f(x)$ is a Gaussian process.

In short, $f(x)$ is a random function, or *a distribution over functions*. We can gain some insights into this distribution by repeatedly sampling values for $w_0, w_1$, and visualizing the corresponding functions $f(x)$, which are straight lines with slopes and different intercepts. **This shows how a distribution over parameters in a model induces a distribution over function.** While we often have ideas about the function we want to model $-$ whether they're smooth, periodic, etc. $-$ it is relatively tedious to reason about the parameters, which are largely uninterpretable. Fortunately, Gaussian processes provide an easy mechanism to reason directly about functions. Since, a Gaussian distribution is entirely defined by its **first two moments**, its mean and covariance matrix, a Gaussian process by extension is defined by its **mean function and covariance function.**


----
**In MGD, we deal with a finite number of random variables, so the covariance matrix $K$ is explicitly defined. If we have $n$ random variables $x_1, x_2, ..., x_n$, we construct $n\times n$ covariance $K$, where each entry $k_{ij} = Cov(x_i, x_j)$. Well, in $\mathcal{GP}$, there are infinite $x$ and thus infinite $f(x)$ random variables. So, how do we define covariance matrix for the $\mathcal{GP}$ then?**

In $\mathcal{GP}$, we are working with infinite number of random variables because the input space $x$ is continuo
us. Instead of a fixed covariance matrix, we define a covariance matrix, also called a kernel function $k(x, x')$, which specifies the covariance between the random variables $f(x)$ and $f(x')$ for an pair of input $x$ and $x'$. While the $\mathcal{GP}$ is defined over an infinite $x$, in practice, we only evaluate the $\mathcal{GP}$ on a **finite subset of points** (e.g., training data or query points).

For $n$ input points $x_1, x_2, ..., x_n$, the kernel function $k(x, x')$ is evaluated pairwise for all points to produce a covariance matrix $K$, where $K_{ij} = k(x_i, i_j)$. This $K$ is a finite $n\times n$ matrix and must be positive semi-definite (PSD) to ensure the joint multivariate Gaussian distribution is valid.

Let's suppose we use the RBF kernel: $k(x, x') = \sigma^2\exp(\frac{||x-x'||^2}{2l^2})$, where $\sigma^2$ is the variance and $l$ is the length scale.
- Now, if we have $x = [x_1, x_2, x_3]$, the covariance matix becomes: 
$$ K = \begin{bmatrix} k(x_1, x_1) \ k(x_1, x_2) \ k(x_1, x_3) \\ k(x_2, x_1) \ k(x_2, x_2) \ k(x_2, x_3)\\ k(x_3, x_1) \ k(x_3, x_2) \ k(x_3, x_3) \end{bmatrix}$$
- If we later evaluate the GP on a new point $x^4$, we can compute its covariance with all existing points using $k(x4, x_i$)$ and augment the covariance matrix. (We will come to this later.)
- The off-diagonal expressions tells us how correlated the function values will be $-$ how strongly determined $f(x_1)$ will be from $f(x_2)$ and $f(x_3)$.

In the above example, the mean function:

$$m(x) = \mathbb{E}[w_0+w_1\cdot x] = \mathbb{E}[w_0] + \mathbb{E}[w_1]x = 0+0 =0$$

Similarly, the covariance function is:
$$k(x, x') = \mathrm{Cov}(f(x), f(x')) = \mathbb{E}[f(x)f(x')]-E[f(x)f(x')]$$
$$k(x, x') = \mathbb{E}[w_0^2+w_0w_1x'+w_1w_0x+w_1^2xx'] = 1+xx' $$

Now this means, distribution over functions can be directly specified and sampled from, without needing to sample from the distribution over parameters. For example, to draw from $f(x)$, we can simply form our multivariate Gaussian distribution associated with any collection of x we want to query, and sample from it directly. 

Let's see the same derivation for any model of the form $f(x)=w^T \phi(x)$, with $w \sim \mathcal{N}(\mu, S)$ ($w$ is joint multivariate gaussian distributed with mean vector $\mu$ and covariance matrix $S$)

$$ \mathbb{E}[f(x)] = \mathbb{E}[w^T\phi(x)] = \mathbb{E}[f(x)] \cdot \phi(x)$$
$$ \mathbb{E}[f(x)] = \mu ^T \phi(x) $$

Similarly, for the covariance function:
$$k(x, x') = \mathrm{Cov}(f(x), f(x')) = \mathbb{E}[f(x)f(x')]-\mathbb{E}[f(x)]\mathbb{E}[f(x')]$$
So,
$$\mathbb{E}[f(x)f(x')] = (w^T\phi(x))(w^T\phi(x')) = (\phi(x)^Tw)(w^T \phi(x'))$$
We can do this because they are basically constants (dot products).
Therefore,
$$\mathbb{E}[f(x)f(x')] = \phi(x)^T \mathbb{E}[ww^T] \phi(x')$$

And we know from the properties of the multivariate Gaussian distribution that, the expected value of $ww^T$ is:
$$ \mathbb{E}[ww^T] = S + uu^T$$

So, the first term $\mathbb{E}[f(x)f(x')]$ becomes $[\phi(x)^T (S+uu^T)\phi(x')]$ which on expansion becomes (with slight manipulation like before):
$$\mathbb{E}[f(x)f(x')] = \phi(x)^TS\phi(x') + (u^T\phi(x))(u^T\phi(x'))$$

Similarly, the other term $\mathbb{E}[f(x)]\mathbb{E}(f(x'))$ becomes:
 $$\mathbb{E}[f(x)]\mathbb{E}(f(x')) = (u^T\phi(x))(u^T\phi(x'))$$

Finally:
$$k(x, x') = \phi(x)^TS\phi(x') + (u^T\phi(x))(u^T\phi(x')) - (u^T\phi(x))(u^T\phi(x'))= \phi(x)^TS\phi(x')$$

Now, since $\phi(x)$ can represent a vector of any non-linear basis functions, we are considering a very general model class, including models with an infinite number of parameters. 

**To summarize, a $\mathcal{GP}$ simply says that any collection of function values $f(x_1), f(x_2), ..., f(x_n)$, indexed by any collection of inputs $x_1, ..., x_n$ has a joint multivariate Gaussian distribution. The mean vector $\mu$ of this distribution is given by a *mean* function, which typically taken to be a constant or zero. The covariance matrix of this distribution is given by the *kernel* evaluated at all pairs of the inputs $x$.**

----
### RBF Kernel
The properties of the Gaussian process that we used to fit the data are strongly controlled by what's called a *covariance function*, also known as a *kernel*. The Radial Basis Function (RBF, a.k.a. squared exponential, Gaussian), which has the form

$$ k_{RBF}(x, x') = \mathrm{Cov}(f(x), f(x')) = a^2\exp\left(-\frac{1}{2l^2}||x-x'||^2\right) $$

The hyperparameters are interpretable. The amplitude parameter $a$ controls the vertical scale over which the function is varying, and the *length-scale* parameter $l$ controls the rate of variation (the wiggliness) of the function. Larger $a$ means larger function values, and larger $l$ means more slowly varying functions. 

The *length-scale* has a particularly pronounced effct on the predictions and uncertainty of the $\mathcal{GP}$. At $||x-x'||=l$, the covariance between a pair of function values is $a^2e^{-0.5}$. At larger distance than $l$ i.e, $||x-x'|| \gt l$, the function value diminshes implying if we want to make a prediction at a point $x$, then function values with inputs $x$ such that $||x-x'||\gt l$ will not have a strong effect on our predictions.

It should be noted that as the *length-scale* increases the 'wiggliness' of the function decrease, and our uncertainty decreases. **If the length-scale is small, the uncertainty will quickly increase as we move away from the data, as the datapoints will be less informative about the function values.**

<!-- Before proceeding further, let's understand how we fit the data with a Gaussian process. 

In Gaussian Processes, the first step is to define a prior distribution over the types of functions we consider reasonable. This prior doesn’t focus on fitting the data itself, but rather on specifying broad characteristics of potential solutions, such as how smoothly the functions can vary with the input values.

Once we observe the data, we condition the prior on this information, updating our belief about the functions. This gives us a posterior distribution over the set of functions that are consistent with both the prior and the observed data. -->

Well, I have already shown above why RBF intrinsically maps the input to infinite dimension but let's see it again but this time starting from the weight space. We will see how we can recover this kernel by considering a linear model with infinite feature expansions. Let's consider the linear function.

$$ f(x_i) = w^T\phi(x), \space w \sim \mathcal{N}(0, \sigma^2I) $$

where $\phi: \mathbb{R} \rightarrow \mathbb{R}^q$ is a function that maps **one dimensional** features to a higher dimensional space ($q$). We know:
$$\mathrm{Cov}(f(x_i), f(x_j)) = \sigma^2 \phi(x_i)\phi(x_j)$$

Now, let's **suppose** the following form for $\phi$:
$$\phi(x)_k = \exp\left(-\frac{x^2}{2l^2} \right) \frac{x^k}{l^k\sqrt{k!}}, \space k=1, ..., q$$
where $\phi(x)_k$ denotes the $k$ th element of $\phi(x)$. It's useful to notice that this is essentially a polynomial feature expansion,
$$\phi(x) =  \begin{bmatrix} x^0 \ x^1 \ x^2 ...\ x^q \end{bmatrix}^T$$
but scaled by a constant term.

Now, plugging into our covariance above, we have:
$$ \mathrm{Cov}(f(x_i), f(x_j)) = \sigma^2\phi(x_i)^T\phi(x_j)$$
$$ = \sigma^2 \exp\left(-\frac{x_i^2}{2l^2} \right) \exp\left(-\frac{x_j^2}{2l^2} \right) \sum_{k=0}^{q} \frac{x_i^k}{l^k\sqrt{k!}} \cdot \frac{x_j^k}{l^k\sqrt{k!}}$$
$$ = \sigma^2 \exp\left(-\frac{x_i^2+x_j^2}{2l^2} \right)\sum_{k=0}^{q} \left(\frac{x_i x_j}{l^2}\right)^k \cdot \frac{1}{\sqrt{k!}}$$

Now, if we let $q \to \infty$, then we can recognize the sum as the taylor expansion of $\exp\left(\frac{x_ix_j}{l^2}\right)$. Then after some simple algebra, we end up with:
$$\mathrm{Cov}(f(x_i), f(x_j)) = \sigma ^2 \exp\left(-\frac{(x_i-x_j)^2}{2l^2}\right)$$

Thus, we can see that the kernel arises from a feature expansion of infinite polynomials.