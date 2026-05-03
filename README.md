# Stochastic Gradient Descent Tricks (Bottou, 2012) 🚀

This repository contains a Python implementation (from-scratch) of the highly optimized Machine Learning algorithms detailed in Léon Bottou’s foundational 2012 paper, **"Stochastic Gradient Descent Tricks"** (included in this repository as `tricks-2012.pdf`). 

The objective of this project is to demonstrate how applying specific mathematical and engineering "tricks" can transform standard Stochastic Gradient Descent (SGD) from a noisy, slow algorithm into an incredibly fast, highly scalable optimizer for massive, sparse datasets.

## 1. The Core Philosophy: Why SGD?
In large-scale machine learning, we are typically constrained by computing time, not data availability. 

The total excess error of a machine learning model is broken down into:
Total Error = Approximation Error + Estimation Error + Optimization Error

While standard Batch Gradient Descent rigorously minimizes optimization error by calculating exact gradients, it is incredibly slow. Bottou argues that for large-scale problems, it is mathematically superior to use a "noisy" optimizer like SGD. Because SGD updates weights after a single example, it processes vast amounts of data in a fraction of the time, heavily driving down the estimation error and resulting in a better overall predictive model.

## 2. The Implemented "Tricks"

This repository features an Study comparing a Vanilla SGD implementation against a Fast Averaged SGD (ASGD) model that utilizes the following optimizations from the paper:

### 1. The $O(1)$ Sparsity Update (The "Speed" Trick)
In standard SGD with $L_2$ regularization, the weight decay penalty requires us to update the entire weight vector $w$ at every step:
$$w_{t+1} = w_t (1 - \gamma_t \lambda) - \gamma_t \nabla Q(z_t, w_t)$$
If our data has 10,000 features, but an individual training example only has 5 non-zero values, updating all 10,000 weights is a massive computational waste. 

**The Solution:** We decompose the weight vector into a scalar $s$ and a vector $W$ such that $w = sW$. 
* We apply the $L_2$ penalty only to the scalar: $s_{t+1} = s_t (1 - \gamma_t \lambda)$
* We apply the gradient update only to the active features: $W_{t+1} = W_t - \frac{\gamma_t}{s_{t+1}} \nabla Q$

This mathematical trick turns an $O(d)$ operation into an $O(\text{non-zeros})$ operation, speeding up training on sparse data exponentially.

### 2. Polyak-Ruppert Averaging / ASGD (The "Smoothness" Trick)
Because SGD uses a constant stream of noisy, single-example gradients, the model never truly settles at the absolute minimum.
Instead of aggressively shrinking the learning rate to force it to stop, **Averaged SGD (ASGD)** maintains a running average of the weights over time:
$$\bar{w}_t = \frac{1}{t - t_0 + 1} \sum_{i=t_0}^{t} w_i$$
This averaging perfectly cancels out the variance of the noisy stochastic steps, resulting in a smooth, highly accurate final prediction function that mimics second-order optimization methods.

### 3. Engineering Best Practices
Following Bottou’s strict recommendations, the pipeline also implements:
* **Pre-training Gradient Checks:** Uses finite difference approximations ($\frac{f(w + \delta) - f(w)}{\delta}$) on a small subset to prove the analytical gradient math is flawless.
* **Hyperparameter Search on Small Data:** Determines the optimal initial learning rate ($\gamma_0$) using a 1,000-sample subset before scaling up.
* **Random Shuffling:** Destroys dataset order bias by generating random permutations of indices before every epoch.


## 3. How to Run the Code

This project is built to be 100% reproducible with zero external data dependencies. It generates a synthetic sparse dataset perfectly tailored to benchmark the sparsity tricks.

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the Code:**
```bash
python main.py
```

## 4. Expected Results

When running the full script, the console will output the pre-training sanity checks, followed by the training times. You will observe that:
1. **Time:** The Fast ASGD (using the $w = sW$ trick) trains faster than Vanilla SGD.
2. **Convergence:** The resulting Matplotlib graph will show Vanilla SGD "bouncing" (high variance in validation loss), while the Fast ASGD curve will be incredibly smooth and converge to a lower overall log-loss.

Note:
In this Python implementation, you might notice that Vanilla SGD remains highly competitive in time with FastSGD; 
1. Vanilla SGD relies on "self.w *= scalar", which Python outsources to NumPy's highly-optimized (C-backend). 
2. The Fast ASGD requires Python-level control flow (if statements, scalar maintenance, and sparse index lookups) which introduces interpreter overhead.
While the mathematical algorithmic complexity is vastly reduced in the ASGD implementation, Python's interpreter overhead creates a bottleneck. 