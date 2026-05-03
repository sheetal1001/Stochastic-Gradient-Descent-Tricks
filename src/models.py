"""
Model architectures demonstrating the algorithms from Bottou's 2012 paper.
"""
import numpy as np
from src.utils import log_loss_derivative

class VanillaSGD:
    # Standard SGD (No Sparsity Trick, No Averaging).
    # Serves as our baseline. It updates the ENTIRE weight vector on every 
    # step to apply L2 regularization, which is extremely slow on sparse data.
    def __init__(self, d, lambd=1e-4, gamma0=0.1):
        self.w = np.zeros(d)
        self.lambd = lambd    # Regularization penalty
        self.gamma0 = gamma0  # Initial learning rate
        self.t = 0            # Step counter

    def update(self, x_indices, x_data, y):
        self.t += 1
        # Bottou's learning rate schedule: shrinks over time to force convergence
        gamma_t = self.gamma0 / (1.0 + self.gamma0 * self.lambd * self.t)
        
        # Calculate gradient
        margin = y * np.sum(self.w[x_indices] * x_data)
        g_t = log_loss_derivative(margin)
        # PENALIZE ENTIRE VECTOR
        self.w *= (1.0 - gamma_t * self.lambd)
        # Add gradient to active features
        self.w[x_indices] -= (gamma_t * y * g_t) * x_data
        
    def get_weights(self):
        return self.w


class FastASGD:
    # Averaged SGD (ASGD) utilizing the Sparsity Trick.
    """
    Sparsity Trick (w = s*W): Applies L2 regularization in O(1) time by 
        shrinking a single scalar `s` instead of the whole vector.
    ASGD: Averages the weights over time to achieve smooth, second-order-like 
        convergence, stopping the "bouncing" effect at the minimum.
    """
    def __init__(self, d, lambd=1e-4, gamma0=0.1, t0=5000):
        # Trick 1: Sparsity Scaler (w = s*W)
        self.W = np.zeros(d)
        self.s = 1.0
        
        # Trick 2: ASGD Running Average
        self.A = np.zeros(d)
        self.alpha = 0.0
        self.beta = 1.0
        
        self.lambd = lambd  # Regularization penalty
        self.gamma0 = gamma0  # Initial learning rate
        self.t0 = t0
        self.t = 0  # Step counter

    def update(self, x_indices, x_data, y):
        self.t += 1
        # ASGD requires a slightly different decay schedule: t^(-3/4)
        gamma_t = self.gamma0 * (1.0 + self.gamma0 * self.lambd * self.t) ** -0.75
        mu_t = 1.0 / max(1, self.t - self.t0)
        
        margin = y * self.s * np.sum(self.W[x_indices] * x_data)
        g_t = log_loss_derivative(margin)
        
        # Sparsity Trick 
        self.s *= (1.0 - gamma_t * self.lambd)
        if self.s < 1e-10: 
            self.W *= self.s
            self.A /= self.s 
            self.alpha *= self.s
            self.s = 1.0
            
        update_val = (gamma_t * y * g_t / self.s) * x_data
        self.W[x_indices] -= update_val
        
        # ASGD Trick (Polyak-Ruppert Averaging) 
        if mu_t == 1.0:
            self.A = np.zeros_like(self.W)
            self.beta = 1.0
            self.alpha = self.s
        else:
            self.A[x_indices] += (self.alpha * update_val)
            self.beta /= (1.0 - mu_t)
            self.alpha += mu_t * self.beta * self.s

    def get_weights(self):
        return (self.A + self.alpha * self.W) / self.beta