"""
Mathematical utilities, loss functions, and Gradient checks.
"""
import numpy as np

def log_loss(m):
    #Computes log-loss: l(m) = log(1 + exp(-m))
    return np.log(1.0 + np.exp(-m))

def log_loss_derivative(m):
    #Computes derivative of log-loss: l'(m) = -1 / (1 + exp(m))
    return -1.0 / (1.0 + np.exp(m))

def compute_loss(model, X, y):
    #Calculates the average log-loss across a dataset.
    w = model.get_weights()
    margins = y * X.dot(w)
    return np.mean(log_loss(margins))

def finite_difference_check(model, X_sample, y_sample, epsilon=1e-5):
    """
    Bottou's Recommendation: Verify analytical gradients via finite differences 
    before running large-scale experiments.
    """
    print("\nRunning Finite Difference Gradient Check")
    row = X_sample.getrow(0)
    x_indices, x_data = row.indices, row.data
    y = y_sample[0]
    
    w_orig = model.get_weights().copy()
    
    # 1. Base loss
    margin = y * np.sum(w_orig[x_indices] * x_data)
    loss_before = log_loss(margin)
    
    # 2. Analytical gradient
    grad = y * log_loss_derivative(margin) * x_data
    
    # 3. Perturb weights
    delta = np.random.randn(len(x_indices)) * epsilon
    w_perturbed = w_orig.copy()
    w_perturbed[x_indices] += delta
    
    # 4. New loss
    margin_new = y * np.sum(w_perturbed[x_indices] * x_data)
    loss_after = log_loss(margin_new)
    
    # 5. Compare
    predicted_loss = loss_before + np.sum(delta * grad)
    if np.isclose(loss_after, predicted_loss, atol=epsilon*10):
        print("Gradient Check PASSED!")
    else:
        print("Gradient Check FAILED!")

def find_best_gamma0(model_class, d, X_small, y_small, candidates=[0.001, 0.01, 0.1, 1.0]):
    """
    Bottou's Recommendation: Use a small sample to find the optimal initial 
    learning rate (gamma0) before scaling up.
    """
    print("\n[Hyperparameter Search] Finding optimal gamma0 on small dataset")
    best_gamma = candidates[0]
    best_loss = float('inf')
    
    for gamma in candidates:
        model = model_class(d, lambd=1e-4, gamma0=gamma)
        for i in range(X_small.shape[0]):
            row = X_small.getrow(i)
            model.update(row.indices, row.data, y_small[i])
        
        loss = compute_loss(model, X_small, y_small)
        print(f"  Tested gamma0={gamma:<5} | Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_gamma = gamma
            
    print(f"Selected optimal gamma0: {best_gamma}")
    return best_gamma