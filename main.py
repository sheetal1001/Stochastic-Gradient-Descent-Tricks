"""
Main execution script for the SGD ablation study.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_sparse_data
from src.models import VanillaSGD, FastASGD
from src.utils import finite_difference_check, find_best_gamma0, compute_loss

def main():
    # 1. Generate Dataset
    X_train, y_train = generate_sparse_data(n_samples=50000, n_features=10000)
    X_val, y_val = generate_sparse_data(n_samples=5000, n_features=10000)
    d = X_train.shape[1]

    # 2. Paper Recommendations (Checks on Small Data)
    X_small, y_small = X_train[:1000], y_train[:1000]
    
    finite_difference_check(VanillaSGD(d), X_small, y_small)
    optimal_gamma = find_best_gamma0(FastASGD, d, X_small, y_small)

    # 3. Main Training Performance comparisons
    print("\n[Training] Starting Performance Comparisons")
    epochs = 3
    history_vanilla = []
    history_asgd = []

    # Train Vanilla SGD 
    model_vanilla = VanillaSGD(d, lambd=1e-4, gamma0=optimal_gamma)
    start_time = time.time()
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(X_train.shape[0])
        for step, i in enumerate(shuffled_indices):
            row = X_train.getrow(i)
            model_vanilla.update(row.indices, row.data, y_train[i])
            if step % 5000 == 0:
                history_vanilla.append(compute_loss(model_vanilla, X_val, y_val))
    vanilla_time = time.time() - start_time
    print(f"Vanilla SGD Time: {vanilla_time:.2f} seconds")

    # Train Fast ASGD 
    model_asgd = FastASGD(d, lambd=1e-4, gamma0=optimal_gamma, t0=5000)
    start_time = time.time()
    for epoch in range(epochs):
        # Recommended in paper, shuffling the data
        shuffled_indices = np.random.permutation(X_train.shape[0])
        for step, i in enumerate(shuffled_indices):
            row = X_train.getrow(i)
            model_asgd.update(row.indices, row.data, y_train[i])
            if step % 5000 == 0:
                history_asgd.append(compute_loss(model_asgd, X_val, y_val))
    asgd_time = time.time() - start_time
    print(f"Fast ASGD Time  : {asgd_time:.2f} seconds")

    # 4. Results & Plotting
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_vanilla, label='Vanilla SGD (No Tricks)', color='red', alpha=0.7)
    plt.plot(history_asgd, label='Fast ASGD (Bottou Tricks)', color='blue', linewidth=2)
    plt.xlabel("Training Steps (x5000)")
    plt.ylabel("Validation Log-Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()