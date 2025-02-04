import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

####################################
# 1. Generate Synthetic Data
####################################

n_samples = 1000  # number of data samples
dim = 5          # data dimensionality

# Generate random data and center it
data = torch.randn(n_samples, dim)
data_centered = data - data.mean(dim=0, keepdim=True)

# Compute the sample covariance matrix C (20x20)
C = (data_centered.T @ data_centered) / (n_samples - 1)
print("Measured Covariance Matrix C shape:", C.shape)

####################################
# 2. Define the Low-Rank Model: C_hat = U S U^T
####################################

# We want to approximate C with a rank-2 model.
n_components = 2

# Initialize U as a (dim x n_components) matrix.
# It should eventually have orthonormal columns.
U_param = nn.Parameter(torch.randn(dim, n_components))

# Initialize S as a diagonal matrix. We parameterize it as a vector.
# To ensure positive eigenvalues (since C is PSD) we will use the exponential.
S_param = nn.Parameter(torch.randn(n_components))

# Define an optimizer for both U_param and S_param
optimizer = optim.Adam([U_param, S_param], lr=0.01)

# Hyperparameter for the orthonormality penalty on U.
lambda_ortho = 10.0

# List to store loss values at each epoch
loss_history = []

####################################
# 3. Fit the Model via Backpropagation
####################################

n_epochs = 1000

# Record the start time
start_time = time.time()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Build the diagonal matrix S ensuring positivity.
    # (You could also use torch.relu, but exp ensures strictly positive values.)
    S_diag = torch.diag(torch.exp(S_param))
    
    # Compute the low-rank approximation C_hat = U S U^T.
    C_hat = U_param @ S_diag @ U_param.T
    
    # Reconstruction loss (Frobenius norm between C and C_hat)
    loss_fit = torch.norm(C - C_hat, p='fro')**2
    
    # Orthonormality penalty for U: we want U^T U = I.
    I = torch.eye(n_components, device=U_param.device)
    ortho_penalty = torch.norm(U_param.T @ U_param - I, p='fro')**2
    
    # Total loss: trade-off between fitting C and enforcing orthonormality.
    loss = loss_fit + lambda_ortho * ortho_penalty

    # Record the loss value for this epoch
    loss_history.append(loss.item())
    
    # Backpropagation and parameter update.
    loss.backward()
    optimizer.step()
    
    # Optionally, print progress every 100 epochs.
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.4e}, Fit Loss = {loss_fit.item():.4e}, Ortho Penalty = {ortho_penalty.item():.4e}")

# Record the end time and compute the elapsed time.
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nFitting took: {elapsed_time:.2f} seconds.")

####################################
# 3b. plot the loss history
####################################
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

####################################
# 4. Results from the Learned Model
####################################

# Final learned U and S (as a diagonal matrix)
U_learned = U_param.detach()
S_learned = torch.diag(torch.exp(S_param.detach()))

# The low-rank covariance approximation:
C_hat_final = U_learned @ S_learned @ U_learned.T

print("\nLearned U (from gradient descent):")
print(U_learned)
print("\nLearned diagonal S (eigenvalues, via exp(S_param)):")
print(S_learned)

####################################
# 5. Classical PCA (Eigenvalue Decomposition)
####################################

# Since C is symmetric and real, we can use torch.linalg.eigh.
# Note: torch.linalg.eigh returns eigenvalues in ascending order.
eigvals, eigvecs = torch.linalg.eigh(C)

# Reverse the order so that the largest eigenvalues come first.
eigvals = eigvals.flip(dims=[0])
eigvecs = eigvecs.flip(dims=[1])

# Select the top n_components eigenvectors and eigenvalues.
U_pca = eigvecs[:, :n_components]
S_pca = torch.diag(eigvals[:n_components])

print("\nClassical PCA results:")
print("Top eigenvectors (U_pca):")
print(U_pca)
print("\nTop eigenvalues (in diagonal matrix S_pca):")
print(S_pca)

####################################
# 6. Compare Subspaces
####################################

# Because the eigenvectors are defined only up to sign (and rotation for subspaces),
# we can compare the subspaces spanned by U_learned and U_pca.
# One simple approach is to compute the absolute values of the inner product matrix.

inner_product = torch.abs(U_learned.T @ U_pca)
print("\nAbsolute inner product between learned U and PCA U:")
print(inner_product)

# The closer the inner products are to 1, the more aligned the corresponding directions.

# Check orthonormality of the learned U
print("\nU_learned^T U_learned (should be close to I):")
print(U_learned.T @ U_learned)

