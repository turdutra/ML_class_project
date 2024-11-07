import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

# --------------------- Define Functions --------------------- #
def compute_rst(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")

    # Define Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),        # σ_x
        np.array([[0, -1j], [1j, 0]], dtype=complex),     # σ_y
        np.array([[1, 0], [0, -1]], dtype=complex)        # σ_z
    ]

    # Compute r vector
    r = np.array([0.5 * np.trace(matrix @ np.kron(sigma_j, I)).real for sigma_j in sigma])

    # Compute s vector
    s = np.array([0.5 * np.trace(matrix @ np.kron(I, sigma_k)).real for sigma_k in sigma])

    # Compute T matrix
    T = np.array([[0.25 * np.trace(matrix @ np.kron(sigma[j], sigma[k])).real for k in range(3)] for j in range(3)])

    return r, s, T

def compute_norms(matrix):
    r, s, T = compute_rst(matrix)
    norm_r = np.linalg.norm(r)
    norm_s = np.linalg.norm(s)
    norm_T = np.linalg.norm(T, ord='fro')  # Frobenius norm for matrices
    return pd.Series({'norm_r': norm_r, 'norm_s': norm_s, 'norm_T': norm_T})

def compute_statistics(array):
    mean = np.mean(array)
    median = np.median(array)
    stddev = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    skewness = stats.skew(array)
    kurtosis_val = stats.kurtosis(array)
    return {'Mean': mean, 'Median': median, 'StdDev': stddev, 'Min': min_val, 'Max': max_val, 'Skewness': skewness, 'Kurtosis': kurtosis_val}

def partial_transpose(rho, subsystem=2):
    """Partial transpose with respect to subsystem 1 or 2"""
    return rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)

def negativity(rho):
    """Calculate the negativity of a 4x4 density matrix."""
    # Compute the partial transpose
    rho_pt = partial_transpose(rho)

    # Compute the eigenvalues of the partially transposed matrix
    eigenvalues = np.linalg.eigvals(rho_pt)

    # Negativity is the sum of the absolute values of negative eigenvalues
    return sum(abs(val) for val in eigenvalues if val < 0)

def concurrence(rho):
    """Calculate the concurrence of a 4x4 density matrix."""
    # Define the Pauli Y matrix
    Y = np.array([[0, -1j], [1j, 0]])

    # Compute the spin-flipped density matrix
    rho_tilde = (np.kron(Y, Y) @ rho.conj() @ np.kron(Y, Y))

    # Compute the eigenvalues of R
    R = sqrtm(sqrtm(rho) @ rho_tilde @ sqrtm(rho))
    eigenvalues = np.real(np.linalg.eigvals(R))

    # Sort the eigenvalues in decreasing order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Concurrence is given by max(0, λ1 - λ2 - λ3 - λ4)
    return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

# --------------------- Main Script --------------------- #
def main():
    # Load the DataFrame from the pickle file with error handling
    try:
        df = pd.read_pickle('steering_ml_ds_valid.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'steering_ml_ds_valid.pkl' was not found. Please ensure it exists in the working directory.")
    except pd.errors.PickleError:
        raise ValueError("The file 'steering_ml_ds_valid.pkl' is not a valid pickle file or is corrupted.")

    # Compute norms of r, s, T and add to DataFrame
    df[['norm_r', 'norm_s', 'norm_T']] = df['State'].apply(compute_norms)

    # Compute statistics for norms
    norm_r_array = df['norm_r'].values
    norm_s_array = df['norm_s'].values
    norm_T_array = df['norm_T'].values

    stats_norm_r = compute_statistics(norm_r_array)
    stats_norm_s = compute_statistics(norm_s_array)
    stats_norm_T = compute_statistics(norm_T_array)

    # Prepare statistics DataFrame for norms
    stats_df = pd.DataFrame({
        'Norm': ['norm_r', 'norm_s', 'norm_T'],
        'Mean': [stats_norm_r['Mean'], stats_norm_s['Mean'], stats_norm_T['Mean']],
        'Median': [stats_norm_r['Median'], stats_norm_s['Median'], stats_norm_T['Median']],
        'StdDev': [stats_norm_r['StdDev'], stats_norm_s['StdDev'], stats_norm_T['StdDev']],
        'Min': [stats_norm_r['Min'], stats_norm_s['Min'], stats_norm_T['Min']],
        'Max': [stats_norm_r['Max'], stats_norm_s['Max'], stats_norm_T['Max']],
        'Skewness': [stats_norm_r['Skewness'], stats_norm_s['Skewness'], stats_norm_T['Skewness']],
        'Kurtosis': [stats_norm_r['Kurtosis'], stats_norm_s['Kurtosis'], stats_norm_T['Kurtosis']]
    })

    # Save the statistics to a CSV file
    stats_df.to_csv('rst_norm_statistics.csv', index=False)

    # Compute negativity and concurrence for non-separable states
    df_entangled = df[df['Separable'] == False].copy()

    # Assuming 'State' column contains 4x4 density matrices
    df_entangled['Concurrence'] = df_entangled['State'].apply(concurrence)



    # Plot histogram for Concurrence with range 0 to 1
    plt.figure()
    plt.hist(df_entangled['Concurrence'], bins=30, range=(0, 1), edgecolor='black')
    plt.title('Histogram of Concurrence for Entangled States')
    plt.xlabel('Concurrence')
    plt.ylabel('Frequency')
    plt.savefig('concurrence_histogram.png')

    plt.close('all')

    print("Statistics saved to 'rst_norm_statistics.csv'.")
    print("Histograms saved to 'concurrence_histogram.png'.")


# Entry point
if __name__ == "__main__":
    main()
