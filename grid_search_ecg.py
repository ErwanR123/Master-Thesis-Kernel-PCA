from code_ecg import *
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import KernelPCA
import time
import pandas as pd

def evaluate_kpca_params(beats_clean, beats_noisy, params, cv=5):
    """
    Perform grid search to find optimal parameters for Kernel PCA denoising.
    
    Parameters:
    -----------
    beats_clean : array-like
        Original clean ECG beats
    beats_noisy : array-like
        Noisy ECG beats to denoise
    params : dict
        Dictionary of parameters to test
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with results for all parameter combinations
    """
    results = []
    
    # Create parameter combinations
    param_combinations = []
    for kernel in params['kernel']:
        for n_components in params['n_components']:
            if kernel == 'rbf':
                for gamma in params['gamma']:
                    param_combinations.append({
                        'kernel': kernel,
                        'n_components': n_components,
                        'gamma': gamma
                    })
            elif kernel == 'poly':
                for degree in params['degree']:
                    for gamma in params['gamma']:
                        param_combinations.append({
                            'kernel': kernel,
                            'n_components': n_components,
                            'gamma': gamma,
                            'degree': degree
                        })
    
    # Prepare cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations with {cv}-fold cross validation")
    
    for i, params_dict in enumerate(param_combinations):
        start_time = time.time()
        print(f"Combination {i+1}/{total_combinations}: {params_dict}")
        
        rmse_values = []
        
        for train_idx, test_idx in kf.split(beats_noisy):
            # Get train and test sets
            X_train, X_test = beats_noisy[train_idx], beats_noisy[test_idx]
            y_train, y_test = beats_clean[train_idx], beats_clean[test_idx]
            
            # Apply KPCA for denoising
            if params_dict['kernel'] == 'rbf':
                kpca = KernelPCA(
                    n_components=params_dict['n_components'],
                    kernel=params_dict['kernel'],
                    gamma=params_dict['gamma'],
                    fit_inverse_transform=True,
                    alpha=1e-5
                )
            else:  # poly kernel
                kpca = KernelPCA(
                    n_components=params_dict['n_components'],
                    kernel=params_dict['kernel'],
                    gamma=params_dict['gamma'],
                    degree=params_dict['degree'],
                    fit_inverse_transform=True,
                    alpha=1e-5
                )
            
            # Fit on training data
            mean_beat = np.mean(X_train, axis=0)
            centered_beats = X_train - mean_beat
            kpca.fit(centered_beats)
            
            # Transform test data
            centered_test = X_test - mean_beat
            transformed = kpca.transform(centered_test)
            reconstructed = kpca.inverse_transform(transformed) + mean_beat
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, reconstructed))
            rmse_values.append(rmse)
        
        # Average RMSE across folds
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        
        # Create result entry
        result = {
            'kernel': params_dict['kernel'],
            'n_components': params_dict['n_components'],
            'gamma': params_dict['gamma'],
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'time': time.time() - start_time
        }
        
        if params_dict['kernel'] == 'poly':
            result['degree'] = params_dict['degree']
        
        results.append(result)
        print(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)
    return results_df

def run_grid_search():
    """Run grid search for KPCA denoising parameters"""
    # Download and prepare data
    print("Preparing data...")
    record_path = download_physionet_record('mitdb', '100')
    ecg_signal, fs = load_ecg_signal(record_path)
    clean_signal = preprocess_signal(ecg_signal, fs)
    
    # Add noise
    noise_signal = load_noise_signal('ma')  # muscle artifact noise
    snr_db = 5  # 5dB SNR
    noisy_signal = add_noise(clean_signal, noise_signal[:len(clean_signal)], snr_db)
    
    # Segment into beats
    beat_length = 311  # as in paper
    clean_beats = segment_heartbeats(clean_signal, fs, beat_length)
    noisy_beats = segment_heartbeats(noisy_signal, fs, beat_length)
    
    # Define parameters to search
    params = {
        'kernel': ['rbf', 'poly'],
        'n_components': [4, 6, 8, 10, 12],
        'gamma': [0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4]  # Only for poly kernel
    }
    
    # Run grid search
    print("Starting grid search...")
    results = evaluate_kpca_params(clean_beats, noisy_beats, params, cv=5)
    
    # Print best parameters
    best_idx = results['mean_rmse'].idxmin()
    best_params = results.loc[best_idx].to_dict()
    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Save results to CSV
    results.to_csv('kpca_grid_search_results.csv', index=False)
    print("Results saved to kpca_grid_search_results.csv")
    
    # Plot results by kernel
    plt.figure(figsize=(15, 10))
    
    for kernel in results['kernel'].unique():
        kernel_results = results[results['kernel'] == kernel]
        if kernel == 'rbf':
            for gamma in kernel_results['gamma'].unique():
                subset = kernel_results[kernel_results['gamma'] == gamma]
                plt.plot(
                    subset['n_components'], 
                    subset['mean_rmse'], 
                    'o-', 
                    label=f'RBF, gamma={gamma}'
                )
        else:  # poly
            for degree in kernel_results['degree'].dropna().unique():
                for gamma in kernel_results[kernel_results['degree'] == degree]['gamma'].unique():

                    subset = kernel_results[(kernel_results['degree'] == degree) & 
                                           (kernel_results['gamma'] == gamma)]
                    plt.plot(
                        subset['n_components'], 
                        subset['mean_rmse'], 
                        'o--', 
                        label=f'Poly, degree={degree}, gamma={gamma}'
                    )
    
    plt.xlabel('Number of Components')
    plt.ylabel('Mean RMSE')
    plt.title('KPCA Denoising Performance by Kernel and Parameters')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('kpca_grid_search_results.png')
    plt.show()
    
    # Test best parameters on the same data
    print("\nTesting best parameters...")
    if best_params['kernel'] == 'rbf':
        best_denoised = denoise_kpca(
            noisy_beats,
            n_components=int(best_params['n_components']),
            kernel=best_params['kernel'],
            gamma=best_params['gamma']
        )
    else:  # poly
        best_denoised = denoise_kpca(
            noisy_beats,
            n_components=int(best_params['n_components']),
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            degree=int(best_params['degree'])
        )
    
    # Reconstruct full signals
    clean_full = reconstruct_signal(clean_beats)
    noisy_full = reconstruct_signal(noisy_beats)
    denoised_full = reconstruct_signal(best_denoised)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(clean_full[:1000])
    plt.title('Original Clean ECG Signal')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy_full[:1000])
    plt.title(f'Noisy ECG Signal (SNR={snr_db}dB)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised_full[:1000])
    plt.title(f'Best KPCA Denoised Signal (Kernel: {best_params["kernel"]})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('best_kpca_result.png')
    plt.show()
    
    print(f"Best RMSE: {best_params['mean_rmse']:.4f}")
    
    return best_params

if __name__ == "__main__":
    best_params = run_grid_search()



