from code_ecg import *
import numpy as np

def grid_search_kpca_on_subset(record_list, noise_list, snr_db=5,
                                kernels=['linear', 'rbf', 'poly'],
                                gammas=[0.01, 0.1, 0.5, 1],
                                degrees=[2, 3, 4]):

    best_results = {}

    for record_name in record_list:
        record_path = download_physionet_record('mitdb', record_name)
        signal_data, fs = load_ecg_signal(record_path)
        clean_signal = preprocess_signal(signal_data, fs)
        clean_beats = segment_heartbeats(clean_signal, fs)

        best_results[record_name] = {}

        for noise_type in noise_list:
            noise_data = load_noise_signal(noise_type)[:len(clean_signal)]
            noisy_signal = add_noise(clean_signal, noise_data, snr_db)
            noisy_beats = segment_heartbeats(noisy_signal, fs)

            best_rmse = float('inf')
            best_params = {}

            for kernel in kernels:
                if kernel == 'linear':
                    denoised_beats = denoise_kpca(noisy_beats, kernel=kernel)
                    recon = reconstruct_signal(denoised_beats)
                    ref = reconstruct_signal(clean_beats[:len(denoised_beats)])
                    rmse = calculate_rmse(ref[:len(recon)], recon)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {'kernel': kernel}

                elif kernel == 'rbf':
                    for gamma in gammas:
                        denoised_beats = denoise_kpca(noisy_beats, kernel=kernel, gamma=gamma)
                        recon = reconstruct_signal(denoised_beats)
                        ref = reconstruct_signal(clean_beats[:len(denoised_beats)])
                        rmse = calculate_rmse(ref[:len(recon)], recon)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'kernel': kernel, 'gamma': gamma}

                elif kernel == 'poly':
                    for degree in degrees:
                        denoised_beats = denoise_kpca(noisy_beats, kernel=kernel, degree=degree)
                        recon = reconstruct_signal(denoised_beats)
                        ref = reconstruct_signal(clean_beats[:len(denoised_beats)])
                        rmse = calculate_rmse(ref[:len(recon)], recon)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'kernel': kernel, 'degree': degree}

            best_results[record_name][noise_type] = {
                'best_rmse': best_rmse,
                'best_params': best_params
            }

            print(f"Record: {record_name}, Noise: {noise_type} => Best RMSE: {best_rmse:.2f} with {best_params}")

    return best_results


# Exécution principale
if __name__ == "__main__":
    selected_records = ['100', '105', '116']
    noise_types = ['ma', 'em', 'wn']
    
    results = grid_search_kpca_on_subset(selected_records, noise_types)

    print("\nSummary of Best Hyperparameters per Record and Noise Type:")
    for rec in results:
        for noise in results[rec]:
            res = results[rec][noise]
            print(f"{rec} - {noise} : RMSE={res['best_rmse']:.2f}, Params={res['best_params']}")

    # Stocke les meilleurs paramètres sous forme exploitable
    best_params = {
        rec: {
            noise: results[rec][noise]['best_params']
            for noise in results[rec]
        }
        for rec in results
    }

else:
    # Si importé dans un autre fichier (comme main_ecg.py)
    selected_records = ['100', '105', '116']
    noise_types = ['ma', 'em', 'wn']
    _results = grid_search_kpca_on_subset(selected_records, noise_types)

    best_params = {
        rec: {
            noise: _results[rec][noise]['best_params']
            for noise in _results[rec]
        }
        for rec in _results
    }
