'''


from code_ecg import *
def main():
    # Records used in the paper
    record_names = ['100', '102', '103', '105', '111', '116', '122', '205']
    noise_types = ['ma', 'em', 'wn']  # muscle artifact, electrode motion, white noise
    
    # We'll focus on SNR=5dB as mentioned in the paper
    snr_db = 5
    
    # Results storage
    results = {}
    
    # Process each record
    for record_name in record_names:
        record_path = download_physionet_record('mitdb', record_name)
        signal_data, fs = load_ecg_signal(record_path)
        
        # Preprocess to remove PLI and BW
        clean_signal = preprocess_signal(signal_data, fs)
        
        results[record_name] = {}
        
        # Process each noise type
        for noise_type in noise_types:
            # Load and prepare noise
            noise_data = load_noise_signal(noise_type)
            # Ensure noise has same length as signal
            noise_data = noise_data[:len(clean_signal)]
            
            # Add noise to the clean signal
            noisy_signal = add_noise(clean_signal, noise_data, snr_db)
            
            # Segment into heartbeats
            clean_beats = segment_heartbeats(clean_signal, fs)
            noisy_beats = segment_heartbeats(noisy_signal, fs)
            
            # Apply PCA denoising
            pca_denoised_beats = denoise_pca(noisy_beats)
            pca_denoised_signal = reconstruct_signal(pca_denoised_beats)
            
            # Apply KPCA denoising
            kpca_denoised_beats = denoise_kpca(noisy_beats)
            kpca_denoised_signal = reconstruct_signal(kpca_denoised_beats)
            
            # Calculate RMSE
            pca_rmse = calculate_rmse(clean_signal[:len(pca_denoised_signal)], 
                                      pca_denoised_signal)
            kpca_rmse = calculate_rmse(clean_signal[:len(kpca_denoised_signal)], 
                                       kpca_denoised_signal)
            
            # Store results
            results[record_name][noise_type] = {
                'PCA_RMSE': pca_rmse,
                'KPCA_RMSE': kpca_rmse
            }
            
            print(f"Record: {record_name}, Noise: {noise_type}")
            print(f"PCA RMSE: {pca_rmse:.2f}")
            print(f"KPCA RMSE: {kpca_rmse:.2f}")
            print("--------")
            
            # For record 103 with ma noise, generate plots as in the paper
            if record_name == '103' and noise_type == 'ma':
                plot_results(clean_signal, noisy_signal, pca_denoised_signal, 
                             kpca_denoised_signal, f"Record {record_name} with {noise_type} noise")
                
                
            
            # For record 100 with ma noise, show KPCA with different kernels as in Fig. 4
            if record_name == '100' and noise_type == 'ma':
                # RBF kernel
                kpca_rbf_beats = denoise_kpca(noisy_beats, kernel='rbf')
                kpca_rbf_signal = reconstruct_signal(kpca_rbf_beats)
                
                
                
                plt.figure(figsize=(15, 9))
                
                plt.subplot(3, 1, 1)
                plt.plot(noisy_signal)
                plt.title(f'Noisy ECG Signal (Record {record_name} with {noise_type} noise)')
                plt.grid(True)
                
                plt.subplot(3, 1, 2)
                plt.plot(kpca_rbf_signal)
                plt.title('KPCA Denoised Signal (RBF Kernel)')
                plt.grid(True)
                
                
                plt.tight_layout()
                plt.show()
    
    # Print summary table
    print("\nSummary Table - RMSE Values for SNR=5dB")
    print("Record\tNoise\tPCA RMSE\tKPCA RMSE")
    print("-" * 50)
    for record in results:
        for noise in results[record]:
            pca_rmse = results[record][noise]['PCA_RMSE']
            kpca_rmse = results[record][noise]['KPCA_RMSE']
            better = "KPCA" if kpca_rmse < pca_rmse else "PCA"
            print(f"{record}\t{noise}\t{pca_rmse:.2f}\t{kpca_rmse:.2f}\t({better} is better)")

if __name__ == "__main__":
    main()
    
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from code_ecg import *
from grid_search_local import best_params

selected_records = ['100', '105', '116']
noise_types = ['ma', 'em', 'wn']
snr_db = 5

# 1. Visualisation signal 100 : original, bruité, débruité (séparément pour chaque bruit)
record = '100'
record_path = download_physionet_record('mitdb', record)
signal_data, fs = load_ecg_signal(record_path)
clean_signal = preprocess_signal(signal_data, fs)

# On limite à une portion pour avoir une meilleure vue (ex: 2000 premiers points)
segment_length = 2000
clean_signal = clean_signal[:segment_length]

# Figure : signal propre seul
plt.figure(figsize=(12, 4))
plt.plot(clean_signal, label='Clean ECG Signal', color='red')
plt.title("Record 100 - Clean ECG Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("record100_clean.png", dpi=300)
plt.show()



# Boucle sur chaque type de bruit
for noise_type in noise_types:
    # Génération du signal bruité
    noise_data = load_noise_signal(noise_type)[:segment_length]
    noisy_signal = add_noise(clean_signal, noise_data, snr_db)

    # Denoising avec KPCA
    params = best_params['100'][noise_type]
    noisy_beats = segment_heartbeats(noisy_signal, fs)
    kpca_denoised_beats = denoise_kpca(noisy_beats, **params)
    kpca_denoised_signal = reconstruct_signal(kpca_denoised_beats)

    # Troncature si dépasse la longueur originale
    kpca_denoised_signal = kpca_denoised_signal[:len(clean_signal)]
    
    #Figure: bruité vs propre
    # Figure : bruité vs propre
    plt.figure(figsize=(12, 4))
    plt.plot(clean_signal, label='Clean Signal', color='green', alpha=0.8)
    plt.plot(noisy_signal, label=f"Noisy ({noise_type.upper()})", alpha=0.6)
    plt.title(f"Record 100 - {noise_type.upper()} Noise: Clean vs Noisy")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"record100_{noise_type}_compare.png", dpi=300)
    plt.show()


    # Figure : bruité vs debruité vs propre
    plt.figure(figsize=(12, 4))
    plt.plot(clean_signal, label='Clean Signal', color='green', alpha=0.8)
    plt.plot(noisy_signal, label=f"Noisy ({noise_type.upper()})", alpha=0.6)
    plt.plot(kpca_denoised_signal, label="KPCA Denoised", alpha=0.9)
    plt.title(f"Record 100 - {noise_type.upper()} Noise: Clean vs Noisy vs KPCA Denoised")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"record100_{noise_type}_compare.png", dpi=300)
    plt.show()
    
    


# 2. Comparaison PCA vs KPCA - RMSE
print("\nComparatif PCA vs KPCA (hyperparamètres optimaux) - RMSE")
print("Record\tNoise\tPCA_RMSE\tKPCA_RMSE\tParams")
print("-" * 70)

results_table = []

for record in selected_records:
    record_path = download_physionet_record('mitdb', record)
    signal_data, fs = load_ecg_signal(record_path)
    clean_signal = preprocess_signal(signal_data, fs)

    for noise in noise_types:
        noise_data = load_noise_signal(noise)[:len(clean_signal)]
        noisy_signal = add_noise(clean_signal, noise_data, snr_db)

        clean_beats = segment_heartbeats(clean_signal, fs)
        noisy_beats = segment_heartbeats(noisy_signal, fs)

        pca_denoised = reconstruct_signal(denoise_pca(noisy_beats))
        kpca_denoised = reconstruct_signal(denoise_kpca(noisy_beats, **best_params[record][noise]))

        pca_rmse = calculate_rmse(clean_signal[:len(pca_denoised)], pca_denoised)
        kpca_rmse = calculate_rmse(clean_signal[:len(kpca_denoised)], kpca_denoised)

        print(f"{record}\t{noise}\t{pca_rmse:.2f}\t\t{kpca_rmse:.2f}\t\t{best_params[record][noise]}")

        results_table.append({
            'record': record,
            'noise': noise,
            'pca_rmse': pca_rmse,
            'kpca_rmse': kpca_rmse,
            'kernel': best_params[record][noise].get('kernel'),
            'gamma': best_params[record][noise].get('gamma', None)
        })

# 3. Visualisation choix des hyperparamètres (gamma vs RMSE pour RBF)
df_results = pd.DataFrame(results_table)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_results[df_results['kernel'] == 'rbf'],
    x='gamma', y='kpca_rmse',
    hue='noise', style='record', s=100
)
plt.xscale('log')
plt.title("KPCA - RMSE selon Gamma (kernel RBF uniquement)")
plt.xlabel("Gamma (log scale)")
plt.ylabel("KPCA RMSE")
plt.grid(True)
plt.tight_layout()
plt.show()
