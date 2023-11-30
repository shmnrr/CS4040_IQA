import pandas as pd
import numpy as np
from scipy import stats
import os

def calculate_plcc(df1, df2, column1, column2):
    return df1[column1].corr(df2[column2], method='pearson')

def calculate_srcc(df1, df2, column1, column2):
    return df1[column1].corr(df2[column2], method='spearman')

def calculate_krcc(df1, df2, column1, column2):
    return df1[column1].corr(df2[column2], method='kendall')

def calculate_results(results_list, mos_results_df, score_column, mos_column):
    # print(results_list)
    # Calculate PLCC, SRCC, and KRCC for each run
    plcc_values = [calculate_plcc(df, mos_results_df, score_column, mos_column) for df in results_list]
    srcc_values = [calculate_srcc(df, mos_results_df, score_column, mos_column) for df in results_list]
    krcc_values = [calculate_krcc(df, mos_results_df, score_column, mos_column) for df in results_list]
    
    print(plcc_values)
    
    plcc = sum(plcc_values) / len(plcc_values)
    srcc = sum(srcc_values) / len(srcc_values)
    krcc = sum(krcc_values) / len(krcc_values)
    
    # Calculate mean for PLCC, SRCC, and KRCC values
    # Redundant but keeping to not change the code
    plcc_mean = np.mean(plcc_values)
    srcc_mean = np.mean(srcc_values)
    krcc_mean = np.mean(krcc_values)

    # Calculate standard deviation for PLCC, SRCC, and KRCC values across all runs
    plcc_std = np.std(plcc_values)
    srcc_std = np.std(srcc_values)
    krcc_std = np.std(krcc_values)
    
    # print(plcc_values)
    
    return plcc, plcc_mean, plcc_std, srcc, srcc_mean, srcc_std, krcc, krcc_mean, krcc_std


def calculate_times(results_list, total_time_list):
    total_time_sum_values = [df['Total time'].sum() for df in results_list]
    total_time_mean_values = [df['Total time'].mean() for df in results_list]
    inference_time_mean_values = [df['Inference time'].mean() for df in results_list]
    preprocessing_time_mean_values = [df['Preprocessing time'].mean() for df in results_list]
    total_time_diff_values = [df['Total time'].sum() - total_time for df, total_time in zip(results_list, total_time_list)]
    
    total_time_sum = sum(total_time_sum_values) / len(total_time_sum_values)
    total_time_mean = sum(total_time_mean_values) / len(total_time_mean_values)
    inference_time_mean = sum(inference_time_mean_values) / len(inference_time_mean_values)
    preprocessing_time_mean = sum(preprocessing_time_mean_values) / len(preprocessing_time_mean_values)
    total_time_diff = abs(sum(total_time_diff_values) / len(total_time_diff_values))
    
    return total_time_sum, total_time_mean, inference_time_mean, preprocessing_time_mean, total_time_diff

def load_results(platform, model):
    results = []
    times = []
    anomalies = pd.DataFrame()
    directory = f'./results/{platform}/{model}'
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract the total run time from the filename
            time = float(filename.split('_')[-1].replace('.csv', ''))
            times.append(time)
            
            # Load the CSV file
            result = pd.read_csv(os.path.join(directory, filename))
            
            # Check if all entries in 'Score' column are numerical
            if not pd.to_numeric(result['Score'], errors='coerce').notnull().all():
                print(f"Non-numerical entries found in 'Score' column in file {filename}.")
                continue
            
            # Calculate Z-scores
            result['Score_zscore'] = np.abs(stats.zscore(result['Score']))
            
            # Define a threshold for anomalies
            threshold = 3
            
            # Find anomalies
            anomalies = pd.concat([anomalies, result[result['Score_zscore'] > threshold]])
            
            results.append(result)
    return results, times, anomalies


def count_anomalies(anomalies):
    """Count the number of anomalies in each result set."""
    anomaly_counts = {key: len(value) for key, value in anomalies.items()}
    return anomaly_counts


def main():
    # Load PC results
    pc_results_cpu_bl, pc_times_cpu_bl, pc_anomalies_cpu_bl = load_results('pc/cpu', 'bl')
    pc_results_cuda_bl, pc_times_cuda_bl, pc_anomalies_cuda_bl = load_results('pc/cuda', 'bl')

    pc_results_cpu_topiq_nr, pc_times_cpu_topiq_nr, pc_anomalies_cpu_topiq_nr = load_results('pc/cpu', 'topiq_nr')
    pc_results_cuda_topiq_nr, pc_times_cuda_topiq_nr, pc_anomalies_cuda_topiq_nr = load_results('pc/cuda', 'topiq_nr')

    pc_results_cpu_topiq_nr_spaq, pc_times_cpu_topiq_nr_spaq, pc_anomalies_cpu_topiq_nr_spaq = load_results('pc/cpu', 'topiq_nr-spaq')
    pc_results_cuda_topiq_nr_spaq, pc_times_cuda_topiq_nr_spaq, pc_anomalies_cuda_topiq_nr_spaq = load_results('pc/cuda', 'topiq_nr-spaq')

    # Load Android results
    android_results_bl, android_times_bl, android_anomalies_bl = load_results('android', 'bl')
    android_results_topiq_nr, android_times_topiq_nr, android_anomalies_topiq_nr = load_results('android', 'topiq_nr')
    android_results_topiq_nr_spaq, android_times_topiq_nr_spaq, android_anomalies_topiq_nr_spaq = load_results('android', 'topiq_nr-spaq')
    
    # Load Android Emulated results
    android_emulated_results_bl, android_emulated_times_bl, android_emulated_anomalies_bl = load_results('android_emulated', 'bl')
    android_emulated_results_topiq_nr, android_emulated_times_topiq_nr, android_emulated_anomalies_topiq_nr = load_results('android_emulated', 'topiq_nr')
    android_emulated_results_topiq_nr_spaq, android_emulated_times_topiq_nr_spaq, android_emulated_anomalies_topiq_nr_spaq = load_results('android_emulated', 'topiq_nr-spaq')

    # Create a dictionary of all anomaly dataframes
    anomalies = {
        'pc_cpu_bl': pc_anomalies_cpu_bl,
        'pc_cuda_bl': pc_anomalies_cuda_bl,
        'pc_cpu_topiq_nr': pc_anomalies_cpu_topiq_nr,
        'pc_cuda_topiq_nr': pc_anomalies_cuda_topiq_nr,
        'pc_cpu_topiq_nr_spaq': pc_anomalies_cpu_topiq_nr_spaq,
        'pc_cuda_topiq_nr_spaq': pc_anomalies_cuda_topiq_nr_spaq,
        'android_bl': android_anomalies_bl,
        'android_topiq_nr': android_anomalies_topiq_nr,
        'android_topiq_nr_spaq': android_anomalies_topiq_nr_spaq,
        'android_emulated_bl': android_emulated_anomalies_bl,
        'android_emulated_topiq_nr': android_emulated_anomalies_topiq_nr,
        'android_emulated_topiq_nr_spaq': android_emulated_anomalies_topiq_nr_spaq,
    }

    # Count anomalies in each result set
    anomaly_counts = count_anomalies(anomalies)

    # Print the number of anomalies in each result set
    for key, value in anomaly_counts.items():
        print(f"The number of anomalies in {key} is {value}.")

    # Load MOS results
    mos_results_df = pd.read_excel('./datasets/SPAQ/TestAnnotations/MOS and Image attribute scores.xlsx')
    

    # Calculate PLCC, SRCC, KRCC, mean, and standard deviation
    # PC
    bl_cpu_plcc, bl_cpu_plcc_mean, bl_cpu_plcc_std, bl_cpu_srcc, bl_cpu_srcc_mean, bl_cpu_srcc_std, bl_cpu_krcc, bl_cpu_krcc_mean, bl_cpu_krcc_std = calculate_results(pc_results_cpu_bl, mos_results_df, 'Score', 'MOS')
    bl_cuda_plcc, bl_cuda_plcc_mean, bl_cuda_plcc_std, bl_cuda_srcc, bl_cuda_srcc_mean, bl_cuda_srcc_std, bl_cuda_krcc, bl_cuda_krcc_mean, bl_cuda_krcc_std = calculate_results(pc_results_cuda_bl, mos_results_df, 'Score', 'MOS')

    topiq_nr_cpu_plcc, topiq_nr_cpu_plcc_mean, topiq_nr_cpu_plcc_std, topiq_nr_cpu_srcc, topiq_nr_cpu_srcc_mean, topiq_nr_cpu_srcc_std, topiq_nr_cpu_krcc, topiq_nr_cpu_krcc_mean, topiq_nr_cpu_krcc_std = calculate_results(pc_results_cpu_topiq_nr, mos_results_df, 'Score', 'MOS')
    topiq_nr_cuda_plcc, topiq_nr_cuda_plcc_mean, topiq_nr_cuda_plcc_std, topiq_nr_cuda_srcc, topiq_nr_cuda_srcc_mean, topiq_nr_cuda_srcc_std, topiq_nr_cuda_krcc, topiq_nr_cuda_krcc_mean, topiq_nr_cuda_krcc_std = calculate_results(pc_results_cuda_topiq_nr, mos_results_df, 'Score', 'MOS')

    topiq_nr_spaq_cpu_plcc, topiq_nr_spaq_cpu_plcc_mean, topiq_nr_spaq_cpu_plcc_std, topiq_nr_spaq_cpu_srcc, topiq_nr_spaq_cpu_srcc_mean, topiq_nr_spaq_cpu_srcc_std, topiq_nr_spaq_cpu_krcc, topiq_nr_spaq_cpu_krcc_mean, topiq_nr_spaq_cpu_krcc_std = calculate_results(pc_results_cpu_topiq_nr_spaq, mos_results_df, 'Score', 'MOS')
    topiq_nr_spaq_cuda_plcc, topiq_nr_spaq_cuda_plcc_mean, topiq_nr_spaq_cuda_plcc_std, topiq_nr_spaq_cuda_srcc, topiq_nr_spaq_cuda_srcc_mean, topiq_nr_spaq_cuda_srcc_std, topiq_nr_spaq_cuda_krcc, topiq_nr_spaq_cuda_krcc_mean, topiq_nr_spaq_cuda_krcc_std = calculate_results(pc_results_cuda_topiq_nr_spaq, mos_results_df, 'Score', 'MOS')

    # Android
    bl_mobile_plcc, bl_mobile_plcc_mean, bl_mobile_plcc_std, bl_mobile_srcc, bl_mobile_srcc_mean, bl_mobile_srcc_std, bl_mobile_krcc, bl_mobile_krcc_mean, bl_mobile_krcc_std = calculate_results(android_results_bl, mos_results_df, 'Score', 'MOS')
    topiq_mobile_plcc, topiq_mobile_plcc_mean, topiq_mobile_plcc_std, topiq_mobile_srcc, topiq_mobile_srcc_mean, topiq_mobile_srcc_std, topiq_mobile_krcc, topiq_mobile_krcc_mean, topiq_mobile_krcc_std = calculate_results(android_results_topiq_nr, mos_results_df, 'Score', 'MOS')
    topiq_spaq_mobile_plcc, topiq_spaq_mobile_plcc_mean, topiq_spaq_mobile_plcc_std, topiq_spaq_mobile_srcc, topiq_spaq_mobile_srcc_mean, topiq_spaq_mobile_srcc_std, topiq_spaq_mobile_krcc, topiq_spaq_mobile_krcc_mean, topiq_spaq_mobile_krcc_std = calculate_results(android_results_topiq_nr_spaq, mos_results_df, 'Score', 'MOS')

    # Android Emulated
    bl_mobile_emulated_plcc, bl_mobile_emulated_plcc_mean, bl_mobile_emulated_plcc_std, bl_mobile_emulated_srcc, bl_mobile_emulated_srcc_mean, bl_mobile_emulated_srcc_std, bl_mobile_emulated_krcc, bl_mobile_emulated_krcc_mean, bl_mobile_emulated_krcc_std = calculate_results(android_emulated_results_bl, mos_results_df, 'Score', 'MOS')
    topiq_mobile_emulated_plcc, topiq_mobile_emulated_plcc_mean, topiq_mobile_emulated_plcc_std, topiq_mobile_emulated_srcc, topiq_mobile_emulated_srcc_mean, topiq_mobile_emulated_srcc_std, topiq_mobile_emulated_krcc, topiq_mobile_emulated_krcc_mean, topiq_mobile_emulated_krcc_std = calculate_results(android_emulated_results_topiq_nr, mos_results_df, 'Score', 'MOS')
    topiq_spaq_mobile_emulated_plcc, topiq_spaq_mobile_emulated_plcc_mean, topiq_spaq_mobile_emulated_plcc_std, topiq_spaq_mobile_emulated_srcc, topiq_spaq_mobile_emulated_srcc_mean, topiq_spaq_mobile_emulated_srcc_std, topiq_spaq_mobile_emulated_krcc, topiq_spaq_mobile_emulated_krcc_mean, topiq_spaq_mobile_emulated_krcc_std = calculate_results(android_emulated_results_topiq_nr_spaq, mos_results_df, 'Score', 'MOS')


    # Calculate times
    # PC
    bl_cpu_total_time_sum, bl_cpu_total_time_mean, bl_cpu_inference_time_mean, bl_cpu_preprocessing_time_mean, bl_cpu_total_time_diff = calculate_times(pc_results_cpu_bl, pc_times_cpu_bl)
    bl_cuda_total_time_sum, bl_cuda_total_time_mean, bl_cuda_inference_time_mean, bl_cuda_preprocessing_time_mean, bl_cuda_total_time_diff = calculate_times(pc_results_cuda_bl, pc_times_cuda_bl)

    topiq_nr_cpu_total_time_sum, topiq_nr_cpu_total_time_mean, topiq_nr_cpu_inference_time_mean, topiq_nr_cpu_preprocessing_time_mean, topiq_nr_cpu_total_time_diff = calculate_times(pc_results_cpu_topiq_nr, pc_times_cpu_topiq_nr)
    topiq_nr_cuda_total_time_sum, topiq_nr_cuda_total_time_mean, topiq_nr_cuda_inference_time_mean, topiq_nr_cuda_preprocessing_time_mean, topiq_nr_cuda_total_time_diff = calculate_times(pc_results_cuda_topiq_nr, pc_times_cuda_topiq_nr)

    topiq_nr_spaq_cpu_total_time_sum, topiq_nr_spaq_cpu_total_time_mean, topiq_nr_spaq_cpu_inference_time_mean, topiq_nr_spaq_cpu_preprocessing_time_mean, topiq_nr_spaq_cpu_total_time_diff = calculate_times(pc_results_cpu_topiq_nr_spaq, pc_times_cpu_topiq_nr_spaq)
    topiq_nr_spaq_cuda_total_time_sum, topiq_nr_spaq_cuda_total_time_mean, topiq_nr_spaq_cuda_inference_time_mean, topiq_nr_spaq_cuda_preprocessing_time_mean, topiq_nr_spaq_cuda_total_time_diff = calculate_times(pc_results_cuda_topiq_nr_spaq, pc_times_cuda_topiq_nr_spaq)

    # Android
    android_bl_total_time_sum, android_bl_total_time_mean, android_bl_inference_time_mean, android_bl_preprocessing_time_mean, android_bl_total_time_diff = calculate_times(android_results_bl, android_times_bl)
    android_topiq_nr_total_time_sum, android_topiq_nr_total_time_mean, android_topiq_nr_inference_time_mean, android_topiq_nr_preprocessing_time_mean, android_topiq_nr_total_time_diff = calculate_times(android_results_topiq_nr, android_times_topiq_nr)
    android_topiq_nr_spaq_total_time_sum, android_topiq_nr_spaq_total_time_mean, android_topiq_nr_spaq_inference_time_mean, android_topiq_nr_spaq_preprocessing_time_mean, android_topiq_nr_spaq_total_time_diff = calculate_times(android_results_topiq_nr_spaq, android_times_topiq_nr_spaq)

    # Android Emulated
    android_emulated_bl_total_time_sum, android_emulated_bl_total_time_mean, android_emulated_bl_inference_time_mean, android_emulated_bl_preprocessing_time_mean, android_emulated_bl_total_time_diff = calculate_times(android_emulated_results_bl, android_emulated_times_bl)
    android_emulated_topiq_nr_total_time_sum, android_emulated_topiq_nr_total_time_mean, android_emulated_topiq_nr_inference_time_mean, android_emulated_topiq_nr_preprocessing_time_mean, android_emulated_topiq_nr_total_time_diff = calculate_times(android_emulated_results_topiq_nr, android_emulated_times_topiq_nr)
    android_emulated_topiq_nr_spaq_total_time_sum, android_emulated_topiq_nr_spaq_total_time_mean, android_emulated_topiq_nr_spaq_inference_time_mean, android_emulated_topiq_nr_spaq_preprocessing_time_mean, android_emulated_topiq_nr_spaq_total_time_diff = calculate_times(android_emulated_results_topiq_nr_spaq, android_emulated_times_topiq_nr_spaq)
    
    # Save results
    # PC
    results_pc_cpu = {
        'model': ['BL CPU', 'TopIQ NR CPU', 'TopIQ NR SPAQ CPU'],
        'PLCC': [bl_cpu_plcc, topiq_nr_cpu_plcc, topiq_nr_spaq_cpu_plcc],
        'PLCC Mean': [bl_cpu_plcc_mean, topiq_nr_cpu_plcc_mean, topiq_nr_spaq_cpu_plcc_mean],
        'PLCC Std': [bl_cpu_plcc_std, topiq_nr_cpu_plcc_std, topiq_nr_spaq_cpu_plcc_std],
        'SRCC': [bl_cpu_srcc, topiq_nr_cpu_srcc, topiq_nr_spaq_cpu_srcc],
        'SRCC Mean': [bl_cpu_srcc_mean, topiq_nr_cpu_srcc_mean, topiq_nr_spaq_cpu_srcc_mean],
        'SRCC Std': [bl_cpu_srcc_std, topiq_nr_cpu_srcc_std, topiq_nr_spaq_cpu_srcc_std],
        'KRCC': [bl_cpu_krcc, topiq_nr_cpu_krcc, topiq_nr_spaq_cpu_krcc],
        'KRCC Mean': [bl_cpu_krcc_mean, topiq_nr_cpu_krcc_mean, topiq_nr_spaq_cpu_krcc_mean],
        'KRCC Std': [bl_cpu_krcc_std, topiq_nr_cpu_krcc_std, topiq_nr_spaq_cpu_krcc_std],
        'Total time sum': [bl_cpu_total_time_sum, topiq_nr_cpu_total_time_sum, topiq_nr_spaq_cpu_total_time_sum],
        'Total time mean': [bl_cpu_total_time_mean, topiq_nr_cpu_total_time_mean, topiq_nr_spaq_cpu_total_time_mean],
        'Inference time mean': [bl_cpu_inference_time_mean, topiq_nr_cpu_inference_time_mean, topiq_nr_spaq_cpu_inference_time_mean],
        'Preprocessing time mean': [bl_cpu_preprocessing_time_mean, topiq_nr_cpu_preprocessing_time_mean, topiq_nr_spaq_cpu_preprocessing_time_mean],
        'Total time diff': [bl_cpu_total_time_diff, topiq_nr_cpu_total_time_diff, topiq_nr_spaq_cpu_total_time_diff],
        'Number of anomalies': [len(pc_anomalies_cpu_bl), len(pc_anomalies_cpu_topiq_nr), len(pc_anomalies_cpu_topiq_nr_spaq)]
    }
    results_pc_df = pd.DataFrame(results_pc_cpu)
    results_pc_df.to_csv('./results/processed/results_pc_cpu.csv', index=False)

    # PC CUDA
    results_pc_cuda = {
        'model': ['BL CUDA', 'TopIQ NR CUDA', 'TopIQ NR SPAQ CUDA'],
        'PLCC': [bl_cuda_plcc, topiq_nr_cuda_plcc, topiq_nr_spaq_cuda_plcc],
        'PLCC Mean': [bl_cuda_plcc_mean, topiq_nr_cuda_plcc_mean, topiq_nr_spaq_cuda_plcc_mean],
        'PLCC Std': [bl_cuda_plcc_std, topiq_nr_cuda_plcc_std, topiq_nr_spaq_cuda_plcc_std],
        'SRCC': [bl_cuda_srcc, topiq_nr_cuda_srcc, topiq_nr_spaq_cuda_srcc],
        'SRCC Mean': [bl_cuda_srcc_mean, topiq_nr_cuda_srcc_mean, topiq_nr_spaq_cuda_srcc_mean],
        'SRCC Std': [bl_cuda_srcc_std, topiq_nr_cuda_srcc_std, topiq_nr_spaq_cuda_srcc_std],
        'KRCC': [bl_cuda_krcc, topiq_nr_cuda_krcc, topiq_nr_spaq_cuda_krcc],
        'KRCC Mean': [bl_cuda_krcc_mean, topiq_nr_cuda_krcc_mean, topiq_nr_spaq_cuda_krcc_mean],
        'KRCC Std': [bl_cuda_krcc_std, topiq_nr_cuda_krcc_std, topiq_nr_spaq_cuda_krcc_std],
        'Total time sum': [bl_cuda_total_time_sum, topiq_nr_cuda_total_time_sum, topiq_nr_spaq_cuda_total_time_sum],
        'Total time mean': [bl_cuda_total_time_mean, topiq_nr_cuda_total_time_mean, topiq_nr_spaq_cuda_total_time_mean],
        'Inference time mean': [bl_cuda_inference_time_mean, topiq_nr_cuda_inference_time_mean, topiq_nr_spaq_cuda_inference_time_mean],
        'Preprocessing time mean': [bl_cuda_preprocessing_time_mean, topiq_nr_cuda_preprocessing_time_mean, topiq_nr_spaq_cuda_preprocessing_time_mean],
        'Total time diff': [bl_cuda_total_time_diff, topiq_nr_cuda_total_time_diff, topiq_nr_spaq_cuda_total_time_diff],
        'Number of anomalies': [len(pc_anomalies_cuda_bl), len(pc_anomalies_cuda_topiq_nr), len(pc_anomalies_cuda_topiq_nr_spaq)]
    }
    results_pc_df = pd.DataFrame(results_pc_cuda)
    results_pc_df.to_csv('./results/processed/results_pc_cuda.csv', index=False)

    # Android
    results_android = {
        'model': ['BL Mobile', 'TopIQ Mobile', 'TopIQ SPAQ Mobile'],
        'PLCC': [bl_mobile_plcc, topiq_mobile_plcc, topiq_spaq_mobile_plcc],
        'PLCC Mean': [bl_mobile_plcc_mean, topiq_mobile_plcc_mean, topiq_spaq_mobile_plcc_mean],
        'PLCC Std': [bl_mobile_plcc_std, topiq_mobile_plcc_std, topiq_spaq_mobile_plcc_std],
        'SRCC': [bl_mobile_srcc, topiq_mobile_srcc, topiq_spaq_mobile_srcc],
        'SRCC Mean': [bl_mobile_srcc_mean, topiq_mobile_srcc_mean, topiq_spaq_mobile_srcc_mean],
        'SRCC Std': [bl_mobile_srcc_std, topiq_mobile_srcc_std, topiq_spaq_mobile_srcc_std],
        'KRCC': [bl_mobile_krcc, topiq_mobile_krcc, topiq_spaq_mobile_krcc],
        'KRCC Mean': [bl_mobile_krcc_mean, topiq_mobile_krcc_mean, topiq_spaq_mobile_krcc_mean],
        'KRCC Std': [bl_mobile_krcc_std, topiq_mobile_krcc_std, topiq_spaq_mobile_krcc_std],
        'Total time sum': [android_bl_total_time_sum, android_topiq_nr_total_time_sum, android_topiq_nr_spaq_total_time_sum],
        'Total time mean': [android_bl_total_time_mean, android_topiq_nr_total_time_mean, android_topiq_nr_spaq_total_time_mean],
        'Inference time mean': [android_bl_inference_time_mean, android_topiq_nr_inference_time_mean, android_topiq_nr_spaq_inference_time_mean],
        'Preprocessing time mean': [android_bl_preprocessing_time_mean, android_topiq_nr_preprocessing_time_mean, android_topiq_nr_spaq_preprocessing_time_mean],
        'Total time diff': [android_bl_total_time_diff, android_topiq_nr_total_time_diff, android_topiq_nr_spaq_total_time_diff],
        'Number of anomalies': [len(android_anomalies_bl), len(android_anomalies_topiq_nr), len(android_anomalies_topiq_nr_spaq)]
    }
    results_android_df = pd.DataFrame(results_android)
    results_android_df.to_csv('./results/processed/results_android.csv', index=False)

    # Android Emulated
    results_android_emulated = {
        'model': ['BL Mobile Emulated', 'TopIQ Mobile Emulated', 'TopIQ SPAQ Mobile Emulated'],
        'PLCC': [bl_mobile_emulated_plcc, topiq_mobile_emulated_plcc, topiq_spaq_mobile_emulated_plcc],
        'PLCC Mean': [bl_mobile_emulated_plcc_mean, topiq_mobile_emulated_plcc_mean, topiq_spaq_mobile_emulated_plcc_mean],
        'PLCC Std': [bl_mobile_emulated_plcc_std, topiq_mobile_emulated_plcc_std, topiq_spaq_mobile_emulated_plcc_std],
        'SRCC': [bl_mobile_emulated_srcc, topiq_mobile_emulated_srcc, topiq_spaq_mobile_emulated_srcc],
        'SRCC Mean': [bl_mobile_emulated_srcc_mean, topiq_mobile_emulated_srcc_mean, topiq_spaq_mobile_emulated_srcc_mean],
        'SRCC Std': [bl_mobile_emulated_srcc_std, topiq_mobile_emulated_srcc_std, topiq_spaq_mobile_emulated_srcc_std],
        'KRCC': [bl_mobile_emulated_krcc, topiq_mobile_emulated_krcc, topiq_spaq_mobile_emulated_krcc],
        'KRCC Mean': [bl_mobile_emulated_krcc_mean, topiq_mobile_emulated_krcc_mean, topiq_spaq_mobile_emulated_krcc_mean],
        'KRCC Std': [bl_mobile_emulated_krcc_std, topiq_mobile_emulated_krcc_std, topiq_spaq_mobile_emulated_krcc_std],
        'Total time sum': [android_emulated_bl_total_time_sum, android_emulated_topiq_nr_total_time_sum, android_emulated_topiq_nr_spaq_total_time_sum],
        'Total time mean': [android_emulated_bl_total_time_mean, android_emulated_topiq_nr_total_time_mean, android_emulated_topiq_nr_spaq_total_time_mean],
        'Inference time mean': [android_emulated_bl_inference_time_mean, android_emulated_topiq_nr_inference_time_mean, android_emulated_topiq_nr_spaq_inference_time_mean],
        'Preprocessing time mean': [android_emulated_bl_preprocessing_time_mean, android_emulated_topiq_nr_preprocessing_time_mean, android_emulated_topiq_nr_spaq_preprocessing_time_mean],
        'Total time diff': [android_emulated_bl_total_time_diff, android_emulated_topiq_nr_total_time_diff, android_emulated_topiq_nr_spaq_total_time_diff],
        'Number of anomalies': [len(android_emulated_anomalies_bl), len(android_emulated_anomalies_topiq_nr), len(android_emulated_anomalies_topiq_nr_spaq)]
    }
    results_android_emulated_df = pd.DataFrame(results_android_emulated)
    results_android_emulated_df.to_csv('./results/processed/results_android_emulated.csv', index=False)


if __name__ == '__main__':
    print("Processing results...")
    main()
    print("Done!")