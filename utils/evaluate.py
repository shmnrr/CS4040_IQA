from BL_demo import Demo_Universal, Baseline
from model_utils import get_device, get_model, get_bl_model
import pandas as pd
import os

def evaluate_model(model, device, n_images=1000):
    print(f"Evaluating {model.metric_name if model.__class__ != Baseline else type(model).__name__} model")
    demo = Demo_Universal(model, device)
    results, time = demo.run_on_n_images(n_images)
    return results, time

def generate_dataframe(results, columns=['Image name', 'Score', 'Total time', 'Inference time', 'Preprocessing time']):
    return pd.DataFrame(results, columns=columns)

def save_results(df, filename):
    df.to_csv(filename, index=False)
    
def evaluate_and_save_results(models):
    results = {}
    dataframes = {}
    
    for model_name, device_name in models.items():
        # Get device
        device = get_device(device_name)
        
        # Get model
        if model_name == 'bl':
            model = get_bl_model(device)
        else:
            model = get_model(model_name, device)
        
        # Evaluate model
        results[model_name], runtime = evaluate_model(model, device)
        
        # Generate dataframe
        dataframes[model_name] = generate_dataframe(results[model_name])
        
        # Save results in csv in ./results
        if device_name == 'cpu':
            if model_name == 'bl':
                file_path = f'./results/pc/cpu/bl/{model_name}_{device_name}_results_{runtime}.csv'
            elif model_name == 'topiq_nr':
                file_path = f'./results/pc/cpu/topiq_nr/{model_name}_{device_name}_results_{runtime}.csv'
            else:
                file_path = f'./results/pc/cpu/topiq_nr-spaq/{model_name}_{device_name}_results_{runtime}.csv'
        else:
            if model_name == 'bl':
                file_path = f'./results/pc/cuda/bl/{model_name}_{device_name}_results_{runtime}.csv'
            elif model_name == 'topiq_nr':
                file_path = f'./results/pc/cuda/topiq_nr/{model_name}_{device_name}_results_{runtime}.csv'
            else:
                file_path = f'./results/pc/cuda/topiq_nr-spaq/{model_name}_{device_name}_results_{runtime}.csv'

        save_results(dataframes[model_name], file_path)

models_cpu = {
    'bl': 'cpu',
    'topiq_nr': 'cpu',
    'topiq_nr-spaq': 'cpu',
}

models_cuda = {
    'bl': 'cuda',
    # 'topiq_nr': 'cuda',
    # 'topiq_nr-spaq': 'cuda'
}

if __name__ == '__main__':
    for i in range(1):
        evaluate_and_save_results(models_cuda)
        