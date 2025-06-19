#Batch testing code - either grid search or some predefined configurations.
import importlib
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
from main import run_experiment
import config
import sys

def update_config(config_dict):
    """Update config.py values with the given configuration"""
    print("\nUpdating configuration:")
    
    # First, update the values in the config module
    for key, value in config_dict.items():
        if hasattr(config, key):
            old_value = getattr(config, key)
            setattr(config, key, value)
            print(f"  {key}: {old_value} -> {value}")
        else:
            print(f"  Warning: {key} not found in config.py")
    
    # Force reload the config module
    importlib.reload(config)
    
    # Update the values again after reload to ensure they stick
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Reload dependent modules
    importlib.reload(importlib.import_module('main'))
    importlib.reload(importlib.import_module('modelstruct'))
    importlib.reload(importlib.import_module('avalanche_training'))
    
    # Force reload of any modules that might have imported these modules
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('avalanche_training') or module_name.startswith('modelstruct'):
            importlib.reload(sys.modules[module_name])
    
    # Verify the configuration was updated
    print("\nVerifying configuration update:")
    for key, value in config_dict.items():
        if hasattr(config, key):
            current_value = getattr(config, key)
            if current_value != value:
                print(f"  Warning: {key} was not properly updated. Expected {value}, got {current_value}")
            else:
                print(f"  {key}: {current_value} (verified)")

def create_grid_configurations():
    """Generate different configurations using grid search"""
    configs = []
    
    # Define parameter ranges to test
    param_ranges = {
        'USE_TTA': [True, False],
        #'STRATEGY': ['Replay', 'Naive'],
        #'PLUGIN': [True, False],
    }
    
    # Generate all combinations
    keys = param_ranges.keys()
    values = param_ranges.values()
    for combination in product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs

def create_targeted_configurations():
    """Generate targeted configurations for specific comparisons"""
    configs = [
        { #Baseline
            'NAME': 'Baseline',
            'USE_TTA': False,
            'STRATEGY': 'Naive',
            'PLUGIN': False,
            'LOGIT_SHIFT_FACTOR': 0,
            'WEIGHT_SHIFT_FACTOR': 0
        },
        { #Replay without plugin
            'NAME': 'Replay',
            'USE_TTA': False,
            'STRATEGY': 'Replay',
            'PLUGIN': False,
            'MEM_SIZE': 100,
            'LOGIT_SHIFT_FACTOR': 0,
            'WEIGHT_SHIFT_FACTOR': 0
        },
        { #Replay with dynamic buffer plugin
            'NAME': 'Replay + Dynamic Buffer',
            'USE_TTA': False,
            'STRATEGY': 'Replay',
            'PLUGIN': True,
            'MEM_SIZE': 50,
            'LOGIT_SHIFT_FACTOR': 0,
            'WEIGHT_SHIFT_FACTOR': 0
        },
        { #Replay logit shift
            'NAME': 'Replay + Logit Shift',
            'USE_TTA': False,
            'STRATEGY': 'Replay',
            'PLUGIN': False,
            'MEM_SIZE': 100,
            'LOGIT_SHIFT_FACTOR': 1.0,
            'WEIGHT_SHIFT_FACTOR': 0
        },
        { #Replay weight shift
            'NAME': 'Replay + Weight Shift',
            'USE_TTA': True,
            'STRATEGY': 'Replay',
            'PLUGIN': False,
            'MEM_SIZE': 100,
            'LOGIT_SHIFT_FACTOR': 0,
            'WEIGHT_SHIFT_FACTOR': 1.0
        }
    ]
    return configs

def process_results(results):
    """Process experiment results into a structured format"""
    processed_results = []
    
    for i, result in enumerate(results):
        config = result['config']
        metrics = result['metrics']
        
        # Extract metrics for each experience
        for exp_idx, exp_metrics in enumerate(metrics):
            processed_result = {
                'config_id': i,
                'experience': exp_idx,
                'config': config,
                'accuracy': exp_metrics.get('Top1_Acc_Stream/eval_phase/test_stream', 0),
                'loss': exp_metrics.get('Loss_Stream/eval_phase/test_stream', 0),
                'forgetting': exp_metrics.get('Forgetting_Stream/eval_phase/test_stream', 0)
            }
            processed_results.append(processed_result)
    
    return processed_results

def save_results_summary(results, output_dir):
    """Save a summary of all experiment results"""
    # Save detailed results
    summary_path = os.path.join(output_dir, 'experiment_summary.csv')
    pd.DataFrame(results).to_csv(summary_path, index=False)
    
    # Save results as JSON for easier parsing
    json_path = os.path.join(output_dir, 'experiment_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

def plot_comparison(results, output_dir):
    """Create comparison plots for different metrics"""
    df = pd.DataFrame(results)
    
    # Set style for better visualization
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create plots for different metrics
    metrics = ['accuracy', 'loss', 'forgetting']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Group by configuration and plot
        for config_id, group in df.groupby('config_id'):
            # Get the configuration name
            config = group['config'].iloc[0]
            config_name = config.get('NAME', f'Config {config_id}')
            
            # Convert to numpy arrays for plotting
            x = group['experience'].values
            y = group[metric].values
            
            plt.plot(x, y, label=config_name, marker='o')
        
        plt.xlabel('Experience')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison Across Configurations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), bbox_inches='tight')
        plt.close()
    
    # Create a summary plot showing final performance
    plt.figure(figsize=(12, 6))
    final_results = df[df['experience'] == df['experience'].max()]
    
    # Sort by accuracy
    final_results = final_results.sort_values('accuracy', ascending=False)
    
    # Create a bar plot
    x = range(len(final_results))
    y = final_results['accuracy'].values
    plt.bar(x, y)
    
    # Create labels using configuration names
    labels = []
    for _, row in final_results.iterrows():
        config = row['config']
        config_name = config.get('name', f'Config {row["config_id"]}')
        labels.append(config_name)
    
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy Comparison Across Configurations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_accuracy_comparison.png'))
    plt.close()

def run_experiments(configs, experiment_dir):
    """Run experiments for the given configurations"""
    results = []
    
    # Run experiments for each configuration
    for i, config_dict in enumerate(configs):
        print(f"\nRunning experiment {i+1}/{len(configs)}")
        print(f"Configuration: {config_dict}")
        
        # Update config.py with current configuration
        update_config(config_dict)
        
        # Verify configuration was updated
        print("\nCurrent configuration state:")
        for key in config_dict:
            if hasattr(config, key):
                print(f"  {key}: {getattr(config, key)}")
        
        # Run experiment
        run_dir = os.path.join(experiment_dir, f'config_{i}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Run the experiment with current configuration
        result = run_experiment(run_dir=run_dir)
        result['config'] = config_dict  # Add the config used to the result dictionary
        results.append(result)
    
    return results

def main(mode='targeted'):
    """
    Run experiments in either grid search or targeted mode.
    
    Args:
        mode (str): Either 'grid' for grid search or 'targeted' for specific comparisons
    """
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(config.OUTPUT_PATH, f'experiment_{timestamp}_{mode}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Generate configurations based on mode
    if mode == 'grid':
        configs = create_grid_configurations()
    else:  # targeted
        configs = create_targeted_configurations()
    
    # Run experiments
    results = run_experiments(configs, experiment_dir)
    
    # Process and save results
    processed_results = process_results(results)
    save_results_summary(processed_results, experiment_dir)
    plot_comparison(processed_results, experiment_dir)
    
    print(f"\nExperiment results saved in: {experiment_dir}")

if __name__ == "__main__":
    # Run in targeted mode by default
    main(mode='targeted') 