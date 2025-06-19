#This runs a single run of the model with the given configuration
import pandas as pd
import os
from datetime import datetime
from config import DATA_PATH, MAX_CLASS, MIN_CLASS, OUTPUT_PATH, save_config_details
from dataprocess import get_label_encoder, preprocess_data, split_data
from avalanche_benchmark import createBenchmark
from avalanche_training import train_clm


def run_experiment(run_dir=None):
    """
    Run a single experiment with the current configuration.
    
    Args:
        run_dir (str): Directory to save results
    
    Returns:
        dict: Experiment results including metrics and run directory
    """
    # Create timestamped run directory if not provided
    if run_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_PATH, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    df, num_classes, num_features = preprocess_data(data, max_class=MAX_CLASS, min_class=MIN_CLASS)
    
    # Create Avalanche benchmark
    print("Creating Avalanche benchmark...")
    benchmark = createBenchmark(df, num_classes, num_features)
    
    # Train with Avalanche
    print("Starting Avalanche training...")
    results = train_clm(benchmark, num_features, num_classes, run_dir)
    
    # Save configuration details
    print("Saving configuration details...")
    save_config_details(run_dir)
    
    return {
        'metrics': results,
        'run_dir': run_dir
    }

def main():
    # Run a single experiment with default configuration
    results = run_experiment()
    print(f"\nResults saved in: {results['run_dir']}")

if __name__ == "__main__":
    main()
