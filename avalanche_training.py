# Main training loop with evaluation metrics
import importlib
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics

from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive, Replay
from avalanche.evaluation import Metric

from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics,
    cpu_usage_metrics, disk_usage_metrics, ram_usage_metrics
)
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from TTA_modules import TestTimeAdaptationPlugin
import config
from custom_plugin import ProportionalReplayPlugin
from modelstruct import Compact1DCNN
from utils import ACI_CATEGORY_MAPPING, ACI_PROPORTION_MAPPING

# Create evaluation plugin with TTA metric
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    ram_usage_metrics(experience=True),
    loggers=[InteractiveLogger()],
    strict_checks=False
)

def plot_results(results, run_dir):
    epoch_accuracies = [result['Top1_Acc_Epoch/train_phase/train_stream'] for result in results]
    stream_accuracies = [result['Top1_Acc_Stream/eval_phase/test_stream'] for result in results]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot accuracies
    plt.plot(range(len(results)), epoch_accuracies, label='Epoch Accuracy', marker='o')
    plt.plot(range(len(results)), stream_accuracies, label='Stream Accuracy', marker='o')

    # Adding labels and title
    plt.xlabel('Task')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Top-1 Accuracy Over Time')
    plt.xticks(range(len(results)))
    plt.legend()

    # Save the plot to the run directory
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'accuracy_over_time.png'))
    plt.close()

def train_clm(benchmark, num_features, num_classes, run_dir):
    """
    Train the continual learning model with the current configuration.
    
    Args:
        benchmark: The Avalanche benchmark
        num_features: Number of input features
        num_classes: Number of output classes
        run_dir: Directory to save results
    """
    # Clear CUDA cache and ensure clean state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Create new model instance
    model = Compact1DCNN(num_features=num_features, num_classes=num_classes).to(config.DEVICE)
    model.train()
    
    # Print current configuration for debugging
    print("\nCurrent Configuration:")
    print(f"USE_TTA: {config.USE_TTA}")
    print(f"STRATEGY: {config.STRATEGY}")
    print(f"PLUGIN: {config.PLUGIN}")
    print(f"MEM_SIZE: {config.MEM_SIZE}")
    print(f"WEIGHT_SHIFT_FACTOR: {config.WEIGHT_SHIFT_FACTOR}")
    print(f"LOGIT_SHIFT_FACTOR: {config.LOGIT_SHIFT_FACTOR}")
    print(f"NORMALIZE_WEIGHTS: {config.NORMALIZE_WEIGHTS}")

    if config.PLUGIN:
        plugins = [ProportionalReplayPlugin(mem_size=config.MEM_SIZE, category_mapping=ACI_CATEGORY_MAPPING, proportion_mapping=ACI_PROPORTION_MAPPING)]
    else:
        plugins = []

    if config.STRATEGY == "Replay":
        cl_strategy = Replay(
            model = model,
            optimizer = Adam(model.parameters(), lr=0.01),
            criterion = CrossEntropyLoss(),
            mem_size=config.MEM_SIZE,
            train_mb_size=config.TRAIN_MB_SIZE,
            train_epochs=config.TRAIN_EPOCHS,
            eval_mb_size=config.EVAL_MB_SIZE,
            device = config.DEVICE,
            evaluator=eval_plugin,
            plugins=plugins
        )
    else:
        cl_strategy = Naive(
            model = model,
            optimizer = Adam(model.parameters(), lr=0.01),
            criterion = CrossEntropyLoss(),
            train_mb_size=config.TRAIN_MB_SIZE,
            train_epochs=config.TRAIN_EPOCHS,
            eval_mb_size=config.EVAL_MB_SIZE,
            device = config.DEVICE,
            evaluator=eval_plugin,
            plugins=plugins
        )

    # Train and Evaluate
    results = []
    
    for experience in benchmark.train_stream:
        print("Input data shape:", experience.dataset[0][0].shape)
        print("\033[92mStart of experience:\033[0m", experience.current_experience)    
        
        # Train on current experience
        cl_strategy.train(experience, num_workers=4)
        print('Training completed')
        
        # Regular evaluation (includes TTA if enabled)
        eval_result = cl_strategy.eval(benchmark.test_stream, num_workers=4)
        results.append(eval_result)

    # Plot and save results
    plot_results(results, run_dir)
    
    # Clean up
    del model
    del cl_strategy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results