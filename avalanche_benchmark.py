#This creates the avalanche benchmark dataset - either domain, class, or gradual.

from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import task_incremental_benchmark
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from collections import Counter

from avalanche.benchmarks.utils import AvalancheDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
from config import BENCHMARK_SEL, DATA_PATH
from dataprocess import preprocess_data, process_data

# Step 1: Convert to PyTorch Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, labels, indices=None):
        self.df = df
        self.labels = labels
        if indices is not None:
            self.df = self.df.iloc[indices]
        self.data = self.df.drop(columns='label').values
        self.targets = self.df['label'].values  # Ensure targets are defined

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        target = self.targets[idx]
        return sample, target



def createClassIncrementalBenchmark(df):
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Create the dataset for Avalanche
    train_data = CustomDataset(train_df, train_df['label'])
    test_data = CustomDataset(test_df, test_df['label'])


    #train_data = CustomDataset(data, data['label'])
    #test_data = CustomDataset(data, data['label'])

    datasets_dict = {
        "train": train_data,  # your training data as AvalancheDataset
        "test": test_data     # your test data as AvalancheDataset
    }

    class_order = list(range(df['label'].nunique()))

    # Define how many classes per experience (if not None)
    num_classes_per_exp = [2, 3, 5]  # Example: Experience 1 has 2 classes, Experience 2 has 3, etc.

    # Create the class-incremental benchmark scenario
    benchmark = class_incremental_benchmark(
        datasets_dict=datasets_dict,
        class_order=class_order,  # or None to shuffle classes
        num_experiences=df['label'].nunique(),  # Number of experiences (you can adjust this based on your scenario)
    )
    
    for experience in benchmark.train_stream:
        print(f"\tSample: {experience.dataset[0]}")
        print(f"Experience {experience.current_experience}")
        print(f"Classes in this experience: {experience.classes_in_this_experience}")
        print(f"Previous classes: {experience.classes_seen_so_far}")
        print(f"Future classes: {experience.future_classes}")
    
    return benchmark

def createDomainStreams(df, num, test_ratio):
    num_domains = num
    domain_size = len(df) // num_domains
    datasets_train = []
    datasets_test = []

    for i in range(num_domains):
        # Get the range of indices for this domain
        indices = range(i * domain_size, (i + 1) * domain_size)

        # Split indices into train and test
        test_indices = indices[::int(1 / test_ratio)]  # Every nth sample for test
        train_indices = [idx for idx in indices if idx not in test_indices]

        # Create CustomDataset for train and test
        train_dataset = CustomDataset(df.iloc[train_indices], df['label'])
        test_dataset = CustomDataset(df.iloc[test_indices], df['label'])

        # Convert to AvalancheDataset
        datasets_train.append(AvalancheDataset(train_dataset))
        datasets_test.append(AvalancheDataset(test_dataset))

    return datasets_train, datasets_test

def createDomainIncrementalBenchmark(df):
    # First, prepare the domain streams for later experiences
    train_data, test_data = createDomainStreams(df, 10, 0.2)

    # Get a random subset of all data for the first experience (training on all classes)
    all_indices = df.index.tolist()
    random.shuffle(all_indices)
    first_experience_size = int(0.2 * len(df))  # You can adjust this size based on your scenario
    first_experience_indices = all_indices[:first_experience_size]
    first_experience_data = df.iloc[first_experience_indices]

    # Create a CustomDataset for the first experience
    first_experience_train = CustomDataset(first_experience_data, first_experience_data['label'])
    first_experience_test = CustomDataset(df.iloc[first_experience_indices], df['label'])

    # Create the first experience using the mixed dataset
    datasets_train = [AvalancheDataset(first_experience_train)] + [d for d in train_data]
    datasets_test = [AvalancheDataset(first_experience_test)] + [d for d in test_data]

    # Create the benchmark with the updated domain streams
    benchmark = benchmark_from_datasets(
        train=datasets_train,
        test=datasets_test
    )

    return benchmark

def createGradualIncrementalBenchmark(df):
    train_data, test_data = createDomainStreams(df, 10, 0.2)

    benchmark = benchmark_from_datasets(
        train=train_data,
        test=test_data
    )
    return benchmark

def displayClassDistribution(benchmark):
    print("\nClass distribution per experience:")
    for stream_name, stream in [("Train", benchmark.train_stream), ("Test", benchmark.test_stream)]:
        print(f"\n{stream_name} Stream:")
        for experience in stream:
            # Get dataset for the experience
            experience_dataset = experience.dataset
            # Count the classes
            class_counts = Counter()
            
            for data in experience_dataset:
                if isinstance(data, (tuple, list)) and len(data) >= 2:
                    _, target = data[:2]  # Extract the second element as the label
                    class_counts[target] += 1
                else:
                    raise ValueError(f"Unexpected dataset format: {data}")

            print(f"Experience {experience.current_experience}: {dict(class_counts)}")

def createBenchmark(data, max_class, min_class):
  df, num_classes, num_features = preprocess_data(data, max_class, min_class )
  if(BENCHMARK_SEL == 1):
      benchmark = createDomainIncrementalBenchmark(df)
      
  if(BENCHMARK_SEL == 2):
      benchmark = createClassIncrementalBenchmark(df)
      
  if(BENCHMARK_SEL == 3):
      benchmark = createGradualIncrementalBenchmark(df)
  
  displayClassDistribution(benchmark)
  return benchmark

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    benchmark = createBenchmark(data, 10, 5)
    displayClassDistribution(benchmark)