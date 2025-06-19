# Some attempts to create a custom replay buffer with higher performance. These didn't work; they're generally
# worse than normal replay.
from collections import Counter, defaultdict
from typing import Optional, TYPE_CHECKING
import random

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer

from utils import ACI_CATEGORY_MAPPING, ACI_PROPORTION_MAPPING

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from avalanche.benchmarks.utils import concat_datasets, AvalancheDataset, as_avalanche_dataset
from collections import Counter, defaultdict
import random


class ProportionalReplayBuffer(ExemplarsBuffer):
    """
    A custom storage policy that adjusts class proportions in memory dynamically
    based on a defined mapping.

    :param max_size: The maximum number of exemplars to store.
    :param category_mapping: A mapping from class indices to categories.
    :param proportion_mapping: A mapping from categories to desired proportions.
    """

    def __init__(self, max_size: int, category_mapping: dict, proportion_mapping: dict):
        super().__init__(max_size)
        self.category_mapping = category_mapping
        self.proportion_mapping = proportion_mapping
        self.class_to_indices = defaultdict(list)

    def resize(self, new_size: int):
      print(f"Resizing buffer from {len(self.buffer)} to {new_size}")
      """
      Resize the buffer to a new maximum size. If reducing size,
      remove excess samples while maintaining proportions.

      :param new_size: The new maximum size of the buffer.
      """
      if new_size >= len(self.buffer):
          # If increasing size or keeping it the same, just update max_size
          self.max_size = new_size
          return

      # If reducing size, remove excess samples while maintaining proportions
      current_category_counts = defaultdict(int)

      # Count current samples per category
      for sample in self.buffer:
          _, y = sample[0], sample[1]
          category = self.category_mapping[y]
          current_category_counts[category] += 1

      # Calculate target counts per category based on proportions
      total_samples = len(self.buffer)
      target_counts = {
          category: int(new_size * (current_count / total_samples))
          for category, current_count in current_category_counts.items()
      }

      # Create a new buffer with reduced size
      selected_samples = []
      remaining_per_category = target_counts.copy()

      for sample in self.buffer:
          _, y = sample[0], sample[1]
          category = self.category_mapping[y]
          if remaining_per_category[category] > 0:
              selected_samples.append(sample)
              remaining_per_category[category] -= 1

              if len(selected_samples) == new_size:
                  break

      # Update buffer and max size
      self.buffer = as_avalanche_dataset(selected_samples)
      self.max_size = new_size


    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Update the buffer by adding new exemplars from the current experience.
        """
        # Add new samples to the buffer
        new_data = strategy.experience.dataset
        if len(self.buffer) == 0:
            # If buffer is empty, initialize it with the new data
            self.buffer = new_data
        else:
            # Concatenate new data with existing buffer
            self.buffer = concat_datasets([self.buffer, new_data])
            print(f"Buffer size: {len(self.buffer)}")

        # Ensure buffer does not exceed max_size
        if len(self.buffer) > self.max_size:
            current_category = self.get_current_category(strategy)
            desired_proportions = self.proportion_mapping[current_category]
            self.rebalance_buffer(desired_proportions)

    def get_current_category(self, strategy: "SupervisedTemplate") -> str:
        """
        Determine the current category based on strategy predictions or context.
        For simplicity, we use the most frequent predicted class in the last batch.
        """
        if hasattr(strategy, 'predictions') and strategy.predictions is not None:
            predicted_classes = strategy.predictions.cpu().numpy()
            most_common_class = Counter(predicted_classes).most_common(1)[0][0]
            return self.category_mapping[most_common_class]
        else:
            # Default to 'Benign' if no predictions are available
            return "Benign"
    def rebalance_buffer2(self, desired_proportions: dict):
        """
        Rebalance the buffer according to desired proportions.
        """
        total_samples = len(self.buffer)
        target_counts = {
            category: int(total_samples * proportion)
            for category, proportion in desired_proportions.items()
        }

        # Collect samples for each category
        category_to_samples = defaultdict(list)
        for idx in range(len(self.buffer)):
            sample = self.buffer[idx]
            _, y = sample[0], sample[1]
            category = self.category_mapping[y]
            category_to_samples[category].append(sample)

        # Select samples based on target counts
        selected_samples = []
        for category, target_count in target_counts.items():
            samples = category_to_samples[category]
            selected_samples.extend(random.sample(samples, min(len(samples), target_count)))

        # Create a new AvalancheDataset with selected samples
        self.buffer = as_avalanche_dataset(selected_samples)
        
    def rebalance_buffer(self, desired_proportions: dict):
        """
        Rebalance the buffer according to desired proportions while enforcing max_size.
        If there are not enough samples for a class, upsample the available samples.

        :param desired_proportions: A dictionary mapping categories to their desired proportions.
        """
        # Calculate target counts for each category based on max_size
        target_counts = {
            category: int(self.max_size * proportion)
            for category, proportion in desired_proportions.items()
        }

        # Collect samples for each category
        category_to_samples = defaultdict(list)
        for idx in range(len(self.buffer)):
            sample = self.buffer[idx]
            _, y = sample[0], sample[1]
            category = self.category_mapping[y]
            category_to_samples[category].append(sample)

        # Select or upsample samples based on target counts
        selected_samples = []
        for category, target_count in target_counts.items():
            samples = category_to_samples[category]

            if len(samples) >= target_count:
                # If enough samples are available, randomly select the required number
                selected_samples.extend(random.sample(samples, target_count))
            else:
                # If not enough samples are available, upsample (duplicate) them
                selected_samples.extend(samples)
                remaining_count = target_count - len(samples)
                selected_samples.extend(random.choices(samples, k=remaining_count))

        # Shuffle the final buffer to mix different classes
        random.shuffle(selected_samples)

        # Create a new AvalancheDataset with selected samples
        self.buffer = as_avalanche_dataset(selected_samples)




class ProportionalReplayPlugin(SupervisedPlugin):
    """
    A replay plugin with a proportional replay buffer.

    :param mem_size: The total number of patterns to store in memory.
    :param category_mapping: A mapping from class indices to categories.
    :param proportion_mapping: A mapping from categories to desired proportions.
    :param batch_size: The size of the data batch. If None, it will be set to the strategy's batch size.
    :param batch_size_mem: The size of the memory batch. If None, it will be set to the data batch size.
    :param task_balanced_dataloader: If True, buffer data loaders will be task-balanced.
    """

    def __init__(
        self,
        mem_size: int = 50,
        category_mapping: dict = ACI_CATEGORY_MAPPING,
        proportion_mapping: dict = ACI_PROPORTION_MAPPING,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
    ):
        super().__init__()
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.storage_policy = ProportionalReplayBuffer(
            max_size=mem_size,
            category_mapping=category_mapping,
            proportion_mapping=proportion_mapping,
        )

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            return

        batch_size = self.batch_size or strategy.train_mb_size
        batch_size_mem = self.batch_size_mem or strategy.train_mb_size

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
