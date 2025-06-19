# An attempt at TTA- Not currently in use
from avalanche.training.plugins import SupervisedPlugin
import torch
import torch.nn.functional as F

class TestTimeAdaptationPlugin(SupervisedPlugin):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.optimizer = None
        self.training_mode = None

    def before_eval_iteration(self, strategy, **kwargs):
        # Store original training mode
        self.training_mode = strategy.model.training
        # Set to training mode for TTA
        strategy.model.train()
        
        # Enable gradients for adaptation
        for param in strategy.model.parameters():
            param.requires_grad = True

        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(strategy.model.parameters(), lr=self.lr)

        # Perform one optimization step for entropy minimization
        x = strategy.mb_x
        with torch.set_grad_enabled(True):
            y_pred = strategy.model(x)
            entropy_loss = -torch.mean(torch.sum(F.softmax(y_pred, dim=1) * F.log_softmax(y_pred, dim=1), dim=1))
            
            self.optimizer.zero_grad()
            entropy_loss.backward()
            self.optimizer.step()

    def after_eval_iteration(self, strategy, **kwargs):
        # Restore original training mode
        strategy.model.train(self.training_mode)
        
        # Disable gradients after adaptation
        for param in strategy.model.parameters():
            param.requires_grad = False
