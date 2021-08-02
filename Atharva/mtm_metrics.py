"""Implements metrics for multi-task setting."""
from torchflare.metrics import MetricMeter, _BaseMetric
import sklearn.metrics as skm
import torch

# Implementation F1-score customized for TorchFlare
class SklearnF1(MetricMeter, _BaseMetric):
    def __init__(self, target_name: str):
        super(SklearnF1, self).__init__(multilabel=False)
        self.f1 = skm.f1_score
        self._outputs = None
        self._targets = None
        self.target_name = target_name
        self.reset()

    def handle(self):
        return f"{self.target_name}_{self.f1.__name__.lower()}"

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to accumulate the outputs and targets.
        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : Ground truth targets
        """
        outputs, targets = self.detach_tensor(outputs[self.target_name]), self.detach_tensor(targets[self.target_name])
        outputs = torch.argmax(outputs, dim=1)
        self._outputs.append(outputs)
        self._targets.append(targets)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []

    @property
    def value(self):
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)
        f1_score = self.f1(targets.numpy(), outputs.numpy())
        return torch.tensor(f1_score)

# Implementation MSE customized for TorchFlare
class SklearnMSE(MetricMeter, _BaseMetric):
    def __init__(self, target_name: str):
        super(SklearnMSE, self).__init__(multilabel=False)
        self.mse = partial(skm.mean_squared_error, squared=False)
        self._outputs = None
        self._targets = None
        self.target_name = target_name
        self.reset()

    def handle(self):
        return f"{self.target_name}_{self.mse.func.__name__.lower()}"

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to accumulate the outputs and targets.
        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : Ground truth targets
        """
        outputs, targets = self.detach_tensor(outputs[self.target_name]), self.detach_tensor(targets[self.target_name])
        self._outputs.append(outputs)
        self._targets.append(targets)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []

    @property
    def value(self):
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)
        mse_score = self.mse(targets.numpy(), outputs.numpy())
        return torch.tensor(mse_score)



