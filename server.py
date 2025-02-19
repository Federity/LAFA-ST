import numpy as np
from utils.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from utils.typing import NDArray, NDArrays

class LAFASTServerNumpy:
    def __init__(self, global_params: NDArrays, alpha: float = 0.9, eta: float = 0.1, epsilon: float = 1e-8):
        """
        Args:
            global_params: Global model parameters represented as a list of numpy arrays.
            alpha: Stability decay rate.
            eta: Global learning rate.
            epsilon: Small constant to avoid division by zero.
        """
        self.global_params: NDArrays = global_params
        self.stability: NDArrays = [np.ones_like(param) for param in global_params]
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

    def apply_update(self, client_params: NDArrays, client_delta: NDArrays) -> None:
        """
        Apply a client's update to the global parameters.

        Args:
            client_params: A list of numpy arrays (one per layer) from the client.
            client_delta: A list of numpy arrays representing the client’s update magnitude.
        """
        for i in range(len(self.stability)):
            self.stability[i] = self.alpha * self.stability[i] + (1 - self.alpha) * client_delta[i]
        
        for i in range(len(self.global_params)):
            diff = client_params[i] - self.global_params[i]
            delta = diff / (self.stability[i] + self.epsilon)
            self.global_params[i] += self.eta * delta
        
        noise_std = 0.01
        for i in range(len(self.stability)):
            self.stability[i] += np.random.rand(*self.stability[i].shape) * noise_std

    def get_global_params(self) -> NDArrays:
        """Return the current global model parameters as a list of numpy arrays."""
        return self.global_params