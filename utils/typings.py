# Typing for the federated averaging algorithm
# Debashish Buragohain

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Any, Union

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrays = list[NDArray]

Scalar = Union[bool, bytes, float, int, str]

@dataclass
class Parameters:
    """Model parameters."""
    tensors: list[bytes]
    tensor_type: str
    
# fit response is not defined in this version of the code
@dataclass
class FitIns:
    """Fit instructions for a client."""
    parameters: Parameters
    num_examples: int               # the number of examples used for training
    config: dict[str, Scalar]       # stores the learning rate for that client