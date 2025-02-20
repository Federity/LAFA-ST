# the federated averaging strategy developed as a standalone module for Federity
# Debashish Buragohain

from typing import Tuple, List, Dict, Optional
from typings import Parameters, FitIns, Scalar
from utils.aggregate import aggregate, weighted_loss_avg
from utils.parameter import parameters_to_ndarrays, ndarrays_to_parameters

class FedAvg():
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        num_clients: int = 2,
    ) -> None: 
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.num_clients = num_clients
    
    # fits and returns the list of client instances and fit instrctions for every client
    def configure_fit(
            self, server_round: int, parameters: Parameters, num_examples: int) -> List[FitIns]:
        """Configure the next round of training."""

        # create custom configs
        half_clients = self.num_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configs = []
        # clients greater than the middle are assigned a higher learning rate   
        # this introduces diversity, accelerate convergance and adapt to heterogenous data
        for idx in range(self.num_clients):
            if idx < half_clients:
                fit_configs.append(FitIns(parameters, num_examples, standard_config))
            else:
                fit_configs.append(FitIns(parameters, num_examples, higher_lr_config))

        """
        Diversity of Updates:
        By assigning a higher learning rate to some clients, the updates from these clients will differ more significantly from those with a lower learning rate.
        This diversity can help the global model escape local minima or saddle points during training.
        Accelerating Learning for Some Clients:
        Clients with higher learning rates will make faster progress, which could be useful in scenarios where a mix of fast and slow learners improves overall convergence.
        Experimental Setup:
        The division into two groups (standard and higher learning rates) might be part of an experimental strategy to study how different learning rates affect the global model's performance.
        Heterogeneous Data:
        In federated learning, clients often have non-IID data distributions. Assigning higher learning rates to some clients might help the global model quickly adapt to specific data patterns.
        Improving Stability:
        Assigning lower learning rates to some clients provides stability, while higher learning rates introduce exploration. This balance between stability and exploration can improve training robustness.
        """
        return fit_configs
        
    # aggregates the parameters and gives the aggregation results
    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitIns],
    ) -> Tuple[Parameters, Optional[Dict[str, Scalar]]]:
        """ Aggregate fit results using weighted average."""
        # converts the models parameters received from each client into a format suitable for aggregation ndarrays and pairs them with the number of examples the client trained on
        # parameters is a serialized format and it needs to be converted into ndarrays
        weight_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)  
            for _, fit_res in results
        ]
        # performs a weighted aggregation of the model updates to create the new global model
        # aggregate() computes a weighted average of the model parameters from all clients. Clients that trained on more data indicated by num_examples have greater influence on the global model
        parameters_aggregated = ndarrays_to_parameters(aggregate(weight_results))
        # this is a placeholder for aggregated metrics e.g. average trainig loss or accuracy across clients
        # metrics is not generated in this version        
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated