"""droplet_pinn_studio: PINN for the normalized droplet d^2 ODE."""

from .studio import TRAIN_DEFAULTS, exact_solution, hard_ansatz, make_network, train_pin

__all__ = ["TRAIN_DEFAULTS", "exact_solution", "hard_ansatz", "make_network", "train_pin"]
__version__ = "1.0.0"
