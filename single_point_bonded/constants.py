
import numpy as np
from typing import Final, Dict

# --- Constantes de Base ---
PERMEABILIDADE_MAGNETICA: Final[float] = 4 * np.pi * 1e-7
J: Final[complex] = 1j
OPERADOR_ALPHA: Final[complex] = np.exp(J * 2 * np.pi / 3)

# Constante adiabática do Cobre (IEC 60949)
K_COBRE_ADIABATICO: Final[Dict[str, float]] = {
    "XLPE": 143.0, "EPR": 143.0, "PVC": 115.0,
}
