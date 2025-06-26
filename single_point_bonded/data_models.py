
from dataclasses import dataclass
from typing import List, Final
import numpy as np

@dataclass(frozen=True)
class SVL:
    """Representa um modelo de Dispositivo de Proteção contra Surtos."""
    modelo: str
    tensao_nominal_ur_kv: float
    tensao_residual_ures_10kA_kvp: float

@dataclass(frozen=True)
class CaboECC:
    """Representa um modelo de Cabo de Continuidade de Terra (ECC)."""
    secao_mm2: int
    resistencia_ohm_km: float
    raio_gmr_m: float
    tipo_isolacao: str

@dataclass(frozen=True)
class Coordenadas:
    """Representa um ponto imutável no plano cartesiano (x, y)."""
    x: float
    y: float
    def distancia_ate(self, outro: 'Coordenadas') -> float:
        return np.sqrt((self.x - outro.x)**2 + (self.y - outro.y)**2)

# "Banco de dados" de dispositivos e cabos disponíveis.
SVL_DISPONIVEIS: Final[List[SVL]] = [
    SVL(modelo="SVL-3kV", tensao_nominal_ur_kv=3.0, tensao_residual_ures_10kA_kvp=11.5),
    SVL(modelo="SVL-6kV", tensao_nominal_ur_kv=6.0, tensao_residual_ures_10kA_kvp=18.5),
    SVL(modelo="SVL-12kV", tensao_nominal_ur_kv=12.0, tensao_residual_ures_10kA_kvp=36.0),
]
ECC_DISPONIVEIS: Final[List[CaboECC]] = [
    CaboECC(secao_mm2=70, resistencia_ohm_km=0.268, raio_gmr_m=0.0041, tipo_isolacao="XLPE"),
    CaboECC(secao_mm2=120, resistencia_ohm_km=0.153, raio_gmr_m=0.0054, tipo_isolacao="XLPE"),
    CaboECC(secao_mm2=240, resistencia_ohm_km=0.0754, raio_gmr_m=0.0075, tipo_isolacao="XLPE"),
]
