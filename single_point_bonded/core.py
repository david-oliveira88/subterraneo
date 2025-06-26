
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

from constants import J, PERMEABILIDADE_MAGNETICA, OPERADOR_ALPHA, K_COBRE_ADIABATICO
from data_models import Coordenadas, SVL, CaboECC, SVL_DISPONIVEIS, ECC_DISPONIVEIS

class CondutorBase:
    """Classe base para condutores com propriedades de impedância."""
    def __init__(self, resistencia_ohm_m: float, sistema: 'SistemaDeCabos'):
        self.resistencia = resistencia_ohm_m
        self.sistema = sistema

    def _calcular_reatancia_mutua(self, dist: float) -> complex:
        """Calcula apenas a parte reativa da impedância mútua."""
        if np.isclose(dist, 0):
            raise ValueError("A distância entre condutores para cálculo de reatância mútua não pode ser zero.")
        return J * (self.sistema.freq_angular * PERMEABILIDADE_MAGNETICA / (2 * np.pi)) * np.log(self.sistema.profundidade_retorno_terra / dist)

    def _calcular_impedancia_mutua_com_terra(self, coord1: Coordenadas, coord2: Coordenadas) -> complex:
        """Calcula a impedância mútua entre dois condutores via retorno por terra."""
        dist = coord1.distancia_ate(coord2)
        return self._calcular_reatancia_mutua(dist)

class Blindagem(CondutorBase):
    def __init__(self, raio_medio_m: float, resistencia_ohm_m: float, sistema: 'SistemaDeCabos'):
        super().__init__(resistencia_ohm_m, sistema)
        self.raio_medio = raio_medio_m
    
    def calcular_impedancia_propria(self) -> complex:
        reatancia = self._calcular_reatancia_mutua(self.raio_medio)
        return self.resistencia + self.sistema.termo_resistivo_terra + reatancia

class Condutor(CondutorBase):
    def __init__(self, raio_gmr_m: float, resistencia_ohm_m: float, sistema: 'SistemaDeCabos'):
        super().__init__(resistencia_ohm_m, sistema)
        self.raio_gmr = raio_gmr_m

    def calcular_impedancia_propria(self) -> complex:
        reatancia = self._calcular_reatancia_mutua(self.raio_gmr)
        return self.resistencia + self.sistema.termo_resistivo_terra + reatancia

class Cabo:
    def __init__(self, nome: str, coordenadas: Coordenadas, condutor: Condutor, blindagem: Blindagem):
        self.nome, self.coordenadas, self.condutor, self.blindagem = nome, coordenadas, condutor, blindagem

class SistemaDeCabos:
    """Gerencia múltiplos circuitos de cabos e calcula impedâncias e tensões induzidas."""
    def __init__(self, *, cabos_config: list, eccs_config: list, comprimento_linha_km: float, 
                 resistencia_malha_terra_ohm: float, resistividade_solo: float, frequencia_hz: float, 
                 considerar_capacitancia: bool = False):
        
        self.comprimento_linha_m = comprimento_linha_km * 1000
        self.resistencia_malha_terra_ohm = resistencia_malha_terra_ohm
        self.resistividade_solo = resistividade_solo
        self.frequencia_hz = frequencia_hz
        self.freq_angular = 2 * np.pi * self.frequencia_hz
        self.profundidade_retorno_terra = (1.85 / np.sqrt(self.freq_angular * PERMEABILIDADE_MAGNETICA / self.resistividade_solo))
        self.termo_resistivo_terra = self.freq_angular * PERMEABILIDADE_MAGNETICA / 8.0

        if considerar_capacitancia:
            raise NotImplementedError("O cálculo com capacitância ainda não foi implementado.")

        self.cabos = [
            Cabo(c['nome'], c['coord'], 
                 Condutor(c['cond_gmr'], c['cond_res_oh_m'], self), 
                 Blindagem(c['blin_raio_m'], c['blin_res_oh_m'], self))
            for c in cabos_config
        ]
        self.eccs_info = [
            (ecc, coord) for ecc, coord in eccs_config
        ]

        self.n_cabos = len(self.cabos)
        self.n_circuitos = self.n_cabos // 3
        if not self.cabos or self.n_cabos % 3 != 0:
            raise ValueError("A lista de cabos deve conter um número de cabos múltiplo de 3.")
        if len(self.eccs_info) != self.n_circuitos:
            raise ValueError("Deve haver um ECC para cada circuito.")

        self._matrizes_impedancia_lumped: Dict[str, np.ndarray] = {}
        logging.info(f"Sistema inicializado: {self.n_circuitos} circuito(s), {len(self.eccs_info)} ECC(s), L={comprimento_linha_km}km, Rho={resistividade_solo}Ohm.m")
    
    def _limpar_cache(self):
        self._matrizes_impedancia_lumped.clear()
        logging.info("Cache de matrizes de impedância limpo.")

    def _get_matriz_lumped(self, nome_matriz: str, funcao_calculo_por_metro, *args) -> np.ndarray:
        if nome_matriz not in self._matrizes_impedancia_lumped:
            z_por_metro = funcao_calculo_por_metro(*args)
            self._matrizes_impedancia_lumped[nome_matriz] = z_por_metro * self.comprimento_linha_m
        return self._matrizes_impedancia_lumped[nome_matriz]

    def _calcular_matriz_impedancia_fases_por_metro(self) -> np.ndarray:
        z = np.zeros((self.n_cabos, self.n_cabos), dtype=complex)
        for i in range(self.n_cabos):
            for j in range(self.n_cabos):
                z[i, j] = self.cabos[i].condutor.calcular_impedancia_propria() if i == j else self.cabos[i].condutor._calcular_impedancia_mutua_com_terra(self.cabos[i].coordenadas, self.cabos[j].coordenadas)
        return z
    
    def calcular_matriz_impedancia_fases(self) -> np.ndarray: 
        return self._get_matriz_lumped('z_fase', self._calcular_matriz_impedancia_fases_por_metro)

    def _calcular_matriz_impedancia_blindagens_por_metro(self) -> np.ndarray:
        z = np.zeros((self.n_cabos, self.n_cabos), dtype=complex)
        for i in range(self.n_cabos):
            for j in range(self.n_cabos):
                z[i, j] = self.cabos[i].blindagem.calcular_impedancia_propria() if i == j else self.cabos[i].blindagem._calcular_impedancia_mutua_com_terra(self.cabos[i].coordenadas, self.cabos[j].coordenadas)
        return z

    def calcular_matriz_impedancia_blindagens(self) -> np.ndarray: 
        return self._get_matriz_lumped('z_blindagem', self._calcular_matriz_impedancia_blindagens_por_metro)

    def _calcular_matriz_impedancia_mutua_fase_blindagem_por_metro(self) -> np.ndarray:
        z = np.zeros((self.n_cabos, self.n_cabos), dtype=complex)
        for i in range(self.n_cabos):
            for j in range(self.n_cabos):
                if i == j:
                    reatancia_coaxial = self.cabos[i].condutor._calcular_reatancia_mutua(self.cabos[i].blindagem.raio_medio)
                    z[i, j] = self.cabos[i].blindagem.resistencia + self.termo_resistivo_terra + reatancia_coaxial
                else:
                    z[i, j] = self.cabos[i].condutor._calcular_impedancia_mutua_com_terra(self.cabos[i].coordenadas, self.cabos[j].coordenadas)
        return z
    
    def calcular_matriz_impedancia_mutua_fase_blindagem(self) -> np.ndarray: 
        return self._get_matriz_lumped('z_mutua_sf', self._calcular_matriz_impedancia_mutua_fase_blindagem_por_metro)

    def _calcular_matrizes_ecc_por_metro(self) -> Dict[str, np.ndarray]:
        n_ecc = len(self.eccs_info)
        z_ecc_ecc_pm = np.zeros((n_ecc, n_ecc), dtype=complex)
        for i in range(n_ecc):
            for j in range(n_ecc):
                res_i = self.eccs_info[i][0].resistencia_ohm_km / 1000
                gmr_i = self.eccs_info[i][0].raio_gmr_m
                ecc_condutor = Condutor(gmr_i, res_i, self)
                if i == j:
                    z_ecc_ecc_pm[i, j] = ecc_condutor.calcular_impedancia_propria() + self.resistencia_malha_terra_ohm / self.comprimento_linha_m
                else:
                    z_ecc_ecc_pm[i, j] = ecc_condutor._calcular_impedancia_mutua_com_terra(self.eccs_info[i][1], self.eccs_info[j][1])
        
        z_ecc_fase_pm = np.zeros((n_ecc, self.n_cabos), dtype=complex)
        for i in range(n_ecc):
            res_i = self.eccs_info[i][0].resistencia_ohm_km / 1000
            gmr_i = self.eccs_info[i][0].raio_gmr_m
            ecc_condutor = Condutor(gmr_i, res_i, self)
            for j in range(self.n_cabos):
                z_ecc_fase_pm[i, j] = ecc_condutor._calcular_impedancia_mutua_com_terra(self.eccs_info[i][1], self.cabos[j].coordenadas)
        
        return {'z_ecc_ecc': z_ecc_ecc_pm, 'z_ecc_fase': z_ecc_fase_pm}

    def calcular_matrizes_impedancia_ecc(self) -> Dict[str, np.ndarray]:
        if 'z_ecc_ecc' not in self._matrizes_impedancia_lumped:
            matrizes_pm = self._calcular_matrizes_ecc_por_metro()
            self._matrizes_impedancia_lumped['z_ecc_ecc'] = matrizes_pm['z_ecc_ecc'] * self.comprimento_linha_m
            self._matrizes_impedancia_lumped['z_ecc_fase'] = matrizes_pm['z_ecc_fase'] * self.comprimento_linha_m
            self._matrizes_impedancia_lumped['z_ecc_blindagem'] = self._matrizes_impedancia_lumped['z_ecc_fase']
        
        return {
            'z_ecc_ecc': self._matrizes_impedancia_lumped['z_ecc_ecc'],
            'z_ecc_fase': self._matrizes_impedancia_lumped['z_ecc_fase'],
            'z_ecc_blindagem': self._matrizes_impedancia_lumped['z_ecc_blindagem']
        }

    def _calcular_correntes_ecc(self, v_fonte_nos_eccs: np.ndarray) -> np.ndarray:
        z_ecc_ecc = self.calcular_matrizes_impedancia_ecc()['z_ecc_ecc']
        try:
            return np.linalg.solve(z_ecc_ecc, v_fonte_nos_eccs)
        except np.linalg.LinAlgError:
            logging.warning("Matriz Z_ecc_ecc singular, usando pseudo-inversa.")
            return np.linalg.pinv(z_ecc_ecc) @ v_fonte_nos_eccs

    def analisar_regime_permanente(self, correntes_op: List[float]):
        correntes_fase = np.zeros(self.n_cabos, dtype=complex)
        for i in range(self.n_circuitos):
            if correntes_op[i] != 0:
                correntes_fase[i*3 : i*3+3] = [correntes_op[i], correntes_op[i] * OPERADOR_ALPHA**2, correntes_op[i] * OPERADOR_ALPHA]
        
        z_sf = self.calcular_matriz_impedancia_mutua_fase_blindagem()
        v_induzida_remota = z_sf @ correntes_fase

        z_ef = self.calcular_matrizes_impedancia_ecc()['z_ecc_fase']
        v_induzida_nos_eccs = - (z_ef @ correntes_fase)
        correntes_ecc = self._calcular_correntes_ecc(v_induzida_nos_eccs)

        z_be = self.calcular_matrizes_impedancia_ecc()['z_ecc_blindagem'].T
        v_total_local = (v_induzida_remota + z_be @ correntes_ecc)
        
        return v_total_local, correntes_ecc

    def analisar_falta(self, i_falta: float, i_fase_pre_falta: np.ndarray):
        z_sf = self.calcular_matriz_impedancia_mutua_fase_blindagem()
        v_induzida_remota = z_sf @ i_fase_pre_falta

        z_ef = self.calcular_matrizes_impedancia_ecc()['z_ecc_fase']
        v_induzida_nos_eccs = -(z_ef @ i_fase_pre_falta)
        correntes_ecc = self._calcular_correntes_ecc(v_induzida_nos_eccs)
        
        if not np.isclose(i_falta, 0):
             corrente_retorno_calculada = np.sum(correntes_ecc)
             corrente_retorno_esperada = -i_falta
             corrente_erro = corrente_retorno_esperada - corrente_retorno_calculada
             
             z_ecc_ecc = self.calcular_matrizes_impedancia_ecc()['z_ecc_ecc']
             try:
                 y_ecc_ecc = np.linalg.inv(z_ecc_ecc)
             except np.linalg.LinAlgError:
                 y_ecc_ecc = np.linalg.pinv(z_ecc_ecc)
             
             admitancias_nodais = np.sum(y_ecc_ecc, axis=1)
             soma_admitancias = np.sum(admitancias_nodais)
             
             if not np.isclose(soma_admitancias, 0):
                 fatores_dist = admitancias_nodais / soma_admitancias
                 correntes_ecc += fatores_dist * corrente_erro

        z_be = self.calcular_matrizes_impedancia_ecc()['z_ecc_blindagem'].T
        v_total_local = (v_induzida_remota + z_be @ correntes_ecc)
        return v_total_local, correntes_ecc
    
    @staticmethod
    def dimensionar_svl(tensao_max_falta_kv: float, nbi_blindagem_kvp: float, k_tov_rede: float = 1.0, 
                        margem_tov: float = 1.15, margem_nbi: float = 1.15) -> Tuple[Optional[SVL], str]:
        tensao_necessaria_ur_kv = tensao_max_falta_kv * k_tov_rede * margem_tov
        tensao_maxima_ures_kvp = nbi_blindagem_kvp / margem_nbi
        
        candidatos = [svl for svl in SVL_DISPONIVEIS if svl.tensao_nominal_ur_kv >= tensao_necessaria_ur_kv and svl.tensao_residual_ures_10kA_kvp <= tensao_maxima_ures_kvp]
        if not candidatos:
            return None, "Nenhum SVL disponível atende aos critérios."
        
        svl_selecionado = min(candidatos, key=lambda s: s.tensao_nominal_ur_kv)
        return svl_selecionado, f"SVL '{svl_selecionado.modelo}' atende aos critérios."

    @staticmethod
    def dimensionar_ecc(corrente_falta_amp: float, tempo_s: float, tipo_isolacao: str) -> Tuple[Optional[CaboECC], str, float]:
        k_const = K_COBRE_ADIABATICO.get(tipo_isolacao)
        if not k_const:
            raise ValueError(f"Tipo de isolação '{tipo_isolacao}' inválido para ECC.")
            
        area_req = (corrente_falta_amp * np.sqrt(tempo_s)) / k_const
        candidatos = [ecc for ecc in ECC_DISPONIVEIS if ecc.secao_mm2 >= area_req]
        if not candidatos:
            return None, "Nenhum cabo ECC disponível atende à seção requerida.", area_req
        
        ecc_selecionado = min(candidatos, key=lambda e: e.secao_mm2)
        justificativa = f"Cabo ECC de {ecc_selecionado.secao_mm2}mm^2 ({ecc_selecionado.tipo_isolacao}) selecionado."
        return ecc_selecionado, justificativa, area_req
