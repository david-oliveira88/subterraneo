"""
Módulo para cálculo de impedâncias e tensões induzidas em sistemas de 
cabos elétricos subterrâneos, com suporte a múltiplos circuitos, dimensionamento 
de SVL, ECC e geração de memorial de cálculo.

Versão final revisada conforme análise técnica detalhada, alinhada com as 
brochuras CIGRÉ TB-283/797 e IEEE 575.
"""
import numpy as np
import logging
from typing import Final, List, Tuple, Dict, Optional
from dataclasses import dataclass
import datetime

# --- Constantes de Base ---
PERMEABILIDADE_MAGNETICA: Final[float] = 4 * np.pi * 1e-7
J: Final[complex] = 1j
OPERADOR_ALPHA: Final[complex] = np.exp(J * 2 * np.pi / 3)

# Constante adiabática do Cobre (IEC 60949)
K_COBRE_ADIABATICO: Final[Dict[str, float]] = {
    "XLPE": 143.0, "EPR": 143.0, "PVC": 115.0,
}

# --- Definição de Classes de Dados ---
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
        # Conforme CIGRÉ, o termo resistivo da terra (R_E) não entra na impedância mútua.
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
        
        # Parâmetros Físicos do Sistema
        self.comprimento_linha_m = comprimento_linha_km * 1000
        self.resistencia_malha_terra_ohm = resistencia_malha_terra_ohm
        self.resistividade_solo = resistividade_solo
        self.frequencia_hz = frequencia_hz

        # Constantes Derivadas
        self.freq_angular = 2 * np.pi * self.frequencia_hz
        self.profundidade_retorno_terra = (1.85 / np.sqrt(self.freq_angular * PERMEABILIDADE_MAGNETICA / self.resistividade_solo))
        self.termo_resistivo_terra = self.freq_angular * PERMEABILIDADE_MAGNETICA / 8.0

        if considerar_capacitancia:
            raise NotImplementedError("O cálculo com capacitância ainda não foi implementado.")

        # Inicialização dos objetos
        self.cabos = [
            Cabo(c['nome'], c['coord'], 
                 Condutor(c['cond_gmr'], c['cond_res_oh_m'], self), 
                 Blindagem(c['blin_raio_m'], c['blin_res_oh_m'], self))
            for c in cabos_config
        ]
        self.eccs_info = [
            (ecc, coord) for ecc, coord in eccs_config
        ]

        # Validações
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
        for i in range(self.n_cabos): # Linha = Blindagem
            for j in range(self.n_cabos): # Coluna = Fase
                if i == j:
                    reatancia_coaxial = self.cabos[i].condutor._calcular_reatancia_mutua(self.cabos[i].blindagem.raio_medio)
                    # Z_sb_propria = R_s + R_E + jX_mutua(De/Rb)
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
                    # Impedância própria do ECC (Ohm/m) + Resistência de malha distribuída (Ohm/m)
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
        """Resolve o sistema Z.I = V para as correntes nos ECCs."""
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

def gerar_memorial_calculo(filename: str, dados: dict):
    logging.info(f"Gerando memorial de cálculo no arquivo: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        sistema = dados['sistema']
        def format_matriz(matriz, nome, unidades="Ohm"):
            linhas = [f"--- {nome} ({unidades}) ---"]
            for i, linha in enumerate(matriz):
                linha_formatada = f"  Linha {i+1}: ["
                for val in linha: linha_formatada += f"({val.real:9.4f} {val.imag:+.4f}j) "
                linhas.append(linha_formatada.strip() + " ]")
            return "\n".join(linhas) + "\n"

        def format_vetor_tensao(vetor, cabos):
            linhas = []
            v_por_m = vetor / sistema.comprimento_linha_m
            for i, _ in enumerate(vetor):
                mag_v_m, fase_v_m = np.abs(v_por_m[i]), np.angle(v_por_m[i], deg=True)
                mag_v, fase_v = np.abs(vetor[i]), np.angle(vetor[i], deg=True)
                linhas.append(f"  Blindagem '{cabos[i].nome}': {mag_v_m:8.4f} ∠ {fase_v_m:6.2f}° V/m  =>  {mag_v:8.2f} ∠ {fase_v:6.2f}° V")
            return "\n".join(linhas)
            
        def format_vetor_corrente(vetor, label="ECC"):
            linhas = []
            for i, val in enumerate(vetor):
                mag, fase = np.abs(val), np.angle(val, deg=True)
                linhas.append(f"  Corrente no {label} {i+1}: {mag:8.2f} ∠ {fase:6.2f}° A")
            return "\n".join(linhas)

        f.write("="*80 + "\nMEMORIAL DE CÁLCULO DE TENSÕES INDUZIDAS E DIMENSIONAMENTOS\n" + "="*80 + "\n")
        f.write(f"Data da Geração: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\nComprimento da Linha: {sistema.comprimento_linha_m / 1000} km\n\n")

        f.write("-" * 25 + " DADOS DE ENTRADA DO SISTEMA " + "-" * 26 + "\n")
        f.write(f"Frequência da Rede: {sistema.frequencia_hz} Hz\nResistividade do Solo: {sistema.resistividade_solo} Ohm.m\n")
        f.write(f"Resistência da Malha de Terra (extremidades): {sistema.resistencia_malha_terra_ohm} Ohm\n")
        f.write(f"Profundidade Equivalente de Retorno (De): {sistema.profundidade_retorno_terra:.2f} m\n\n")
        f.write("Configuração dos Cabos:\n")
        for i, cabo in enumerate(sistema.cabos):
            f.write(f"  {i+1}. {cabo.nome}: Coordenadas=(x={cabo.coordenadas.x}, y={cabo.coordenadas.y}) m\n")
        
        f.write("\n" + "-" * 27 + " DIMENSIONAMENTO DOS CABOS ECC " + "-" * 26 + "\n\n")
        for i, ecc_info in enumerate(dados['eccs']):
            f.write(f"ECC para Circuito {i+1}:\n")
            f.write(f"  Corrente de Falta: {ecc_info['corrente_falta_a']} A | Tempo: {ecc_info['tempo_s']} s\n")
            f.write(f"  Isolação considerada: {ecc_info['resultado'][0].tipo_isolacao}\n")
            f.write(f"  Seção Mínima Requerida: {ecc_info['area_requerida_mm2']:.2f} mm^2\n")
            f.write(f"  Cabo Selecionado: {ecc_info['resultado'][1]}\n\n")
        
        f.write("-" * 29 + " MATRIZES DE IMPEDÂNCIA (Valores Totais em Ohm)" + "-" * 20 + "\n\n")
        # CORREÇÃO: Chama os métodos de cálculo para popular o cache antes de acessar.
        matrizes = dados['sistema']._matrizes_impedancia_lumped
        f.write(format_matriz(matrizes['z_fase'], "Matriz de Impedância de Fase (Z_fase)"))
        f.write(format_matriz(matrizes['z_blindagem'], "Matriz de Impedância da Blindagem (Z_blindagem)"))
        f.write(format_matriz(matrizes['z_mutua_sf'], "Matriz de Impedância Mútua Fase-Blindagem (Z_sf)"))
        f.write(format_matriz(matrizes['z_ecc_ecc'], "Matriz de Impedância Própria e Mútua dos ECCs (Z_ecc_ecc)"))
        f.write(format_matriz(matrizes['z_ecc_fase'], "Matriz de Impedância Mútua ECC-Fase (Z_ecc_fase)"))

        f.write("\n" + "-" * 31 + " RESULTADOS DAS ANÁLISES " + "-" * 30 + "\n\n")
        for analise in sorted(dados['analises'].keys()):
            info = dados['analises'][analise]
            f.write(f"--- {analise} ---\n")
            f.write("Tensões Induzidas nas Blindagens (à terra local):\n" + format_vetor_tensao(info['tensao_vetor_v'], sistema.cabos) + "\n")
            f.write("Correntes nos Cabos ECC:\n" + format_vetor_corrente(info['corrente_ecc']) + "\n\n")
            
        f.write("-" * 29 + " DIMENSIONAMENTO DO SVL " + "-" * 30 + "\n\n")
        svl_info = dados['svl']
        f.write(f"Cenário Crítico: {svl_info['cenario']}\n")
        f.write(f"Tensão Máxima de Falta (TOV) na Linha: {svl_info['tensao_max_falta_kv']:.2f} kV\n")
        f.write(f"Fator de Sobretensão Temporária da Rede (k_tov): {svl_info['k_tov_rede']}\n")
        f.write(f"Nível de Impulso Suportável (NBI) da Blindagem: {svl_info['nbi_blindagem_kvp']:.2f} kVp\n\n")
        if svl_info['resultado'][0]:
            f.write(f"SVL Selecionado: {svl_info['resultado'][0].modelo}\n")
        f.write(f"Justificativa: {svl_info['resultado'][1]}\n")
        
        f.write("\n" + "="*80 + "\nFIM DO MEMORIAL\n" + "="*80 + "\n")
    logging.info("Memorial de cálculo gerado com sucesso.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # --- DADOS DE ENTRADA GERAIS ---
        COMPRIMENTO_LINHA_KM = 0.8
        RESISTENCIA_MALHA_TERRA_OHM = 0.5
        RESISTIVIDADE_SOLO_OHM_M = 100.0
        FREQUENCIA_REDE_HZ = 60.0
        CORRENTES_NOMINAIS_AMP = [340, 300] 
        CORRENTE_FALTA_3PH_AMP = 10000
        CORRENTE_FALTA_1PH_AMP = 8000
        TEMPO_ELIMINACAO_FALTA_S = 0.5
        TIPO_ISOLACAO_ECC = "XLPE"
        NBI_BLINDAGEM_KVP = 50.0
        K_TOV_REDE = 1.2 # Fator de sobretensão da rede

        # --- DADOS DE ENTRADA DOS CABOS ---
        cabos_config = [
            {'nome': "Fase 1A", 'coord': Coordenadas(x=-0.8, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
            {'nome': "Fase 1B", 'coord': Coordenadas(x=-0.5, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
            {'nome': "Fase 1C", 'coord': Coordenadas(x=-0.2, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
            {'nome': "Fase 2A", 'coord': Coordenadas(x=0.2, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
            {'nome': "Fase 2B", 'coord': Coordenadas(x=0.5, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
            {'nome': "Fase 2C", 'coord': Coordenadas(x=0.8, y=4.5), 'cond_res_oh_m': 0.00021, 'cond_gmr': 0.0060351, 'blin_res_oh_m': 0.000119866, 'blin_raio_m': 0.02389},
        ]

        # --- INÍCIO DOS CÁLCULOS ---
        
        eccs_dimensionados_info = []
        ecc_selecionados = []
        for _ in range(len(CORRENTES_NOMINAIS_AMP)):
            ecc, just, area = SistemaDeCabos.dimensionar_ecc(CORRENTE_FALTA_1PH_AMP, TEMPO_ELIMINACAO_FALTA_S, TIPO_ISOLACAO_ECC)
            if not ecc: raise ValueError(f"Não foi possível dimensionar um ECC.")
            eccs_dimensionados_info.append({'resultado': (ecc, just), 'corrente_falta_a': CORRENTE_FALTA_1PH_AMP, 'tempo_s': TEMPO_ELIMINACAO_FALTA_S, 'area_requerida_mm2': area})
            ecc_selecionados.append(ecc)

        eccs_config = list(zip(ecc_selecionados, [Coordenadas(x=-0.5, y=4.2), Coordenadas(x=0.5, y=4.2)]))

        sistema = SistemaDeCabos(
            cabos_config=cabos_config, eccs_config=eccs_config,
            comprimento_linha_km=COMPRIMENTO_LINHA_KM,
            resistencia_malha_terra_ohm=RESISTENCIA_MALHA_TERRA_OHM,
            resistividade_solo=RESISTIVIDADE_SOLO_OHM_M, frequencia_hz=FREQUENCIA_REDE_HZ
        )
        
        analises: Dict[str, Dict] = {}
        max_tensao_falta_v = 0.0
        pior_caso_cenario = ""

        estados_operacionais = {
            "Ambos Circuitos ON": CORRENTES_NOMINAIS_AMP,
            "Apenas Circuito 1 ON": [CORRENTES_NOMINAIS_AMP[0], 0],
            "Apenas Circuito 2 ON": [0, CORRENTES_NOMINAIS_AMP[1]],
        }

        for nome_estado, correntes_op in estados_operacionais.items():
            logging.info(f"--- ANALISANDO ESTADO OPERACIONAL: {nome_estado} ---")
            
            tensoes_regime, correntes_ecc_regime = sistema.analisar_regime_permanente(correntes_op)
            analises[f'Regime Permanente ({nome_estado})'] = {'tensao_vetor_v': tensoes_regime, 'corrente_ecc': correntes_ecc_regime}

            for i in range(sistema.n_circuitos):
                if correntes_op[i] == 0: continue

                i_fase_pre_falta_3ph = np.zeros(sistema.n_cabos, dtype=complex)
                for c_idx in range(sistema.n_circuitos):
                    I = CORRENTE_FALTA_3PH_AMP if c_idx == i else correntes_op[c_idx]
                    if I != 0: i_fase_pre_falta_3ph[c_idx*3 : c_idx*3+3] = [I, I * OPERADOR_ALPHA**2, I * OPERADOR_ALPHA]

                nome_analise_3ph = f'Falta 3ph no Ckt {i+1} ({nome_estado})'
                tensoes_3ph, ecc_3ph = sistema.analisar_falta(0, i_fase_pre_falta_3ph)
                analises[nome_analise_3ph] = {'tensao_vetor_v': tensoes_3ph, 'corrente_ecc': ecc_3ph}
                
                tensao_max_atual_v = np.abs(tensoes_3ph).max()
                if tensao_max_atual_v > max_tensao_falta_v:
                    max_tensao_falta_v = tensao_max_atual_v
                    pior_caso_cenario = nome_analise_3ph

                for fase_local_idx in range(3):
                    fase_global_idx = i * 3 + fase_local_idx
                    i_fase_pre_falta_1ph = np.zeros(sistema.n_cabos, dtype=complex)
                    for c_idx in range(sistema.n_circuitos):
                        if c_idx != i and correntes_op[c_idx] != 0:
                            i_fase_pre_falta_1ph[c_idx*3:c_idx*3+3] = [correntes_op[c_idx], correntes_op[c_idx]*OPERADOR_ALPHA**2, correntes_op[c_idx]*OPERADOR_ALPHA]
                    i_fase_pre_falta_1ph[fase_global_idx] = CORRENTE_FALTA_1PH_AMP

                    nome_analise_1ph = f"Falta 1ph na Fase '{sistema.cabos[fase_global_idx].nome}' ({nome_estado})"
                    tensoes_1ph, ecc_1ph = sistema.analisar_falta(CORRENTE_FALTA_1PH_AMP, i_fase_pre_falta_1ph)
                    analises[nome_analise_1ph] = {'tensao_vetor_v': tensoes_1ph, 'corrente_ecc': ecc_1ph}
                    
                    tensao_max_atual_v = np.abs(tensoes_1ph).max()
                    if tensao_max_atual_v > max_tensao_falta_v:
                        max_tensao_falta_v = tensao_max_atual_v
                        pior_caso_cenario = nome_analise_1ph
        
        tensao_total_pior_caso_kv = max_tensao_falta_v / 1000
        svl_sel, just_svl = SistemaDeCabos.dimensionar_svl(tensao_total_pior_caso_kv, NBI_BLINDAGEM_KVP, K_TOV_REDE)

        # CORREÇÃO: Garante que todas as matrizes sejam calculadas e cacheadas antes de gerar o memorial
        sistema.calcular_matriz_impedancia_fases()
        sistema.calcular_matriz_impedancia_blindagens()
        sistema.calcular_matriz_impedancia_mutua_fase_blindagem()
        sistema.calcular_matrizes_impedancia_ecc()
        
        dados_memorial = {
            'sistema': sistema, 'eccs': eccs_dimensionados_info,
            'matrizes': sistema._matrizes_impedancia_lumped, 'analises': analises,
            'svl': {
                'cenario': f"{pior_caso_cenario} (Pior Caso)", 'tensao_max_falta_kv': tensao_total_pior_caso_kv,
                'nbi_blindagem_kvp': NBI_BLINDAGEM_KVP, 'k_tov_rede': K_TOV_REDE, 'resultado': (svl_sel, just_svl),
            }
        }
        gerar_memorial_calculo("memorial_de_calculo_final.txt", dados_memorial)

    except (ValueError, TypeError, NotImplementedError) as e:
        logging.error(f"Ocorreu um erro durante a execução: {e}", exc_info=True)

    logging.info("--- Fim da Execução ---")
