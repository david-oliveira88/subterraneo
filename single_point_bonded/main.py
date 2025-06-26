
import matplotlib.pyplot as plt
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import numpy as np

from core import SistemaDeCabos
from data_models import Coordenadas
from reporting import gerar_memorial_calculo

app = FastAPI(
    title="API para Cálculo de Tensão Induzida em Cabos Subterrâneos",
    description="Esta API permite a análise de tensões induzidas em sistemas de cabos subterrâneos, com base em parâmetros físicos e elétricos.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoordenadasInput(BaseModel):
    x: float
    y: float

class CaboInput(BaseModel):
    nome: str
    coord: CoordenadasInput
    cond_res_oh_m: float
    cond_gmr: float
    blin_res_oh_m: float
    blin_raio_m: float

class EccConfigInput(BaseModel):
    coord: CoordenadasInput

class SystemInput(BaseModel):
    comprimento_linha_km: float
    resistencia_malha_terra_ohm: float
    resistividade_solo_ohm_m: float
    frequencia_rede_hz: float
    correntes_nominais_amp: List[float]
    corrente_falta_3ph_amp: float
    corrente_falta_1ph_amp: float
    tempo_eliminacao_falta_s: float
    tipo_isolacao_ecc: str
    nbi_blindagem_kvp: float
    k_tov_rede: float
    cabos_config: List[CaboInput]
    eccs_config: List[EccConfigInput]

@app.post("/plot_layout/", summary="Gera um gráfico da disposição dos cabos e ECCs")
async def plot_layout(system_input: SystemInput) -> Dict[str, str]:
    try:
        # Extrair coordenadas dos cabos
        cable_coords_x = [c.coord.x for c in system_input.cabos_config]
        cable_coords_y = [c.coord.y for c in system_input.cabos_config]

        # Extrair coordenadas dos ECCs
        ecc_coords_x = [e.coord.x for e in system_input.eccs_config]
        ecc_coords_y = [e.coord.y for e in system_input.eccs_config]

        # Criar o gráfico
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(cable_coords_x, cable_coords_y, color='blue', marker='o', label='Cabos')
        ax.scatter(ecc_coords_x, ecc_coords_y, color='red', marker='s', label='ECCs')

        # Adicionar rótulos para os cabos
        for i, cable in enumerate(system_input.cabos_config):
            ax.text(cable.coord.x, cable.coord.y, f' {cable.nome}', fontsize=9)

        ax.set_xlabel("Coordenada X (m)")
        ax.set_ylabel("Coordenada Y (m)")
        ax.set_title("Disposição dos Cabos e ECCs")
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box') # Manter proporção para visualização correta

        # Salvar o gráfico em um buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig) # Fechar a figura para liberar memória

        # Codificar a imagem em Base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {"image_base64": image_base64}

    except Exception as e:
        logging.error(f"Erro ao gerar o gráfico de layout: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar o gráfico de layout: {str(e)}")

@app.post("/calculate/", 
          summary="Calcula as tensões induzidas e dimensiona os componentes do sistema",
          response_description="Retorna o memorial de cálculo em formato de texto.")
async def calculate(system_input: SystemInput) -> Dict[str, Any]:
    """
    Recebe os dados do sistema, realiza os cálculos e retorna o memorial.
    """
    try:
        cabos_config = [
            {
                'nome': c.nome,
                'coord': Coordenadas(x=c.coord.x, y=c.coord.y),
                'cond_res_oh_m': c.cond_res_oh_m,
                'cond_gmr': c.cond_gmr,
                'blin_res_oh_m': c.blin_res_oh_m,
                'blin_raio_m': c.blin_raio_m
            } for c in system_input.cabos_config
        ]

        eccs_dimensionados_info = []
        ecc_selecionados = []
        for _ in range(len(system_input.correntes_nominais_amp)):
            ecc, just, area = SistemaDeCabos.dimensionar_ecc(system_input.corrente_falta_1ph_amp, system_input.tempo_eliminacao_falta_s, system_input.tipo_isolacao_ecc)
            if not ecc: raise ValueError(f"Não foi possível dimensionar um ECC.")
            eccs_dimensionados_info.append({'resultado': (ecc, just), 'corrente_falta_a': system_input.corrente_falta_1ph_amp, 'tempo_s': system_input.tempo_eliminacao_falta_s, 'area_requerida_mm2': area})
            ecc_selecionados.append(ecc)

        eccs_config = list(zip(ecc_selecionados, [Coordenadas(x=c.coord.x, y=c.coord.y) for c in system_input.eccs_config]))

        sistema = SistemaDeCabos(
            cabos_config=cabos_config, eccs_config=eccs_config,
            comprimento_linha_km=system_input.comprimento_linha_km,
            resistencia_malha_terra_ohm=system_input.resistencia_malha_terra_ohm,
            resistividade_solo=system_input.resistividade_solo_ohm_m, 
            frequencia_hz=system_input.frequencia_rede_hz
        )
        
        analises: Dict[str, Dict] = {}
        max_tensao_falta_v = 0.0
        pior_caso_cenario = ""

        estados_operacionais = {
            "Ambos Circuitos ON": system_input.correntes_nominais_amp,
            "Apenas Circuito 1 ON": [system_input.correntes_nominais_amp[0], 0],
            "Apenas Circuito 2 ON": [0, system_input.correntes_nominais_amp[1]],
        }

        for nome_estado, correntes_op in estados_operacionais.items():
            tensoes_regime, correntes_ecc_regime = sistema.analisar_regime_permanente(correntes_op)
            analises[f'Regime Permanente ({nome_estado})'] = {'tensao_vetor_v': tensoes_regime, 'corrente_ecc': correntes_ecc_regime}

            for i in range(sistema.n_circuitos):
                if correntes_op[i] == 0: continue

                i_fase_pre_falta_3ph = np.zeros(sistema.n_cabos, dtype=complex)
                for c_idx in range(sistema.n_circuitos):
                    I = system_input.corrente_falta_3ph_amp if c_idx == i else correntes_op[c_idx]
                    if I != 0: i_fase_pre_falta_3ph[c_idx*3 : c_idx*3+3] = [I, I * np.exp(1j * 2 * np.pi / 3)**2, I * np.exp(1j * 2 * np.pi / 3)]

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
                            i_fase_pre_falta_1ph[c_idx*3:c_idx*3+3] = [correntes_op[c_idx], correntes_op[c_idx]*np.exp(1j * 2 * np.pi / 3)**2, correntes_op[c_idx]*np.exp(1j * 2 * np.pi / 3)]
                    i_fase_pre_falta_1ph[fase_global_idx] = system_input.corrente_falta_1ph_amp

                    nome_analise_1ph = f"Falta 1ph na Fase '{sistema.cabos[fase_global_idx].nome}' ({nome_estado})"
                    tensoes_1ph, ecc_1ph = sistema.analisar_falta(system_input.corrente_falta_1ph_amp, i_fase_pre_falta_1ph)
                    analises[nome_analise_1ph] = {'tensao_vetor_v': tensoes_1ph, 'corrente_ecc': ecc_1ph}
                    
                    tensao_max_atual_v = np.abs(tensoes_1ph).max()
                    if tensao_max_atual_v > max_tensao_falta_v:
                        max_tensao_falta_v = tensao_max_atual_v
                        pior_caso_cenario = nome_analise_1ph
        
        tensao_total_pior_caso_kv = max_tensao_falta_v / 1000
        svl_sel, just_svl = SistemaDeCabos.dimensionar_svl(tensao_total_pior_caso_kv, system_input.nbi_blindagem_kvp, system_input.k_tov_rede)

        sistema.calcular_matriz_impedancia_fases()
        sistema.calcular_matriz_impedancia_blindagens()
        sistema.calcular_matriz_impedancia_mutua_fase_blindagem()
        sistema.calcular_matrizes_impedancia_ecc()
        
        dados_memorial = {
            'sistema': sistema, 'eccs': eccs_dimensionados_info,
            'matrizes': sistema._matrizes_impedancia_lumped, 'analises': analises,
            'svl': {
                'cenario': f"{pior_caso_cenario} (Pior Caso)", 'tensao_max_falta_kv': tensao_total_pior_caso_kv,
                'nbi_blindagem_kvp': system_input.nbi_blindagem_kvp, 'k_tov_rede': system_input.k_tov_rede, 'resultado': (svl_sel, just_svl),
            }
        }
        
        output_filename = "memorial_de_calculo_api.txt"
        gerar_memorial_calculo(output_filename, dados_memorial)
        
        with open(output_filename, 'r', encoding='utf-8') as f:
            report_content = f.read()

        return {"memorial_de_calculo": report_content}

    except (ValueError, TypeError, NotImplementedError) as e:
        logging.error(f"Erro na requisição: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Erro inesperado no servidor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")
