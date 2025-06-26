
import numpy as np
import logging
import datetime

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
