import re
import pandas as pd

# Função para extrair os dados do arquivo
def extrair_dados(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    parametros = re.findall(r"Parameters: (.+?)\n", data)
    custos = re.findall(r"Average Best Cost: (\d+\.\d+)", data)
    tempos = re.findall(r"Average Execution Time: (\d+\.\d+)", data)
    
    # Calcular a média de todos os custos
    custos_float = [float(custo) for custo in custos]
    media_custo = sum(custos_float) / len(custos_float)
    
    return parametros, custos, tempos, media_custo

# Função para criar um DataFrame
def criar_dataframe(parametros, custos, tempos):
    rows = []
    for param, custo, tempo in zip(parametros, custos, tempos):
        param_dict = eval(param)
        param_dict['Average Best Cost'] = float(custo)
        param_dict['Average Execution Time'] = float(tempo)
        rows.append(param_dict)
    
    df = pd.DataFrame(rows)
    return df

# Função para calcular a média de custo e tempo de execução para cada valor de parâmetro isolado
def calcular_medias_parametros(df, parametros):
    medias = {}
    for parametro in parametros:
        medias[parametro] = df.groupby(parametro)[['Average Best Cost', 'Average Execution Time']].mean()
    return medias

# Caminho do arquivo
file_name = 'resultados_ACO_GRID20v3.txt'

# Extração dos dados
parametros, custos, tempos, media_custo = extrair_dados(file_name)

# Criação do DataFrame
df = criar_dataframe(parametros, custos, tempos)

# Lista de parâmetros
parametros_lista = ['num_ants', 'num_iterations', 'alpha', 'beta', 'rho', 'Q']

# Cálculo das médias para cada parâmetro
medias_parametros = calcular_medias_parametros(df, parametros_lista)

# Impressão dos resultados
print(f"Média de todos os custos: {media_custo}\n")

for parametro, medias in medias_parametros.items():
    print(f"Médias para o parâmetro '{parametro}':")
    print(medias)
    print("\n")
