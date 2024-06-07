import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Função para extrair os dados do arquivo
def extrair_dados(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    parametros = re.findall(r"Parameters: (.+?)\n", data)
    custos = re.findall(r"Average Best Cost: (\d+\.\d+)", data)
    tempos = re.findall(r"Average Execution Time: (\d+\.\d+)", data)
    
    return parametros, list(map(float, custos)), list(map(float, tempos))

# Função para criar um DataFrame
def criar_dataframe(parametros, custos, tempos):
    rows = []
    for i, (param, custo, tempo) in enumerate(zip(parametros, custos, tempos)):
        param_dict = eval(param)
        param_dict['Average Best Cost'] = custo
        param_dict['Average Execution Time'] = tempo
        param_dict['Iteration'] = i + 1  # Adiciona a iteração
        rows.append(param_dict)
    
    df = pd.DataFrame(rows)
    
    # Cria uma coluna para combinar os valores dos parâmetros em uma única string
    df['Combined Params'] = df.apply(lambda row: f"ants: {row['num_ants']}, iter: {row['num_iterations']}, alpha: {row['alpha']}, beta: {row['beta']}, rho: {row['rho']}, Q: {row['Q']}", axis=1)
    
    return df

# Função para gerar o gráfico de custo por soluções usando matplotlib
def gerar_grafico_custo_solucoes(custos, file_name):
    solucoes = range(len(custos))
    custos = [float(custo) for custo in custos]
    
    plt.figure(figsize=(12, 8))
    plt.plot(solucoes, custos, label='Custo por Solução', color='brown')
    
    plt.xlabel('Soluções', fontsize=21)
    plt.ylabel('Custo', fontsize=21)
    plt.title('Custo por Solução', fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.legend(loc='upper right', fontsize='x-large', frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True)
    plt.xlim([0, len(custos)])
    
    plt.savefig(file_name)
    plt.close()

# Função para gerar gráficos de curva de influência dos parâmetros
def gerar_graficos_influencia_parametros(df, parametros_lista):
    for param, valores in parametros_lista.items():
        plt.figure(figsize=(8, 6))
        
        # Filtrando o DataFrame para os valores específicos do parâmetro
        df_filtered = df[df[param].isin(valores)]
        
        # Agrupando por valores únicos do parâmetro e calculando a média de custo
        df_mean = df_filtered.groupby(param)['Average Best Cost'].mean().reset_index()
        
        sns.lineplot(x=param, y='Average Best Cost', data=df_mean, color='blue', alpha=0.7, marker='o', markersize=8, linestyle='-', linewidth=2)
        
        plt.xlabel(param, fontsize=12)
        plt.ylabel('Average Best Cost', fontsize=12)
        plt.title(f'{param} vs Average Best Cost', fontsize=14)
        plt.xticks(valores, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(f'{param}_influence_curve.pdf')
        plt.close()

# Caminho do arquivo
file_name = 'resultados_ACO_GRID20v3.txt'

# Extração dos dados
parametros, custos, tempos = extrair_dados(file_name)

# Criação do DataFrame
df = criar_dataframe(parametros, custos, tempos)

# Lista de parâmetros para gráficos individuais
parametros_lista = {
    'num_ants': [10, 20, 30, 40, 50],
    'num_iterations': [25, 50, 75, 100, 125],
    'alpha': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    'beta': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    'rho': [0.10, 0.15, 0.20, 0.25, 0.30],
    'Q': [25, 50, 75, 100, 125]
}

# Geração do gráfico de custo por soluções
gerar_grafico_custo_solucoes(custos, 'custo_por_solucoes.pdf')

# Geração dos gráficos de curva de influência dos parâmetros
# gerar_graficos_influencia_parametros(df, parametros_lista)
