import numpy as np
import matplotlib.pyplot as plt

# Função para ler resultados do arquivo
def read_results(file_name):
    results = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("Total Cost of Best Solution:"):
                total_cost = float(line.split(": ")[1])
                tour = eval(lines[i-1].split(": ")[1])
                results.append((tour, total_cost))
    return results

# Ler arquivos de resultados
file_names = ['resultados_greedy10.txt', 'resultados_greedy15.txt', 'resultados_greedy20.txt']
results_dict = {file_name: read_results(file_name) for file_name in file_names}

# # Função para plotar gráficos para um arquivo de resultados
# def plot_results(file_name, results):
#     iterations = list(range(1, len(results) + 1))
#     total_costs = [result[1] for result in results]

#     # Configurações do gráfico
#     plt.figure(figsize=(12, 8))
#     plt.plot(iterations, total_costs, label='Custo Total da Solução', linestyle='-', color='purple')
#     plt.xlabel('Iterações')
#     plt.ylabel('Custo Total')
#     plt.title(f'Melhor Solução ao Longo das Iterações ({file_name})')
#     plt.legend()
#     plt.grid(True)

#     # Ajustando o intervalo do eixo Y para ser igual em todos os gráficos
#     plt.ylim(min(total_costs) - 50, max(total_costs) + 50)

#     # Salvando o gráfico em PDF
#     plt.savefig(f'melhor_solucao_iteracoes_{file_name}.pdf')

#     # Exibindo o gráfico
#     plt.show()
#     plt.close()

# # Plotar gráficos para cada arquivo de resultados
# for file_name, results in results_dict.items():
#     plot_results(file_name, results)


x = np.linspace(1, 10, 10, dtype=int)
y1 = np.repeat(2264, 10)
y2 = np.repeat(3544.4, 10)
y3 = np.repeat(3755.6, 10)

fig, (ax1) = plt.subplots(1,1, figsize=(4,3))
ax1.plot(x, y1, label='10 vértices')
ax1.plot(x, y2, label='15 vértices')
ax1.plot(x, y3, label='20 vértices')
ax1.set_xlabel('Iterações')
ax1.set_ylabel('Custo')
ax1.grid()
ax1.legend()
#plt.show() #imprimir na tela
plt.savefig('image.pdf', bbox_inches='tight') #salvar a figura em pdf
plt.show()