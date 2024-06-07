import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Função objetivo
def objective_function(solution, cost_matrix, prize_vector, penalty_vector, alpha, max_penalty, min_prize):
    total_cost = sum(cost_matrix[solution[i]][solution[i+1]] for i in range(len(solution) - 1))
    total_cost += cost_matrix[solution[-1]][solution[0]]  # custo de retorno ao início

    # Cálculo do valor coletado
    collected_value = sum(prize_vector[i] for i in solution)

    # Penalidade por não atingir o valor mínimo esperado
    missing_prize_penalty = max(0, min_prize - collected_value)
    
    # Total objective
    total_objective = total_cost + alpha * (max_penalty - collected_value) + alpha * missing_prize_penalty
    return total_objective

# Algoritmo ACO
def ant_colony_optimization(cost_matrix, prize_vector, penalty_vector, num_ants, num_iterations, alpha, beta, rho, Q, min_prize):
    num_nodes = len(cost_matrix)
    pheromone = np.ones((num_nodes, num_nodes))  # matriz de feromônio inicial
    best_solution = None
    best_cost = float('inf')
    
    best_solution_costs = []
    prizes_collected = []
    penalties = []
    average_colony_costs = []
    worst_solution_costs = []

    start_time = time.time()
    
    for iteration in range(num_iterations):
        solutions = []
        costs = []
        iteration_prizes = []
        iteration_penalties = []

        for ant in range(num_ants):
            current_node = 0  # seleciona o nó inicial fixo (nó 0)
            solution = [current_node]
            visited = set(solution)
            cost = 0

            while len(solution) < num_nodes:
                # Calcula as probabilidades de transição
                unvisited_nodes = [node for node in range(num_nodes) if node not in visited]
                probabilities = np.zeros(len(unvisited_nodes))
                for idx, node in enumerate(unvisited_nodes):
                    # Introduzir um fator de ruído aleatório
                    noise = random.uniform(0.9, 1.1)
                    probabilities[idx] = (pheromone[current_node][node] ** alpha) * ((1 / max(1, cost_matrix[current_node][node])) ** beta) * noise
                probabilities /= probabilities.sum()
                
                # Seleciona o próximo nó com base nas probabilidades calculadas
                next_node = np.random.choice(unvisited_nodes, p=probabilities)
                solution.append(next_node)
                visited.add(next_node)
                cost += cost_matrix[current_node][next_node]
                current_node = next_node

            # Calcula o custo total da solução
            cost += cost_matrix[solution[-1]][solution[0]]  # custo de retorno ao nó inicial
            solutions.append(solution)
            costs.append(cost)

            total_prize = sum(prize_vector[i] * int(i in solution) for i in range(len(prize_vector)))
            total_penalty = sum(penalty_vector[i] * (1 - int(i in solution)) for i in range(len(penalty_vector)))
            missing_prize_penalty = max(0, min_prize - total_prize)
            total_penalty += alpha * missing_prize_penalty
            
            iteration_prizes.append(total_prize)
            iteration_penalties.append(total_penalty)

            # Atualiza a melhor solução encontrada até agora
            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        best_solution_costs.append(best_cost)
        prizes_collected.append(max(iteration_prizes))
        penalties.append(max(iteration_penalties))
        average_colony_costs.append(np.mean(costs))
        worst_solution_costs.append(max(costs))

        # Atualiza a matriz de feromônio
        pheromone *= (1 - rho)
        for solution, cost in zip(solutions, costs):
            for i in range(len(solution) - 1):
                pheromone[solution[i]][solution[i + 1]] += Q / cost
            # Atualiza o feromônio também para o caminho de retorno ao nó inicial
            pheromone[solution[-1]][solution[0]] += Q / cost
    
    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_cost, execution_time, best_solution_costs, prizes_collected, penalties, average_colony_costs, worst_solution_costs

# Função para ler dados do arquivo
def read_data(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        cost_matrix_start = False
        prize_vector_start = False
        penalty_vector_start = False
        
        cost_matrix = []
        prize_vector = []
        penalty_vector = []
        
        for line in lines:
            if 'Cost Matrix:' in line:
                cost_matrix_start = True
                continue
            elif line.strip() == '':
                cost_matrix_start = False
                prize_vector_start = False
                penalty_vector_start = False
                continue
            elif 'Prize Vector:' in line:
                cost_matrix_start = False
                prize_vector_start = True
                continue
            elif 'Penalty Vector:' in line:
                prize_vector_start = False
                penalty_vector_start = True
                continue
            
            if cost_matrix_start and line.strip():
                cost_matrix.append(list(map(int, line.strip().split())))
            elif prize_vector_start and line.strip():
                prize_vector.extend(map(int, line.strip().split()))
            elif penalty_vector_start and line.strip():
                penalty_vector.extend(map(int, line.strip().split()))
                
    return prize_vector, penalty_vector, cost_matrix

# Parâmetros do ACO
num_ants = 20
num_iterations = 125
alpha = 0.9
beta = 1.3
rho = 0.15
Q = 25

# Nome do arquivo de entrada
file_name = 'data20.txt'

# Nome do arquivo de resultados
results_file = 'best_ACO_GRID20.txt'

# Lê os dados do arquivo
prize_vector, penalty_vector, cost_matrix = read_data(file_name)

# Executa o algoritmo ACO 10 vezes e armazena os resultados
results = []
for i in range(10):
    best_solution, best_cost, execution_time, best_solution_costs, prizes_collected, penalties, average_colony_costs, worst_solution_costs = ant_colony_optimization(
        cost_matrix, prize_vector, penalty_vector,
        num_ants, num_iterations, alpha, beta, rho, Q,
        min_prize=int(0.75 * sum(prize_vector))
    )
    results.append((best_solution, best_cost, execution_time, best_solution_costs, prizes_collected, penalties, average_colony_costs, worst_solution_costs))

# Calcula a média total dos custos de todas as execuções
total_average_cost = np.mean([result[1] for result in results])
total_execution_time = np.sum([result[2] for result in results])

# Escreve os resultados no arquivo de resultados
with open(results_file, 'w') as file:
    file.write(f"Parameters:\n")
    file.write(f"num_ants = {num_ants}\n")
    file.write(f"num_iterations = {num_iterations}\n")
    file.write(f"alpha = {alpha}\n")
    file.write(f"beta = {beta}\n")
    file.write(f"rho = {rho}\n")
    file.write(f"Q = {Q}\n\n")

    for i, result in enumerate(results):
        best_solution, best_cost, execution_time, best_solution_costs, prizes_collected, penalties, average_colony_costs, worst_solution_costs = result
        file.write(f"Execution {i+1}:\n")
        file.write(f"Best Solution: {best_solution}\n")
        file.write(f"Cost: {best_cost}\n")
        file.write(f"Execution Time: {execution_time} seconds\n\n")
        
        file.write("Results per Iteration:\n")
        file.write("Iteration, Best Cost, Prizes Collected, Penalties, Average Colony Cost, Worst Solution Cost\n")
        for j in range(len(best_solution_costs)):
            file.write(f"{j+1}, {best_solution_costs[j]}, {prizes_collected[j]}, {penalties[j]}, {average_colony_costs[j]}, {worst_solution_costs[j]}\n")
        file.write("\n")

    file.write(f"Total Average Cost: {total_average_cost}\n")
    file.write(f"Total Execution Time: {total_execution_time} seconds\n")

print("Results saved to", results_file)

# Agrupa os resultados das 10 execuções
mean_best_solution_costs = np.mean([result[3] for result in results], axis=0)
mean_prizes_collected = np.mean([result[4] for result in results], axis=0)
mean_penalties = np.mean([result[5] for result in results], axis=0)
mean_average_colony_costs = np.mean([result[6] for result in results], axis=0)
mean_worst_solution_costs = np.mean([result[7] for result in results], axis=0)

# Plotando os gráficos considerando os valores das 10 execuções
# Gráfico da melhor solução encontrada ao longo das iterações (individual e média)
plt.figure(figsize=(12, 8))
for i, result in enumerate(results):
    plt.plot(range(1, len(result[3]) + 1), result[3], label=f'Execução {i+1}', alpha=0.5)
plt.plot(range(1, len(mean_best_solution_costs) + 1), mean_best_solution_costs, label='Melhor Solução Média', color='purple', linestyle='--', linewidth=2)
plt.xlabel('Iteração', fontsize=18)
plt.ylabel('Custo', fontsize=18)
plt.title('Melhor Solução Encontrada ao Longo das Iterações', fontsize=16)
plt.legend(loc='best', fontsize='large', frameon=True, facecolor='white', edgecolor='black')
plt.xticks(range(0, len(mean_best_solution_costs) + 1, 5), fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()

# # Gráfico de prêmio coletado e penalidades por iteração
# plt.figure(figsize=(12, 8))
# plt.plot(range(1, len(mean_prizes_collected) + 1), mean_prizes_collected, label='Prêmio Coletado', color='blue')
# plt.plot(range(1, len(mean_penalties) + 1), mean_penalties, label='Penalidade', color='red')
# plt.xlabel('Iteração', fontsize=18)
# plt.ylabel('Valor', fontsize=18)
# plt.title('Prêmio Coletado e Penalidades por Iteração', fontsize=16)
# plt.legend(loc='best', fontsize='xx-large', frameon=True, facecolor='white', edgecolor='black')
# plt.xticks(range(0, len(mean_prizes_collected) + 1, 5), fontsize=18)
# plt.yticks(fontsize=18)
# plt.grid(True)
# plt.show()

# Gráfico da melhor solução média encontrada ao longo das iterações
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(mean_best_solution_costs) + 1), mean_best_solution_costs, label='Melhor Solução Média', color='purple')
plt.xlabel('Iteração', fontsize=18)
plt.ylabel('Custo', fontsize=18)
plt.title('Melhor Solução Média Encontrada ao Longo das Iterações', fontsize=16)
plt.legend(loc='best', fontsize='xx-large', frameon=True, facecolor='white', edgecolor='black')
plt.xticks(range(0, len(mean_best_solution_costs) + 1, 5), fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()

# Gráfico com três linhas: melhor caminho, média da colônia e pior caminho ao longo das iterações
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(mean_best_solution_costs) + 1), mean_best_solution_costs, label='Melhor Caminho')
plt.plot(range(1, len(mean_average_colony_costs) + 1), mean_average_colony_costs, label='Média da Colônia')
plt.plot(range(1, len(mean_worst_solution_costs) + 1), mean_worst_solution_costs, label='Pior Caminho')
plt.xlabel('Iteração', fontsize=18)
plt.ylabel('Custo', fontsize=18)
plt.title('Melhor Caminho, Média da Colônia e Pior Caminho ao Longo das Iterações', fontsize=16)
plt.legend(loc='upper right', fontsize='large', frameon=True, facecolor='white', edgecolor='black')
plt.xticks(range(0, len(mean_best_solution_costs) + 1, 5), fontsize=18)
max_y = max(max(mean_best_solution_costs), max(mean_average_colony_costs), max(mean_worst_solution_costs))
yticks_values = range(1800, int(max_y), 1000)
plt.yticks(yticks_values, fontsize=18)
plt.grid(True)
plt.show()
