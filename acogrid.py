import numpy as np
import random
import time
import matplotlib.pyplot as plt

def objective_function(solution, cost_matrix, prize_vector, alpha, min_prize):
    # Cálculo do custo total do caminho na solução
    total_cost = sum(cost_matrix[solution[i]][solution[i+1]] for i in range(len(solution) - 1))
    total_cost += cost_matrix[solution[-1]][solution[0]]  # custo de retorno ao início

    # Cálculo da pontuação coletada
    collected_value = sum(prize_vector[node] for node in solution)

    # Cálculo da penalidade pela pontuação não coletada
    missing_prize_penalty = max(0, min_prize - collected_value) * alpha

    # Cálculo do objetivo total
    total_objective = total_cost + missing_prize_penalty
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
                    probabilities += 1e-10
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

            total_prize = sum(prize_vector[i] for i in solution)
            total_penalty = max(0, min_prize - total_prize) * alpha
            
            iteration_prizes.append(total_prize)
            iteration_penalties.append(total_penalty)

            # Atualiza a melhor solução encontrada até agora
            if cost + total_penalty < best_cost:
                best_solution = solution
                best_cost = cost + total_penalty

        best_solution_costs.append(best_cost)
        prizes_collected.append(max(iteration_prizes))
        penalties.append(max(iteration_penalties))
        average_colony_costs.append(np.mean(costs))
        worst_solution_costs.append(max(costs))

        # Atualiza a matriz de feromônio
        pheromone *= (1 - rho)
        for solution, cost in zip(solutions, costs):
            for i in range(len(solution) - 1):
                pheromone[solution[i]][solution[i + 1]] += Q / (cost + total_penalty)
            # Atualiza o feromônio também para o caminho de retorno ao nó inicial
            pheromone[solution[-1]][solution[0]] += Q / (cost + total_penalty)
    
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

def grid_search(cost_matrix, prize_vector, penalty_vector, num_ants_range, num_iterations_range, alpha_range, beta_range, rho_range, Q_range, num_repeats):
    best_params = None
    best_solution = None
    best_cost = float('inf')
    grid_params = []

    param_combinations = [(num_ants, num_iterations, alpha, beta, rho, Q)
                          for num_ants in num_ants_range
                          for num_iterations in num_iterations_range
                          for alpha in alpha_range
                          for beta in beta_range
                          for rho in rho_range
                          for Q in Q_range]

    for num_ants, num_iterations, alpha, beta, rho, Q in param_combinations:
        total_best_cost = 0
        total_execution_time = 0
        for _ in range(num_repeats):
            # Removendo min_prize como argumento nomeado
            _, cost, exec_time, _, _, _, _, _ = ant_colony_optimization(cost_matrix, prize_vector, penalty_vector, num_ants, num_iterations, alpha, beta, rho, Q, int(0.75 * sum(prize_vector)))
            total_best_cost += cost
            total_execution_time += exec_time

        avg_best_cost = total_best_cost / num_repeats
        avg_execution_time = total_execution_time / num_repeats

        params = {
            'num_ants': num_ants,
            'num_iterations': num_iterations,
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'Q': Q
        }

        grid_params.append((params, avg_best_cost, avg_execution_time))

        if avg_best_cost < best_cost:
            best_params = params
            best_cost = avg_best_cost
            # Removendo min_prize como argumento nomeado
            best_solution, _, _, _, _, _, _, _ = ant_colony_optimization(cost_matrix, prize_vector, penalty_vector, num_ants, num_iterations, alpha, beta, rho, Q, int(0.75 * sum(prize_vector)))

    return grid_params, best_params, best_solution, best_cost

# Parâmetros do Grid para otimização do ACO
num_ants_range = [10, 20, 30, 40] 
num_iterations_range = [25, 50, 75, 100]
alpha_range = [0,0.5,1,2] 
beta_range = [0,0.5,1,2] 
rho_range = [0.1, 0.2, 0.3, 0.4] 
Q_range = [1, 10, 100, 1000] 
num_repeats = 5

# Nome do arquivo de entrada
file_name = 'data10.txt'

# Nome do arquivo de resultados
results_file = 'best_ACO_GRID10.txt'

# Lê os dados do arquivo
prize_vector, penalty_vector, cost_matrix = read_data(file_name)

# Executa o Grid Search para otimização dos parâmetros do ACO
grid_params, best_params, best_solution, best_cost = grid_search(cost_matrix, prize_vector, penalty_vector, num_ants_range, num_iterations_range, alpha_range, beta_range, rho_range, Q_range, num_repeats)

# Calcula o tempo de execução total
total_execution_time = sum(avg_exec_time for _, _, avg_exec_time in grid_params)

# Escreve os resultados no arquivo de resultados
with open(results_file, 'w') as file:
    for params, avg_best_cost, avg_execution_time in grid_params:
        file.write(f"Parameters: {params}\n")
        file.write("Average Best Cost: {}\n".format(avg_best_cost))
        file.write("Average Execution Time: {} seconds\n\n".format(avg_execution_time))
    
    file.write("Best Parameters Found:\n")
    file.write(f"Parameters: {best_params}\n")
    file.write("Best Solution: {}\n".format(best_solution))
    file.write("Best Cost: {}\n".format(best_cost))

print("Grid search completed. Results saved to", results_file)

# Plotando os gráficos
_, _, _, best_solution_costs, prizes_collected, penalties, average_colony_costs, worst_solution_costs = ant_colony_optimization(
    cost_matrix, prize_vector, penalty_vector,
    best_params['num_ants'], best_params['num_iterations'],
    best_params['alpha'], best_params['beta'], best_params['rho'], best_params['Q'],
    min_prize=int(0.75 * sum(prize_vector))
)

# Gráfico da melhor solução encontrada ao longo das iterações
plt.figure(figsize=(12, 8))
plt.plot(best_solution_costs, label='Melhor Solução')
plt.xlabel('Iteração', fontsize=12)
plt.ylabel('Custo', fontsize=12)
plt.title('Melhor Solução Encontrada ao Longo das Iterações')
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

# Gráfico com três linhas: melhor caminho, média da colônia e pior caminho ao longo das iterações
plt.figure(figsize=(12, 8))
plt.plot(best_solution_costs, label='Melhor Caminho')
plt.plot(average_colony_costs, label='Média da Colônia')
plt.plot(worst_solution_costs, label='Pior Caminho')
plt.xlabel('Iteração', fontsize=12)
plt.ylabel('Custo', fontsize=12)
plt.title('Melhor Caminho, Média da Colônia e Pior Caminho ao Longo das Iterações')
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()
