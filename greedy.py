import numpy as np
import time

# Função objetivo
def objective_function(solution, cost_matrix, prize_vector, penalty_vector, alpha, max_penalty, min_prize):
    total_cost = sum(cost_matrix[solution[i]][solution[i+1]] for i in range(len(solution) - 1))
    total_cost += cost_matrix[solution[-1]][solution[0]]  # custo de retorno ao início

    # Cálculo do valor coletado
    collected_value = sum(prize_vector[i] for i in solution)

    # Total objective
    total_objective = total_cost + alpha * (max_penalty - collected_value)
    return total_objective

# Algoritmo da Busca Gulosa (Greedy) com Aleatoriedade
def greedy_tsp_prize_collecting(cost_matrix, prize_vector, penalty_vector, min_prize, tmax, alpha=0.2, max_penalty=0):
    start_time = time.time()
    
    n = len(cost_matrix)
    unvisited = set(range(1, n))  # Todos os nós, exceto o nó inicial
    current_node = 0  # Começa do nó inicial fixo
    tour = [current_node]
    total_cost = 0
    total_prize = prize_vector[current_node]
    total_penalty = penalty_vector[current_node]
    iterations = 0
    
    costs_per_iteration = [0]  # Inicia com custo zero
    
    while unvisited and time.time() - start_time < tmax:
        neighbors = list(unvisited)
        
        next_node = None
        min_cost = float('inf')
        
        for neighbor in neighbors:
            if cost_matrix[current_node][neighbor] < min_cost:
                min_cost = cost_matrix[current_node][neighbor]
                next_node = neighbor
        
        if next_node is None:
            break
        
        # Adiciona o nó com menor custo ao tour
        tour.append(next_node)
        unvisited.remove(next_node)
        total_cost += min_cost
        total_prize += prize_vector[next_node]
        total_penalty += penalty_vector[next_node]
        current_node = next_node
        iterations += 1
        
        # Armazena o custo atual
        costs_per_iteration.append(total_cost)

    # Adiciona o nó de retorno ao início do tour
    tour.append(tour[0])
    total_cost += cost_matrix[current_node][tour[0]]  # Adiciona o custo para retornar ao nó inicial
    
    # Calcula a função objetivo
    objective = objective_function(tour, cost_matrix, prize_vector, penalty_vector, alpha, max_penalty, min_prize)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return tour, total_cost, total_prize, total_penalty, objective, execution_time, iterations, costs_per_iteration

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

# Função para resolver e armazenar resultados
def solve_and_store_results(file_name, output_file):
    # Lê os dados do arquivo
    prize_vector, penalty_vector, cost_matrix = read_data(file_name)

    # Calcula min_prize
    min_prize = int(0.75 * sum(prize_vector))

    # Parâmetros
    tmax = 6500  # Tempo máximo de execução
    num_execucoes = 10
    alpha = 0.2
    max_penalty = 0

    # Listas para armazenar os resultados
    best_costs = []
    best_objectives = []
    execution_times = []
    best_solutions = []
    iterations_list = []
    collected_prizes = []
    penalties = []
    all_solutions = []
    all_costs_per_iteration = []

    # Resolve o problema
    for i in range(num_execucoes):
        tour_greedy, total_cost_greedy, total_prize_greedy, total_penalty_greedy, objective_greedy, execution_time_greedy, iterations_greedy, costs_per_iteration_greedy = greedy_tsp_prize_collecting(cost_matrix, prize_vector, penalty_vector, min_prize, tmax, alpha, max_penalty)

        best_costs.append(total_cost_greedy)
        best_objectives.append(objective_greedy)
        execution_times.append(execution_time_greedy)
        best_solutions.append(tour_greedy)
        iterations_list.append(iterations_greedy)
        collected_prizes.append(total_prize_greedy)
        penalties.append(total_penalty_greedy)
        all_solutions.append((tour_greedy, total_cost_greedy, total_prize_greedy, total_penalty_greedy))
        all_costs_per_iteration.append(costs_per_iteration_greedy)

        with open(output_file, 'a') as file:
            file.write(f"Execution {i+1}:\n")
            file.write(f"Best Cost: {tour_greedy}\n")
            file.write(f"Total Cost of Best Solution: {total_cost_greedy}\n")
            file.write(f"Objective Value: {objective_greedy}\n")
            file.write(f"Execution Time: {execution_time_greedy} seconds\n\n")

    # Calcula a função objetivo para cada iteração
    objective_values = []
    for costs_per_iteration in all_costs_per_iteration:
        objective_values_iter = []
        for i in range(len(costs_per_iteration)):
            tour = all_solutions[0][0][:i+2]  # Melhor solução até a iteração atual
            objective_value = objective_function(tour, cost_matrix, prize_vector, penalty_vector, alpha, max_penalty, min_prize)
            objective_values_iter.append(objective_value)
        objective_values.append(objective_values_iter)

# Executa o algoritmo para o arquivo data10.txt e armazena os resultados
solve_and_store_results('data15.txt', 'resultados_greedy15.txt')
