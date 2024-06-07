import random

# Função para gerar dados de entrada aleatórios dentro de um intervalo aceitável
def generate_random_data(num_nodes):
    prize_vector = [random.randint(1, 100) for _ in range(num_nodes)]
    penalty_vector = [random.randint(1, 750) for _ in range(num_nodes)]
    cost_matrix = [[random.randint(50, 1000) for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        cost_matrix[i][i] = 0  # Define 0 na diagonal para representar que não há custo de um nó para ele mesmo
    return cost_matrix, prize_vector, penalty_vector

# Gera os dados de entrada
#num_nodes = 10 
#num_nodes = 15 
#num_nodes = 20
num_nodes = 25
#num_nodes = 50
#num_nodes = 100

cost_matrix, prize_vector, penalty_vector = generate_random_data(num_nodes)

# Escreve os dados de entrada em um arquivo
with open('data25txt', 'w') as file:
    file.write(f'Cost Matrix:\n')
    for row in cost_matrix:
        file.write(' '.join(map(str, row)) + '\n')
    file.write(f'\nPrize Vector:\n')
    file.write(' '.join(map(str, prize_vector)) + '\n')
    file.write(f'Penalty Vector:\n')
    file.write(' '.join(map(str, penalty_vector)) + '\n')

print("Dados de entrada gerados e escritos no arquivo 'data.txt'.")
