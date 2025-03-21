import networkx as nx
import os
import matplotlib.pyplot as plt
import random
import numpy as np

def read_fvs_file(filename):
    """
    Legge un file .fvs e costruisce un grafo NetworkX con pesi sui nodi
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    G = nx.Graph()
    node_weights = {}
    adjacency_matrix = []
    reading_weights = False
    reading_matrix = False

    for line in lines:
        line = line.strip()
        
        if line.startswith("NODE_WEIGHT_SECTION"):
            reading_weights = True
            continue
        if line.startswith("ADIACENT_LOWER_TRIANGULAR_MATRIX"):
            reading_weights = False
            reading_matrix = True
            continue

        if reading_weights and line and not line.startswith("END"):
            parts = line.split()
            node_id, weight = int(parts[0]), int(parts[1])
            node_weights[node_id] = weight
            G.add_node(node_id, weight=weight)

        row = []
        if reading_matrix and line and not line.startswith("END"):
            for x in line.split():
                row.append(int(x))
            adjacency_matrix.append(row)

    node_list = sorted(node_weights.keys())
    for i in range(len(node_list)):
        for j in range(i):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(node_list[i], node_list[j])
    
    return G, node_weights

def generate_initial_solution(graph, weights):
    """
    Genera una soluzione iniziale valida introducendo un fattore di casualità moderata:
    1. Ordina i nodi per (grado)^2 / peso, con una componente casuale
    2. Aggiunge nodi finché non si ottiene un FVS valido
    3. Rimuove nodi non necessari
    """
    nodes = list(graph.nodes())
    
    # Calcola l'importanza di ciascun nodo
    node_importance = []
    for n in nodes:
        importance = (graph.degree(n))**2 / weights[n] * (1 + random.uniform(-0.3, 0.3))
        node_importance.append((n, importance))
    
    # Ordina i nodi per importanza: (grado)^2 / peso, con una componente casuale
    sorted_nodes = []
    node_importance.sort(key=lambda x: x[1], reverse=True)
    for node_imp in node_importance:
        sorted_nodes.append(node_imp[0])
    
    # Costruzione della soluzione greedy
    current_fvs = set()
    temp_graph = graph.copy()
    
    for node in sorted_nodes:
        current_fvs.add(node)
        temp_graph.remove_node(node)
        
        # Verifica se il grafo rimanente è una foresta
        if nx.is_forest(temp_graph):
            break  # Abbiamo trovato un FVS valido
    
    # Fase di pulizia: rimozione nodi non necessari
    pruned_fvs = current_fvs.copy()
    temp_graph = graph.copy()  # Usa una singola copia del grafo per tutti i controlli
    
    # Calcola il criterio di ordinamento per ciascun nodo
    node_sorting = []
    for n in pruned_fvs:
        sorting_value = (-weights[n], graph.degree(n))
        node_sorting.append((n, sorting_value))
    
    # Ordina i nodi per peso decrescente e grado crescente
    node_sorting.sort(key=lambda x: x[1])
    sorted_fvs_nodes = []
    for node_sort in node_sorting:
        sorted_fvs_nodes.append(node_sort[0])
    
    for node in sorted_fvs_nodes:
        # Prova a rimuovere il nodo
        temp_graph.remove_node(node)
        
        # Verifica se il grafo rimanente è ancora una foresta
        if nx.is_forest(temp_graph):
            pruned_fvs.remove(node)  # Tieni il nodo rimosso
        else:
            # Ripristina il nodo e i suoi archi
            temp_graph.add_node(node)
            for neighbor in graph.neighbors(node):
                if neighbor in temp_graph.nodes():
                    temp_graph.add_edge(node, neighbor)
    
    return pruned_fvs

def find_fvs_tabu(graph, weights, max_iterations = 500, tabu_size = 15, time = 100):
    """
    Applica Tabu Search per trovare un Feedback Vertex Set (FVS) valido.
    Si ferma prima se non ci sono miglioramenti per 'time' iterazioni.
    Restituisce il set di candidati migliori, la somma minima, la curva di convergenza
    e il numero di valutazoini della funzione obiettivo
    """
    # Genera soluzione iniziale migliorata
    current_fvs = generate_initial_solution(graph, weights)
    best_fvs = current_fvs.copy()
    
    # Calcola la somma dei pesi della soluzione corrente
    min_weight = 0
    for n in current_fvs:
        min_weight += weights[n]
    
    # Contatore per il numero di valutazioni della funzione obiettivo
    evaluations_counter = 1  # Contiamo la soluzione iniziale
    # Contatore per early stopping
    iterations_without_improvement = 0

    tabu_list = []
    convergence_curve = [min_weight]

    for iteration in range(max_iterations):
        neighbors = []
        nodes = list(graph.nodes())
        
        # Genera vicini aggiungendo/rimuovendo un singolo nodo
        for node in nodes:
            new_fvs = current_fvs.copy()
            
            if node in new_fvs:
                new_fvs.remove(node)
            else:
                new_fvs.add(node)
            
            # Controlla se la soluzione è in tabu
            sorted_new_fvs = sorted(new_fvs)
            if tuple(sorted_new_fvs) in tabu_list:
                continue

            # Verifica validità della soluzione
            temp_graph = graph.copy()
            for n in new_fvs:
                temp_graph.remove_node(n)
            
            if nx.is_forest(temp_graph):
                evaluations_counter += 1  # Incrementa il contatore per ogni valutazione valida
                weight = 0
                for n in new_fvs:
                    weight += weights[n] # si aggiorna la somma dei pesi di ciascun vertice
                neighbors.append((new_fvs, weight))

        if not neighbors:
            break

        # Seleziona il miglior vicino
        neighbors.sort(key=lambda x: x[1])
        current_fvs, current_weight = neighbors[0]

        # Aggiorna la migliore soluzione trovata. Funzione obiettivo : minimizzare la somma dei pesi di ciascun vertice in FVS
        if current_weight < min_weight:
            best_fvs = current_fvs.copy()
            min_weight = current_weight
            iterations_without_improvement = 0  # Reset del contatore per early stopping
        else:
            iterations_without_improvement += 1

        # Verifica condizione di early stopping
        if iterations_without_improvement >= time:
            print(f"Early stopping attivato dopo {iteration+1} iterazioni: nessun miglioramento per {time} iterazioni")
            # Estendi la curva di convergenza con l'ultimo valore per preservare la dimensione
            for i in range(max_iterations - iteration - 1):
                convergence_curve.append(min_weight)
            break

        # Aggiorna la lista Tabu
        tabu_list.append(tuple(sorted(current_fvs)))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        convergence_curve.append(min_weight)

    return best_fvs, min_weight, convergence_curve, evaluations_counter

# -------------------------------
# ESECUZIONE: 10 run indipendenti
# -------------------------------

def process_instances(instance_files, early_stopping_time = 100):

    all_convergence_curves = []
    all_best_weights = []
    all_evaluations = []      # Lista per tenere traccia del numero di valutazioni

    for file_path in instance_files:
        print(f"Processando {os.path.basename(file_path)}...")
        graph, weights = read_fvs_file(file_path)
        
        # Esegui 10 run per ogni istanza
        instance_convergence = []
        instance_best_weights = []
        instance_evaluations = []
        
        for run in range(10):
            best_fvs, best_weight, convergence_curve, evaluations = find_fvs_tabu(
                graph, weights, time = early_stopping_time
            )
            instance_convergence.append(convergence_curve)
            instance_best_weights.append(best_weight)
            instance_evaluations.append(evaluations)
            
            print(f"  Run {run+1}: Somma dei pesi minima = {best_weight},"
                  f"Valutazioni totali = {evaluations}")
        
        # Aggiungi i risultati di questa istanza
        all_convergence_curves.append(instance_convergence)
        all_best_weights.append(instance_best_weights)
        all_evaluations.append(instance_evaluations)

    return all_convergence_curves, all_best_weights, all_evaluations

def plot_convergence(convergence_data):
    """
    Plotta la curva di convergenza per gruppi di 5 istanze.
    """
    plt.figure(figsize=(10, 6))
    
    for i, instance_curves in enumerate(convergence_data):
        normalized_curves = []
        for curve in instance_curves:
            normalized_curves.append(curve)

        # Calcola la media
        mean_values = np.mean(normalized_curves, axis=0)
        
        # Crea il plot
        plt.plot(mean_values, label=f"Istanza {i+1}")
    
    # Aggiungi elementi grafici
    plt.xlabel("Iterazione")
    plt.ylabel("Avg. BS FVS")
    plt.title("Confronto curve di convergenza")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# ESECUZIONE PRINCIPALE / MAIN 
# -------------------------------
# Percorso della cartella contenente le istanze

instances_folder = "./instances/grid/"

# Parametro per early stopping
early_stopping_time = 100  # Numero di iterazioni senza miglioramenti prima di fermarsi

# Raggruppa le istanze per tipo
instances_types = ["Grid_5_5", "Grid_7_7", "Grid_9_9"]
for grid_type in instances_types:
    print(f"\nProcesso istanze di tipo {instances_types}")
    
    # Trova tutti i file corrispondenti a questo tipo
    all_files = []
    for f in os.listdir(instances_folder):
        if f.startswith(grid_type) and f.endswith(".fvs"):
            all_files.append(os.path.join(instances_folder, f))
    
    # Processa i file in gruppi di 5
    for i in range(0, len(all_files), 5):
        group_files = all_files[i:i+5]
        if not group_files:
            continue
        
        print(f"\nGruppo:")
        convergence_data, best_weights, evaluations = process_instances(
            group_files, early_stopping_time
        )
        
        # Calcola statistiche sui pesi / soluzioni
        all_weights = []
        for instance in best_weights:
            for w in instance:
                all_weights.append(w)
        mean_weight = np.mean(all_weights)
        std_weight = np.std(all_weights)
        
        # Calcola statistiche sul numero di valutazioni
        all_evals = []
        for instance in evaluations:
            for ev in instance:
                all_evals.append(ev)
        mean_evals = np.mean(all_evals)
        
        print(f"\nRisultati per il gruppo:")
        print(f"Avg. BS: {mean_weight:.2f}")
        print(f"Std BS: {std_weight:.2f}")
        print(f"Avg. valutazioni della funzione obiettivo: {mean_evals:.2f}")
        
        # Plot della convergenza combinata
        plot_convergence(convergence_data)