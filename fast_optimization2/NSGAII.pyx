# cython: boundscheck=False
import numpy as np
cimport numpy as np
cimport cython
from IHSetYates09.fast_simulator import model_simulation
from fast_optimization2.objective_functions import obj_func

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fast_non_dominated_sort(np.ndarray[double, ndim=2] objectives):
    """
    Fast non-dominated sort algorithm for NSGA-II
    """
    cdef int population_size = objectives.shape[0]
    cdef np.ndarray[int, ndim=1] domination_count = np.zeros(population_size, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] dominated_solutions = np.full((population_size, population_size), -1, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] current_counts = np.zeros(population_size, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] ranks = np.zeros(population_size, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] front_indices = np.full((population_size, population_size), -1, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] front_sizes = np.zeros(population_size, dtype=np.int32)
    cdef int p, q, i, j, k, next_front_size

    for p in range(population_size):
        for q in range(population_size):
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dominated_solutions[p, current_counts[p]] = q
                current_counts[p] += 1
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            front_indices[0, front_sizes[0]] = p
            front_sizes[0] += 1

    i = 0
    while front_sizes[i] > 0:
        next_front_size = 0
        for j in range(front_sizes[i]):
            p = front_indices[i, j]
            for k in range(current_counts[p]):
                q = dominated_solutions[p, k]
                if q == -1:
                    break
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    front_indices[i+1, next_front_size] = q
                    next_front_size += 1
        front_sizes[i+1] = next_front_size
        i += 1

    return ranks, front_indices, front_sizes

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] calculate_crowding_distance(np.ndarray[double, ndim=2] objectives, np.ndarray[int, ndim=1] front):
    """
    Calculate crowding distance for a front
    """
    cdef int num_individuals = len(front)
    cdef int num_objectives = objectives.shape[1]
    cdef np.ndarray[double, ndim=1] distance = np.zeros(num_individuals)
    cdef np.ndarray[int, ndim=1] sorted_indices
    cdef double next_value, prev_value, norm
    cdef int m, i

    print("Check crowding, num_individuals:", num_individuals)
    print("Check crowding, num_objectives:", num_objectives)
    print("Check crowding, len(front):", len(front))
    print("Check crowding, len(objectives):", len(objectives))

    for m in range(num_objectives):
        print("Check crowding 1")
        sorted_indices = np.argsort(objectives[front, m]).astype(np.int32)
        print("Check crowding 1.1")
        print("sorted_indices:", sorted_indices)
        print("objectives[front, m]:", objectives[front, m])
        print("front dtype:", front.dtype)
        print("sorted_indices dtype:", sorted_indices.dtype)
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf
        print("Check crowding 2")

        for i in range(1, num_individuals - 1):
            next_value = objectives[front[sorted_indices[i + 1]], m]
            prev_value = objectives[front[sorted_indices[i - 1]], m]
            norm = objectives[front[sorted_indices[-1]], m] - objectives[front[sorted_indices[0]], m]
            if norm != 0:
                distance[sorted_indices[i]] += (next_value - prev_value) / norm
        print("Check crowding 3")
    print("Check crowding 4")
    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[int, ndim=1] fast_sort(np.ndarray[double, ndim=2] scores):
    """
    Fast sort algorithm
    """
    cdef int num_individuals = scores.shape[0]
    cdef np.ndarray[int, ndim=1] ranks = np.zeros(num_individuals, dtype=np.int32)
    cdef int i, j

    for i in range(num_individuals):
        for j in range(num_individuals):
            if np.all(scores[i] <= scores[j]) and np.any(scores[i] < scores[j]):
                ranks[j] += 1

    return ranks

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] select_next_generation(np.ndarray[double, ndim=2] population, np.ndarray[double, ndim=2] scores, int population_size):
    """
    Select the next generation of the population
    """
    cdef np.ndarray[int, ndim=1] ranks = fast_sort(scores)
    return population[np.argsort(ranks)[:population_size]]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[int, ndim=1] tournament_selection(np.ndarray[double, ndim=2] scores, int pressure):
    """
    Tournament selection algorithm
    """
    cdef int n_select = len(scores)
    cdef int n_random = n_select * pressure
    cdef np.ndarray[int, ndim=1] indices = np.random.permutation(np.arange(n_select).repeat(pressure))[:n_random]
    indices = indices.reshape(n_select, pressure)

    cdef np.ndarray[int, ndim=1] selected_indices = np.empty(n_select, dtype=np.int32)
    cdef int i

    for i in range(n_select):
        selected_indices[i] = indices[i, np.argmin(scores[indices[i]])]
    return selected_indices

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] crossover(np.ndarray[double, ndim=2] population, int num_vars, double crossover_prob, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    """
    Crossover operation for genetic algorithm
    """
    cdef int i, j, cross_point
    cdef np.ndarray[double, ndim=1] temp

    for i in range(0, len(population), 2):
        if i + 1 >= len(population):
            break
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(1, num_vars)
            temp = population[i, cross_point:].copy()
            population[i, cross_point:] = population[i + 1, cross_point:].copy()
            population[i + 1, cross_point:] = temp

            for j in range(cross_point, num_vars):
                population[i, j] = min(max(population[i, j], lower_bounds[j]), upper_bounds[j])
                population[i + 1, j] = min(max(population[i + 1, j], lower_bounds[j]), upper_bounds[j])

    return population

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] polynomial_mutation(np.ndarray[double, ndim=2] population, double mutation_rate, int num_vars, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    """
    Polynomial mutation for genetic algorithm
    """
    cdef int i, j
    cdef double range_val, delta, mutated_value

    for i in range(len(population)):
        for j in range(num_vars):
            if np.random.rand() < mutation_rate:
                range_val = upper_bounds[j] - lower_bounds[j]
                delta = np.random.uniform(-0.05 * range_val, 0.05 * range_val)
                mutated_value = population[i, j] + delta
                mutated_value = max(lower_bounds[j], min(mutated_value, upper_bounds[j]))
                population[i, j] = mutated_value

    return population

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] fitness_sharing(np.ndarray[double, ndim=2] objectives, double sigma_share=0.01):
    """
    Fitness sharing for maintaining diversity in genetic algorithm
    """
    cdef int population_size = len(objectives)
    cdef np.ndarray[double, ndim=2] distances = np.zeros((population_size, population_size))
    cdef np.ndarray[double, ndim=2] sharing_values = np.zeros((population_size, population_size))
    cdef np.ndarray[double, ndim=1] modified_fitness = np.zeros(population_size)
    cdef int i, j
    cdef double niche_count

    for i in range(population_size):
        for j in range(i + 1, population_size):
            distances[i, j] = distances[j, i] = np.sqrt(np.sum((objectives[i] - objectives[j])**2))

    for i in range(population_size):
        for j in range(population_size):
            if distances[i, j] < sigma_share:
                sharing_values[i, j] = 1 - (distances[i, j] / sigma_share)
            else:
                sharing_values[i, j] = 0

    niche_counts = np.sum(sharing_values, axis=1)
    for i in range(population_size):
        modified_fitness[i] = 1.0 / (1.0 + niche_counts[i])

    return modified_fitness

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple select_niched_population(np.ndarray[double, ndim=2] population, np.ndarray[double, ndim=2] objectives, int num_to_select):
    """
    Select population based on fitness sharing
    """
    cdef np.ndarray[double, ndim=1] modified_fitness = fitness_sharing(objectives)
    cdef np.ndarray[int, ndim=1] selected_indices = np.argsort(-modified_fitness)[:num_to_select]
    return population[selected_indices], objectives[selected_indices]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple nsgaii_algorithm_(np.ndarray[double, ndim=1] Obs, int num_generations, int population_size, double cross_prob, double mutation_rate, double regeneration_rate, list index_metrics, np.ndarray[double, ndim=1] E, double dt, np.ndarray[int, ndim=1] idx_obs, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    """
    NSGA-II algorithm with additional parameters
    """
    cdef np.ndarray[double, ndim=2] population, objectives, new_objectives
    cdef np.ndarray[int, ndim=1] ranks, next_population_indices, current_front, selected_indices, sorted_indices
    cdef int npar, num_to_regenerate, i, current_size, remaining_space, generation
    cdef double crowding_distances

    # Chame a função para inicializar a população e obtenha os limites inferiores e superiores
    population, lower_bounds, upper_bounds = initialize_population(population_size, lower_bounds, upper_bounds)
    npar = population.shape[1]
    objectives = np.empty((population_size, 3))

    for i in range(population_size):
        simulation = model_simulation(E, dt, Obs[0], idx_obs, population[i])
        objectives[i] = obj_func(Obs, simulation, index_metrics)

    num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

    for generation in range(num_generations):
        ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
        next_population_indices = np.empty(0, dtype=np.int32)
        current_size = 0
        i = 0
        while i < population_size and front_sizes[i] > 0:
            print("Check 1")
            current_front = front_indices[i, :front_sizes[i]]
            crowding_distances = calculate_crowding_distance(objectives, current_front.astype(np.int32))
            print("Check 2")
            sorted_indices = np.argsort(-crowding_distances)
            selected_indices = current_front[sorted_indices]
            print("Check 3")
            if current_size + len(selected_indices) > population_size - num_to_regenerate:
                remaining_space = population_size - num_to_regenerate - current_size
                next_population_indices = np.append(next_population_indices, selected_indices[:remaining_space])
                break
            print("Check 4")
            next_population_indices = np.append(next_population_indices, selected_indices)
            current_size += len(selected_indices)
            i += 1
            print("Check 5")

        print("Check 6")
        mating_pool = population[next_population_indices.astype(np.int32)]
        offspring = crossover(mating_pool, npar, cross_prob, lower_bounds, upper_bounds)
        offspring = polynomial_mutation(offspring, mutation_rate, npar, lower_bounds, upper_bounds)

        print("Check 7")
        new_individuals = initialize_population(num_to_regenerate, lower_bounds, upper_bounds)
        offspring = np.vstack((offspring, new_individuals))
        print("Check 8")
        new_objectives = np.empty_like(objectives)
        for i in range(population_size):
            simulation = model_simulation(E, dt, Obs[0], idx_obs, offspring[i])
            new_objectives[i] = obj_func(Obs, simulation, index_metrics)

        print("Check 9")
        population = offspring
        objectives = new_objectives

        print("Check 10")
        if generation % 10 == 0:
            print(f"Generation {generation} of {num_generations} completed")
        
        print("Check 11")

    return population, objectives

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple initialize_population(int population_size, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    """
    Initialize population within the given bounds
    """
    cdef np.ndarray[double, ndim=2] population = np.zeros((population_size, len(lower_bounds)))
    cdef int i

    for i in range(len(lower_bounds)):
        population[:, i] = np.random.uniform(lower_bounds[i], upper_bounds[i], population_size)

    return population, lower_bounds, upper_bounds
