# cython: boundscheck=False
# cython -a -c=-O3 -c=-march=native -c=-ffast-math -c=-funroll-loops
import numpy as np
cimport numpy as np
cimport cython
from IHSetYates09.fast_simulator import model_simulation
from fast_optimization2.objective_functions import obj_func
from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX
from libc.math cimport sqrt, exp, log
from libc.time cimport time

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple fast_non_dominated_sort(np.ndarray[double, ndim=2] objectives):
    cdef int population_size = objectives.shape[0]
    cdef int p, q, i, j, k, next_front_size
    cdef int[:] domination_count = np.zeros(population_size, dtype=np.int32)
    cdef int[:] current_counts = np.zeros(population_size, dtype=np.int32)
    cdef int[:] ranks = np.zeros(population_size, dtype=np.int32)
    cdef int[:] front_sizes = np.zeros(population_size, dtype=np.int32)
    cdef int[:,:] dominated_solutions = np.empty((population_size, population_size), dtype=np.int32)
    cdef int[:,:] front_indices = np.empty((population_size, population_size), dtype=np.int32)

    for p in range(population_size):
        for q in range(population_size):
            if all(objectives[p, :] <= objectives[q, :]) and any(objectives[p, :] < objectives[q, :]):
                dominated_solutions[p, current_counts[p]] = q
                current_counts[p] += 1
            elif all(objectives[q, :] <= objectives[p, :]) and any(objectives[q, :] < objectives[p, :]):
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
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=1] calculate_crowding_distance(np.ndarray[double, ndim=2] objectives, np.ndarray[int, ndim=1] front):
    cdef int num_individuals = len(front)
    cdef int num_objectives = objectives.shape[1]
    cdef np.ndarray[double, ndim=1] distance = np.empty(num_individuals)
    cdef int[:] sorted_indices = np.empty(num_individuals, dtype=np.int32)
    cdef double next_value, prev_value, norm
    cdef int m, i

    for m in range(num_objectives):
        # Manually sort indices based on objectives[front, m]
        for i in range(num_individuals):
            sorted_indices[i] = i
        for i in range(num_individuals - 1):
            for j in range(i + 1, num_individuals):
                if objectives[front[sorted_indices[i]], m] > objectives[front[sorted_indices[j]], m]:
                    sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]

        distance[sorted_indices[0]] = 1e308  # equivalent to np.inf
        distance[sorted_indices[num_individuals - 1]] = 1e308

        for i in range(1, num_individuals - 1):
            next_value = objectives[front[sorted_indices[i + 1]], m]
            prev_value = objectives[front[sorted_indices[i - 1]], m]
            norm = objectives[front[sorted_indices[num_individuals - 1]], m] - objectives[front[sorted_indices[0]], m]
            if norm != 0:
                distance[sorted_indices[i]] += (next_value - prev_value) / norm

    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[int, ndim=1] fast_sort(np.ndarray[double, ndim=2] scores):
    cdef int num_individuals = scores.shape[0]
    cdef np.ndarray[int, ndim=1] ranks = np.zeros(num_individuals, dtype=np.int32)
    cdef int i, j

    for i in range(num_individuals):
        for j in range(num_individuals):
            if all(scores[i, :] <= scores[j, :]) and any(scores[i, :] < scores[j, :]):
                ranks[j] += 1

    return ranks

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] select_next_generation(np.ndarray[double, ndim=2] population, np.ndarray[double, ndim=2] scores, int population_size):
    cdef np.ndarray[int, ndim=1] ranks = fast_sort(scores)
    return population[np.argsort(ranks)[:population_size]]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[int, ndim=1] tournament_selection(np.ndarray[double, ndim=2] scores, int pressure):
    cdef int n_select = len(scores)
    cdef int n_random = n_select * pressure
    cdef np.ndarray[int, ndim=1] indices = np.empty(n_random)
    cdef int i, idx

    for i in range(n_random):
        indices[i] = rand() % n_select

    selected_indices = np.empty(n_select, dtype=np.int32)

    for i in range(n_select):
        idx = 0
        for j in range(pressure):
            if scores[indices[i * pressure + j]] < scores[indices[i * pressure + idx]]:
                idx = j
        selected_indices[i] = indices[i * pressure + idx]
    return selected_indices

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] crossover(np.ndarray[double, ndim=2] population, int num_vars, double crossover_prob, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    cdef int i, j, cross_point
    cdef np.ndarray[double, ndim=1] temp

    for i in range(0, len(population), 2):
        if i + 1 >= len(population):
            break
        if rand() / <double>RAND_MAX < crossover_prob:
            cross_point = (rand() % (num_vars - 1)) + 1
            for j in range(cross_point, num_vars):
                temp = population[i, j]
                population[i, j] = population[i + 1, j]
                population[i + 1, j] = temp
                population[i, j] = max(min(population[i, j], upper_bounds[j]), lower_bounds[j])
                population[i + 1, j] = max(min(population[i + 1, j], upper_bounds[j]), lower_bounds[j])

    return population

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] polynomial_mutation(np.ndarray[double, ndim=2] population, double mutation_rate, int num_vars, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    cdef int i, j
    cdef double range_val, delta, mutated_value

    for i in range(len(population)):
        for j in range(num_vars):
            if rand() / <double>RAND_MAX < mutation_rate:
                range_val = upper_bounds[j] - lower_bounds[j]
                delta = (rand() / <double>RAND_MAX - 0.5) * 0.1 * range_val
                mutated_value = population[i, j] + delta
                mutated_value = max(lower_bounds[j], min(mutated_value, upper_bounds[j]))
                population[i, j] = mutated_value

    return population

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=1] fitness_sharing(np.ndarray[double, ndim=2] objectives, double sigma_share=0.01):
    cdef int population_size = len(objectives)
    cdef np.ndarray[double, ndim=2] distances = np.zeros((population_size, population_size))
    cdef np.ndarray[double, ndim=2] sharing_values = np.zeros((population_size, population_size))
    cdef np.ndarray[double, ndim=1] modified_fitness = np.zeros(population_size)
    cdef int i, j
    cdef double niche_count

    for i in range(population_size):
        for j in range(i + 1, population_size):
            distances[i, j] = distances[j, i] = sqrt(sum([(objectives[i, k] - objectives[j, k]) ** 2 for k in range(objectives.shape[1])]))

    for i in range(population_size):
        for j in range(population_size):
            if distances[i, j] < sigma_share:
                sharing_values[i, j] = 1 - (distances[i, j] / sigma_share)
            else:
                sharing_values[i, j] = 0

    niche_counts = [sum(sharing_values[i, :]) for i in range(population_size)]
    for i in range(population_size):
        modified_fitness[i] = 1.0 / (1.0 + niche_counts[i])

    return modified_fitness

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple select_niched_population(np.ndarray[double, ndim=2] population, np.ndarray[double, ndim=2] objectives, int num_to_select):
    cdef np.ndarray[double, ndim=1] modified_fitness = fitness_sharing(objectives)
    cdef np.ndarray[int, ndim=1] selected_indices = np.argsort(-modified_fitness)[:num_to_select]
    return population[selected_indices], objectives[selected_indices]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple initialize_population(int population_size, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):
    cdef np.ndarray[double, ndim=2] population = np.empty((population_size, len(lower_bounds)))
    cdef int i, j
    cdef double scale, offset

    # Seed the random number generator with the current time
    srand(<unsigned int>time(NULL))

    for j in range(len(lower_bounds)):
        scale = upper_bounds[j] - lower_bounds[j]
        offset = lower_bounds[j]
        for i in range(population_size):
            # Generate a random number between 0 and 1, then scale and offset it
            population[i, j] = offset + scale * (rand() / <double>RAND_MAX)

    return population, lower_bounds, upper_bounds

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple nsgaii_algorithm_(np.ndarray[double, ndim=1] Obs, int num_generations, int population_size, double cross_prob, double mutation_rate, double regeneration_rate, list index_metrics, np.ndarray[double, ndim=1] E, double dt, np.ndarray[int, ndim=1] idx_obs, np.ndarray[double, ndim=1] lower_bounds, np.ndarray[double, ndim=1] upper_bounds):

    cdef np.ndarray[double, ndim=2] population, objectives, new_objectives
    cdef np.ndarray[int, ndim=1] ranks, next_population_indices, current_front, selected_indices, sorted_indices
    cdef int npar, num_to_regenerate, i, current_size, remaining_space, generation
    cdef np.ndarray[double, ndim=1] crowding_distances

    population, lower_bounds, upper_bounds = initialize_population(population_size, lower_bounds, upper_bounds)
    npar = population.shape[1]
    objectives = np.empty((population_size, 3))

    for i in range(population_size):
        simulation = model_simulation(E, dt, Obs[0], idx_obs, population[i])
        objectives[i] = obj_func(Obs, simulation, index_metrics)

    num_to_regenerate = int((regeneration_rate * population_size) + 0.5)

    for generation in range(num_generations):
        ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
        next_population_indices = np.empty(0, dtype=np.int32)
        current_size = 0
        i = 0
        while i < population_size and front_sizes[i] > 0:
            current_front = front_indices[i, :front_sizes[i]]
            crowding_distances = calculate_crowding_distance(objectives, current_front.astype(np.int32))
            sorted_indices = np.argsort(-crowding_distances).astype(np.int32)
            selected_indices = current_front[sorted_indices]
            if current_size + len(selected_indices) > population_size - num_to_regenerate:
                remaining_space = population_size - num_to_regenerate - current_size
                next_population_indices = np.concatenate((next_population_indices, selected_indices[:remaining_space]))
                break
            next_population_indices = np.concatenate((next_population_indices, selected_indices))
            current_size += len(selected_indices)
            i += 1

        mating_pool = population[next_population_indices.astype(np.int32)]
        offspring = crossover(mating_pool, npar, cross_prob, lower_bounds, upper_bounds)
        offspring = polynomial_mutation(offspring, mutation_rate, npar, lower_bounds, upper_bounds)

        new_individuals, lower_bounds, upper_bounds = initialize_population(num_to_regenerate, lower_bounds, upper_bounds)
        offspring = np.concatenate((offspring, new_individuals))
        new_objectives = np.empty_like(objectives)
        for i in range(population_size):
            simulation = model_simulation(E, dt, Obs[0], idx_obs, offspring[i])
            new_objectives[i] = obj_func(Obs, simulation, index_metrics)

        population = offspring
        objectives = new_objectives

        if generation % 100 == 0:
            print(f"Generation {generation} of {num_generations} completed")
        
    return population, objectives



