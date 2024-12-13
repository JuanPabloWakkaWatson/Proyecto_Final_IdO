#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 07:29:53 2024

@author: erick
"""

import math
import numpy as np
import time
import random
from collections import deque

def read_coordinates(file_path):
    """
    Lee las coordenadas de las ciudades desde un archivo y las retorna como un diccionario.
    """
    cities = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith("Qatar"):
                parts = line.split()
                city_id = int(parts[0])
                x, y = map(float, parts[1:])
                cities[city_id] = (x, y)
    return cities

def calculate_distance(coord1, coord2):
    """
    Calcula la distancia euclidiana redondeada entre dos puntos.
    """
    return int(round(math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)))

def create_distance_matrix(cities):
    """
    Crea una matriz de distancias entre todas las ciudades.
    """
    n = len(cities)
    distance_matrix = np.zeros((n, n), dtype=int)
    city_ids = list(cities.keys())
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = calculate_distance(cities[city_ids[i]], cities[city_ids[j]])
            else:
                distance_matrix[i][j] = 0  # Usar 0 para la diagonal principal
    return distance_matrix, city_ids

def nearest_neighbor_heuristic(distance_matrix, city_ids):
    """
    Aplica la heurística del vecino más cercano para aproximar la ruta del agente viajero.
    """
    n = len(city_ids)
    visited = [False] * n
    current_city = 0  # Empezar desde la primera ciudad
    route = [city_ids[current_city]]
    visited[current_city] = True
    total_distance = 0

    for _ in range(n - 1):
        nearest_city = None
        min_distance = float('inf')
        for next_city in range(n):
            if not visited[next_city] and distance_matrix[current_city][next_city] < min_distance:
                nearest_city = next_city
                min_distance = distance_matrix[current_city][next_city]

        route.append(city_ids[nearest_city])
        total_distance += min_distance
        visited[nearest_city] = True
        current_city = nearest_city

    # Volver a la ciudad inicial
    total_distance += distance_matrix[current_city][0]
    route.append(city_ids[0])

    return route, total_distance

def two_opt_heuristic(distance_matrix, city_ids):
    """
    Aplica la heurística 2-opt para mejorar una ruta inicial.
    """
    def calculate_total_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # Volver al inicio
        return total_distance

    # Ruta inicial (orden natural de las ciudades)
    n = len(city_ids)
    route = list(range(n))
    best_distance = calculate_total_distance(route)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Generar una nueva ruta invirtiendo la sección entre i y j
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_distance = calculate_total_distance(new_route)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True

    # Convertir índices a IDs de ciudades
    final_route = [city_ids[i] for i in route]
    return final_route, best_distance


def tabu_search(distance_matrix, city_ids, max_iterations=1000, tabu_tenure=50):
    """
    Implementa la búsqueda tabú para resolver el problema del agente viajero.
    """
    def calculate_total_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # Volver al inicio
        return total_distance

    def two_opt_swap(route, i, j):
        """
        Realiza una inversión de segmento (2-opt) entre los índices i y j.
        """
        new_route = route[:]
        new_route[i:j + 1] = reversed(new_route[i:j + 1])
        return new_route

    # Generar una solución inicial
    current_solution = list(range(len(city_ids)))
    current_distance = calculate_total_distance(current_solution)
    best_solution = current_solution[:]
    best_distance = current_distance

    # Inicializar la lista tabú
    tabu_list = deque(maxlen=tabu_tenure)

    for iteration in range(max_iterations):
        neighborhood = []

        # Generar vecinos usando 2-opt
        for i in range(len(current_solution) - 1):
            for j in range(i + 1, len(current_solution)):
                neighbor = two_opt_swap(current_solution, i, j)
                distance = calculate_total_distance(neighbor)
                if (i, j) not in tabu_list:
                    neighborhood.append((neighbor, distance, (i, j)))

        # Seleccionar el mejor vecino que no esté en la lista tabú
        neighborhood.sort(key=lambda x: x[1])  # Ordenar por distancia
        best_neighbor, best_neighbor_distance, move = neighborhood[0]

        # Actualizar la solución actual
        current_solution = best_neighbor
        current_distance = best_neighbor_distance

        # Actualizar la mejor solución encontrada
        if current_distance < best_distance:
            best_solution = current_solution[:]
            best_distance = current_distance

        # Agregar el movimiento a la lista tabú
        tabu_list.append(move)

        '''
        # Monitorear el progreso
        if iteration % 100 == 0 or iteration == max_iterations - 1:
            print(f"Iteración {iteration}: Mejor distancia = {best_distance}")
        '''

    # Convertir índices a IDs de ciudades
    final_route = [city_ids[i] for i in best_solution]
    return final_route, best_distance

def nearest_neighbor_solution(distance_matrix):
    """
    Genera una solución inicial usando la heurística del vecino más cercano.
    """
    n = len(distance_matrix)
    unvisited = list(range(n))
    route = [unvisited.pop(0)]  # Comenzar desde la primera ciudad

    while unvisited:
        last_city = route[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[last_city][city])
        route.append(next_city)
        unvisited.remove(next_city)

    return route

def simulated_annealing(distance_matrix, city_ids, initial_temperature=1000000, cooling_rate=0.9999, max_iterations=100000000):
    """
    Implementa recocido simulado para resolver el problema del agente viajero.
    """
    def calculate_total_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # Volver al inicio
        return total_distance

    def two_opt_swap(route):
        """
        Realiza un intercambio de segmento (2-opt) en la ruta.
        """
        i, j = sorted(random.sample(range(len(route)), 2))
        new_route = route[:]
        new_route[i:j + 1] = reversed(new_route[i:j + 1])
        return new_route

    # Generar solución inicial
    current_solution = list(range(len(city_ids)))
    random.shuffle(current_solution)
    current_distance = calculate_total_distance(current_solution)

    best_solution = current_solution[:]
    best_distance = current_distance

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generar un vecino
        new_solution = two_opt_swap(current_solution)
        new_distance = calculate_total_distance(new_solution)

        # Calcular cambio de energía
        delta_energy = new_distance - current_distance

        # Decidir si aceptar el vecino
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_solution = new_solution
            current_distance = new_distance

            # Actualizar la mejor solución
            if current_distance < best_distance:
                best_solution = current_solution[:]
                best_distance = current_distance

        # Enfriar la temperatura
        temperature *= cooling_rate

        '''
        # Monitorear el progreso
        if iteration % 1000 == 0 or iteration == max_iterations - 1:
            print(f"Iteración {iteration}: Mejor distancia = {best_distance}", end="\r")
        '''
        
        # Terminar si la temperatura es muy baja
        if temperature < 1e-3:
            break

    # Convertir índices a IDs de ciudades
    final_route = [city_ids[i] for i in best_solution]
    return final_route, best_distance

def main():
    # Archivo con las coordenadas de las ciudades
    file_path = 'Qatar.txt'
    cities = read_coordinates(file_path)

    # Crear la matriz de distancias
    distance_matrix, city_ids = create_distance_matrix(cities)

    # Aplicar la heurística del vecino más cercano
    
    start = time.time()
    route, total_distance = nearest_neighbor_heuristic(distance_matrix, city_ids)
    end = time.time()

    print("Ruta aproximada (Vecino mas cercano):", route)
    print("Distancia total aproximada (Vecino mas cercano):", total_distance)
    print("Tiempo de ejecución (Vecino mas cercano):", end - start)
    
    # Aplicar el algoritmo 2opt
    
    start = time.time()
    route, total_distance = two_opt_heuristic(distance_matrix, city_ids)
    end = time.time()

    print("Ruta mejorada (2-opt):", route)
    print("Distancia total mejorada (2-opt):", total_distance)
    print("Tiempo de ejecución (2-opt):", end - start)
    
    # Aplicar busqueda tabu
    
    '''
    start = time.time()
    route, total_distance = tabu_search(distance_matrix, city_ids)
    end = time.time()

    print("Ruta aproximada (Busqueda Tabu):", route)
    print("Distancia total aproximada (Busqueda Tabu):", total_distance)
    print("Tiempo de ejecución (Busqueda Tabu):", end - start)
    '''
    
    # Aplicar recocido simulado
    
    start = time.time()
    route, total_distance = simulated_annealing(distance_matrix, city_ids)
    end = time.time()

    print("Ruta aproximada (Recocido simulado):", route)
    print("Distancia total aproximada (Recocido simulado):", total_distance)
    print("Tiempo de ejecución (Recocido simulado):", end - start)
    

if __name__ == "__main__":
    main()
