from time import time
import pandas as pd
import itertools
from collections import namedtuple
import numpy as np
from random import *
from gurobipy import *
import os

Item = namedtuple("Item", ['id', 'size'])
Candidate = namedtuple("Candidate", ['items', 'fitness'])

#------------BIN-------------
def cost(bins):
    return len(bins)


class Bin(object):
    count = itertools.count()

    def __init__(self, capacity):
        self.id = next(Bin.count)
        self.capacity = capacity
        self.free_space = capacity
        self.items = []
        self.used_space = 0

    def add_item(self, item):
        self.items.append(item)
        self.free_space -= item.size
        self.used_space += item.size

    def remove_item(self, item_index):
        item_to_remove = self.items[item_index]
        del self.items[item_index]
        self.free_space += item_to_remove.size
        self.used_space -= item_to_remove.size

    def fits(self, item):
        return self.free_space >= item.size

    def __str__(self):
        items = [str(it) for it in self.items]
        items_string = '[' + ' '.join(items) + ']'
        return "Bin n° " + str(self.id) + " containing the " + \
               str(len(self.items)) + " following items : " + items_string + \
               " with " + str(self.free_space) + " free space."

    def __copy__(self):
        new_bin = Bin(self.capacity)
        new_bin.free_space = self.free_space
        new_bin.used_space = self.used_space
        new_bin.items = self.items[:]
        return new_bin
    
#-----------GA HEUR-----------

def bestfit(items, current_bins, capacity):
    bins = [copy.copy(b) for b in current_bins]
    if not bins:
        bins = [Bin(capacity)]
    for item in items:
        if item.size > capacity:
            continue
        possible_bins = [bin for bin in bins if bin.fits(item)]
        if not possible_bins:
            bin = Bin(capacity)
            bin.add_item(item)
            bins.append(bin)
        else:
            index, free_space = min(enumerate(possible_bins), key=lambda it: it[1].free_space)
            possible_bins[index].add_item(item)
    return bins

def firstfit(items, current_bins, capacity):
    bins = [copy.copy(b) for b in current_bins]
    if not bins:
        bins = [Bin(capacity)]
    for item in items:
        if item.size > capacity:
            continue
        first_bin = next((bin for bin in bins if bin.free_space >= item.size), None)
        if first_bin is None:
            bin = Bin(capacity)
            bin.add_item(item)
            bins.append(bin)
        else:
            first_bin.add_item(item)
    return bins

#Generation de la population
def population_generator(items, capacity, population_size, greedy_solver):
    candidate = Candidate(items[:], fitness(items, capacity, greedy_solver))
    population = [candidate]
    new_items = items[:]
    for i in range(population_size - 1):
        shuffle(new_items)
        candidate = Candidate(new_items[:], fitness(new_items, capacity, greedy_solver))
        if candidate not in population:
            population.append(candidate)
    return population
#Fitness function
def fitness(candidate, capacity, greedy_solver):
    if greedy_solver == 'FF':
        return firstfit(candidate,[], capacity)
    elif greedy_solver == 'BF':
        return bestfit(candidate,[], capacity)
    return nextfit(candidate,[], capacity)

# Méthodes de selection
# 1 - Tournament selection 
def tournament_selection(population, tournament_selection_probability, k):
    # k : nb individus selectionnés pour jouer le tournoi entre eux , séléctionnés aléatoirement
    candidates = [population[(randint(0, len(population) - 1))]]
    # remplir les candidats
    while len(candidates) < k:
        new_indiv = population[(randint(0, len(population) - 1))]
        if new_indiv not in candidates:
            candidates.append(new_indiv)
    #Tester
    ind = int(np.random.geometric(tournament_selection_probability, 1))
    # Si aucun individu obient un succés
    while ind >= k:
        ind = int(np.random.geometric(tournament_selection_probability, 1))
    # retourner l'individu qui a succédé
    return candidates[ind]

# 2 - Roulette Wheel Selection
# Roulette wheel selection :proba d'etre selectionné = proba d'etre selection (fitness) / somme de probas des N individus

def roulette_wheel_selection(population):
    max = sum([len(e.fitness) for e in population])
    # generer un nb aleatoire entre 0 et max
    pick = uniform(0, max)
    current = max
    # chercher l'individu qui convient 
    for item in population:
        current -= len(item.fitness)
        if current < pick:
            return item
        
# 3 - Rank selection
# Rank selection : Ponderer par le classement en non par le fitness score, 
   # Meilleur => deuxième meilleur => troisième meilleur...

def rank_selection(population):
    length = len(population)
    rank_sum = length * (length + 1) / 2
    pick = uniform(0, rank_sum)
    current = 0
    for item in population:
        current += length
        if current > pick:
            return item
        length -= 1
# 4 - SUS : Stochastic universal sampling
# SUS : Stochastic universal sampling c'est une amélioration de RWS
# n : nombre d'individus a selectionner depuis la population
def SUS(population, n):
    selected = []
    pointers = []
    max = sum([len(e.fitness) for e in population])
    # distance : pas de selection = Sum / n
    distance = max / n
    # Start : premier placement du pointeur entre 0 et pas de selection ( distance )
    start = uniform(0, distance)
    for i in range(n):
        pointers.append(start + i * distance)
    # parcourir les pointeurs: souvent il n y en a qu'un seul
    for pointer in pointers:
        current = 0
        # parcourir les candidats pour chaque pointeur
        for item in population:
            current += len(item.fitness)
            # Si on obtient un score depassant ce pointeur alors selectionner l'item
            if current > pointer:
                selected.append(item)
    # Retourner les items selectionnés
    return selected
#CrossOver:
#Parcourir les deux candidats simultanément et prendre l'element de chaque candidat non encore pris
def crossover(parent1, parent2):
    taken = [False] * len(parent1)
    child = []
    i = 0
    while i < len(parent1):
        element1 , element2 = parent1[i], parent2[i]
        if not taken[element1.id]:
            child.append(element1)
            taken[element1.id] = True
        if not taken[element2.id]:
            child.append(element2)
            taken[element2.id] = True
        i += 1
    return child
# Add mutation
def mutation(candidate, capacity, greedy_solver):
    candidate_items = candidate.items
    a = randint(0, len(candidate_items) - 1)
    b = randint(0, len(candidate_items) - 1)
    while a == b:
        b = randint(0, len(candidate_items) - 1)
    # Permutation
    candidate_items[a] , candidate_items[b] = candidate_items[b] , candidate_items[a]
    candidate = Candidate(candidate_items, fitness(candidate_items, capacity, greedy_solver))
    return candidate

def cand_to_list(cand):
    l = []
    for e in cand:
        l.append(e.size)
    return l

## Tabu searc

def order(s):
    ind = []
    unique_el, counts_el = np.unique(s, return_counts=True)
    l = counts_el.argsort()
    unique_el = unique_el[l]
    counts_el = counts_el[l]
    for i in unique_el:
        ind.extend(list((s==i).nonzero()[0]))
    return ind



def Best_Neighbor(s, f, tabu_list, n, Wj, c):
    ind = order(s)
    for i in ind:
        for j in range(1,f+1):
            if s[i] != j:
                if Wj[i]+Wj[s==j].sum() <= c:
                    cs = s.copy()
                    cs[i] = j
                    if cs[cs==s[i]].size == 0:
                        for k in range(s[i]+1,f+1):
                            cs[cs==k] -= 1
                    if list(cs) not in tabu_list:
                        return cs
    stop = True
    while stop:
        obj = np.random.randint(0,n,2)
        if s[obj[0]] != s[obj[1]] :
            cs = s.copy()
            os = cs[obj[0]]
            cs[obj[0]] = cs[obj[1]]
            cs[obj[1]] = os
            if Wj[cs==cs[obj[0]]].sum() <= c and Wj[cs==cs[obj[1]]].sum() <= c and list(cs) not in tabu_list:
                return cs
            
def Add_Tabu(s, tabu_list, t):
    tabu_list.append(list(s))
    if len(tabu_list) == t+1 :
        tabu_list.pop(0)
    
def Initialize(n, Wj, c, Init_FF):
    if Init_FF : s = FF(n, Wj, c) 
    else : s = np.arange(1,n+1,1)
    f = s.max()
    w = np.array(Wj)
    cs = s.copy()
    tabu_list = [list(cs)]
    sm = w.sum()
    if sm%c == 0 : lb = sm//c 
    else: lb = sm//c + 1
    return s, f, w, cs, tabu_list, lb

def FF(n , Wj, c):
    w = np.array(Wj)
    res = np.full(w.shape, 0)
    if n == 0 :
        return 0
    if w[w>c] :
        return False
    for i in range(n):
        for j in range(1,res.max()+1):
            if w[i]+w[res==j].sum() <= c:
                res[i] = j
                break
        if res[i] == 0 :
            res[i] = res.max()+1
    return res

def TS_BPP(n, Wj, c, N_tabu=5, Nb_iter=100, Init_FF=True):
    
    t = time()
    s, f, w, cs, tabu_list, lb = Initialize(n, Wj, c, Init_FF)
    if f == lb :
        s2 = [[]]
        s2 = [[] for i in range(0,s.max())] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
        for i in range(0,n):
            s2[s[i]-1].append(Wj[i])
        return s, s2, time()-t
  
    i = 0
    while i < Nb_iter:
        cs = Best_Neighbor(cs, cs.max(), tabu_list, n, w, c)
        if cs.max() < f:
            s = cs.copy()
            f = cs.max()
            j = 0
            if f == lb :
                s2 = [[]]
                s2 = [[] for i in range(0,s.max())] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
                for i in range(0,n):
                    s2[s[i]-1].append(Wj[i])
                return s, s2, time()-t
        Add_Tabu(cs, tabu_list, N_tabu)
        i += 1
    s2 = [[]]
    s2 = [[] for i in range(0,s.max())] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
    for i in range(0,n):
        s2[s[i]-1].append(Wj[i])
    
    return s,s2,time()-t

def cand_to_list(cand):
    l = []
    for e in range(len(cand)):
        l.append(cand[e].size)
    return l
def print_solution(solution):
    for e in range(len(solution)):
        print(solution[e])
    print(len(solution))

def hybrid_LTH_1(weights, capacity, mutate=True, population_size=20, generations=20, k=2, tournament_selection_probability=0.7, crossover_probability=1, mutation_probability=0.4, greedy_solver='FF', allow_duplicate_parents=False, selection_method='TS'):
    start_time = time()
    # Recuperation de la liste des objets items
    items = [Item]
    items = [ Item(i,weights[i]) for i in range(len(weights))]
    # Generer une population
    population = population_generator(items, capacity, population_size, greedy_solver)
    # Recupere le score de la liste initiale
    best_solution = fitness(items, capacity, greedy_solver)
    best_comb = items
    i = 0
    # Création des générations et résolution
    for i_gen in range(generations):
        new_generation = []
        best_child = best_solution
        best_comb1 = best_comb
        for j_pop in range(population_size):
            # Selection du candidat (meilleur parent) selon l'une des méthodes suivantes
            if selection_method == 'TS':
                # Tournament selection
                first_parent = tournament_selection(population, tournament_selection_probability, k).items
                second_parent = tournament_selection(population, tournament_selection_probability, k).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = tournament_selection(population, tournament_selection_probability, k).items
            
            elif selection_method == 'RW':
                # Roulette wheel steeling
                first_parent = roulette_wheel_selection(population).items
                second_parent = roulette_wheel_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = roulette_wheel_selection(population).items
            
            elif selection_method == 'RS':
                # Rank selection
                first_parent = rank_selection(population).items
                second_parent = rank_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = rank_selection(population).items
            
            elif selection_method == 'SUS':
                # Stochastic universal sampling 
                first_parent = SUS(population, 1)[0].items
                second_parent = SUS(population, 1)[0].items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = SUS(population, 1)[0].items
            else:
                return
            # Crossover entre les deux meilleurs parents et obtention du fils - child -
            prob = random()
            if prob <= crossover_probability:
                child_items = crossover(first_parent, second_parent)
                child = cand_to_list(child_items)
                sol_TS = []
                sol_TS = TS_BPP(len(child),child,capacity)[1]
                child_list = []
                for e_solts in range(len(sol_TS)):
                    for z_solts in range(len(sol_TS[e_solts])):
                        child_list.append(sol_TS[e_solts][z_solts])
                child = [Item]
                child = [ Item(z,child_list[z]) for z in range(len(child_list))]
                child = Candidate(child[:], fitness(child, capacity, greedy_solver))
            else:
                child = first_parent
                child = cand_to_list(child)
                sol_TS = []
                sol_TS = TS_BPP(len(child),child,capacity)[1]
                child_list = []
                for e_solts in range(len(sol_TS)):
                    for z_solts in range(len(sol_TS[e_solts])):
                        child_list.append(sol_TS[e_solts][z_solts])
                child = [Item]
                child = [ Item(z,child_list[z]) for z in range(len(child_list))]
                child = Candidate(child[:],fitness(child,capacity, greedy_solver))
            # Generer un nb aléa entre 0 et 1
            prob = random()
            if ( mutate == True and  prob <= mutation_probability):
            # Si le nombre est <= a proba de mutation alors muter le fils
                child = mutation(child, capacity, greedy_solver)
            # Si le fils convient mieux que le meilleur résultat obtenu précedemment alors MAJ du meilleur resultat
            if len(child.fitness) < len(best_child):
                best_child = child.fitness
                best_comb_1 = child.items
            # Ajouter le fils a la nouvelle generation meme s'il n'ameliore pas le resultat
            new_generation.append(child)
        # MAJ de la meilleure solution
        if len(best_child) < len(best_solution):
            best_solution = best_child
            best_comb = best_comb_1
        # MAJ de la generation :  passage de generation i a i+1
        population = [Candidate(p.items[:], p.fitness) for p in new_generation]
        # Tri des candidats selon leur score de fitness
        population.sort(key=lambda candidate: len(candidate.fitness), reverse=True)
    # Retoruner la meilleure solution
    return best_solution, best_comb, time()-start_time
    
def hybrid_LTH_2(weights, capacity, mutation=True, population_size=20, generations=20, k=2, tournament_selection_probability=0.7, mutation_probability=0.4, greedy_solver='FF', allow_duplicate_parents=False, selection_method='TS'):
    start_time = time()
    # Recuperation de la liste des objets items
    items = [Item]
    items = [ Item(i,weights[i]) for i in range(len(weights))]
    # Generer une population
    population = population_generator(items, capacity, population_size, greedy_solver)
    # Recupere le score de la liste initiale
    best_solution = fitness(items, capacity, greedy_solver)
    best_comb = items
    i = 0
    # Création des générations et résolution
    for i in range(generations):
        new_generation = []
        best_child = best_solution
        best_comb1 = best_comb
        for j in range(population_size):
            # Selection du candidat (meilleur parent) selon l'une des méthodes suivantes
            if selection_method == 'TS':
                # Tournament selection
                first_parent = tournament_selection(population, tournament_selection_probability, k).items
                second_parent = tournament_selection(population, tournament_selection_probability, k).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = tournament_selection(population, tournament_selection_probability, k).items
            
            elif selection_method == 'RW':
                # Roulette wheel steeling
                first_parent = roulette_wheel_selection(population).items
                second_parent = roulette_wheel_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = roulette_wheel_selection(population).items
            
            elif selection_method == 'RS':
                # Rank selection
                first_parent = rank_selection(population).items
                second_parent = rank_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = rank_selection(population).items
            
            elif selection_method == 'SUS':
                # Stochastic universal sampling 
                first_parent = SUS(population, 1)[0].items
                second_parent = SUS(population, 1)[0].items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = SUS(population, 1)[0].items
            else:
                return
            # Crossover entre les deux meilleurs parents et obtention du fils - child -
            child = first_parent
            prob = random()
            if(prob <= 0.2):
                child = cand_to_list(child)
                sol_TS = []
                sol_TS = TS_BPP(len(child),child,capacity)[1]
                child_list = []
                for e_solts in range(len(sol_TS)):
                    for z_solts in range(len(sol_TS[e_solts])):
                        child_list.append(sol_TS[e_solts][z_solts])
                child = [Item]
                child = [ Item(z_solts,child_list[z_solts]) for z_solts in range(len(child_list))]
                child = Candidate(child[:], fitness(child, capacity, greedy_solver))
            else:
                child = Candidate(child[:], fitness(child, capacity, greedy_solver))
            # Generer un nb aléa entre 0 et 1
            prob = random()
                #child = mutation(child, capacity, greedy_solver)
            # Si le fils convient mieux que le meilleur résultat obtenu précedemment alors MAJ du meilleur resultat
            if len(child.fitness) < len(best_child):
                best_child = child.fitness
                best_comb_1 = child.items
            # Ajouter le fils a la nouvelle generation meme s'il n'ameliore pas le resultat
            new_generation.append(child)
        # MAJ de la meilleure solution
        if len(best_child) < len(best_solution):
            best_solution = best_child
            best_comb = best_comb_1
        # MAJ de la generation :  passage de generation i a i+1
        population = [Candidate(p.items[:], p.fitness) for p in new_generation]
        # Tri des candidats selon leur score de fitness
        population.sort(key=lambda candidate: len(candidate.fitness), reverse=True)
    # Retoruner la meilleure solution
    return best_solution, best_comb, time()-start_time

def show_sol(c,sol,opt,details):
    a = len(sol[0])
    print('nombre de bins : ',a)
    print('Optimum : ',opt)
    print('Gap : ',round( (a-opt)*100/a,2),' % ')
    print('temps : ', sol[2],' s')
    if(details == True):
        r = sol[0]
        print('********************* solution **********************')
        for e in range(len(r)):
            print(r[e])
        print('********************* Combinaison *********************')
        print(sol[1])