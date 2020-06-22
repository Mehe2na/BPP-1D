import numpy as np
import pandas as pd
from random import randint,choice,random
from time import time

def firstFit(C, Weights):
    bins = []
    bins.append(C)
    solution = []

    for j in range(len(Weights)):
        i = 0
        while i < len(bins):
            if bins[i] >=Weights[j]:
                bins[i] -= Weights[j]
                solution.append(i+1)
                break
            i+=1
        if i == len(bins):
            bins.append(C-Weights[j])
            solution.append(i+1)

    return solution


def nb_bins(Configuration, Weights): # renvoit le nombre de boites utilisés dans une configuration
    s=pd.Series(Configuration)
    w=pd.Series(Weights)
    return (w.groupby(s).sum()**2).sum()
  

def creerBins(Configuration, Weights,C): #crée l'ensemble des bins avec les poids suivant la configuration
                             #Retourner une liste de poids restants des bins
    PoidsRes = [C]*max(Configuration)
    for i in range(len(Configuration)):
        if PoidsRes[Configuration[i]-1] >= Weights[i]:
            PoidsRes[Configuration[i]-1] -= Weights[i]
        else :
            return False
    return PoidsRes


def voisin_Insertion(solution, Weights, C): #Voisin par insertion 
    #relocalisation d'un seul élément sélectionné au hasard dans le bin auquel il est actuellement alloué dans un bin sélectionné au hasard
    
    Configuration=solution.copy()
    i=randint(0,len(Configuration)-1) #Selectionne un objet au hasard
    binSelectionne = Configuration[i]   #Sauvegarder son numéro de bin 
    PoidsRestants = creerBins(Configuration,Weights,C)  #créer les bins correspondants à la configuration
    PoidsRestants[Configuration[i] - 1] += Weights[i] #enlever l'objet choisis précedement du bin correspondant
    choices = list(range(0,len(PoidsRestants))) #créer un tableau qui contient la liste de choix des bin
    choices.remove(binSelectionne-1)
    m=choice(choices)   
    for k in range(len(PoidsRestants)-2):
        if PoidsRestants[m] < Weights[i]:  #TQ il n'ya pas de la place pour l'objet i dans le bin choisis au hasard j
            choices.remove(m)
            m=choice(choices)
        else:
            break

    if k == len(PoidsRestants)-3 : m = binSelectionne-1
    Configuration[i] = m+1
    return truncConf(Configuration)


def voisin_Swap(solution, Weights, C): #Voisin par swap
    #sélectionner au hasard deux éléments actuellement attribués à deux bins différents et échanger leurs postes
    
        Configuration=solution.copy()
        i=randint(0,len(Configuration)-1) 
        j=randint(0,len(Configuration)-1)
        PoidsRestants = creerBins(Configuration,Weights,C)  #créer les bins correspondants à la configuration
        PoidsRestants[Configuration[i] - 1] += Weights[i]
        PoidsRestants[Configuration[j] - 1] += Weights[j]
        if PoidsRestants[Configuration[i] - 1] >= Weights[j] and PoidsRestants[Configuration[j] - 1] >= Weights[i]:
            z = Configuration[i]
            Configuration[i] = Configuration[j]
            Configuration[j] = z
        return Configuration

def voisin(solution, Weights,C):
    return voisin_Insertion(solution,Weights, C) if random() < 0.8 else voisin_Swap(solution, Weights, C)


def truncConf(Configuration):  #this function will trunc a conf exemple : [2,3,4] will becam [1,2,3] or [1,22,4,4] will becam [1,2,3,3]
    s = pd.Series(Configuration)
    uniques = s.unique()
    return s.map({c:v for c,v in zip(uniques,range(1,len(uniques)+1))}).values


def pourcentageBins(Configuration,Weights, C): 
                             #Retourner une liste contenante le pourcentage d'utilisation de chaque bin'
    Conf = creerBins(Configuration, Weights,C)
    Pourcentage = []
    for i in range(0,(len(Conf))):
        Pourcentage.append( ((C-Conf[i])*100)/C)
        
    return Pourcentage

def Recuit_simule(C,Weights,n,T,nb_iterations):
    start_time = time()
    Configuration = firstFit(C, Weights) #Générer une solution initiale
    nombre_bins = nb_bins(Configuration, Weights) #trouver le nombre de bins utilisés dans la solution
    bestFitness = max(Configuration)
    bestConf = Configuration
    T_ = T
    Configurations = []
    proba_acceptation = []
    for k in range(nb_iterations):
        voisinConfiguration = voisin(Configuration, Weights, C)
        nb_bins_sol_vois = nb_bins(voisinConfiguration, Weights)
      
        if ( nb_bins_sol_vois > nombre_bins) or random() < np.exp(( nb_bins_sol_vois - nombre_bins) / T_):
            Configuration = voisinConfiguration
            nombre_bins =  nb_bins_sol_vois
            if (max(Configuration) < bestFitness):
                bestFitness = max(Configuration)
                bestConf = Configuration
        T_ *= 0.99
    s2 = [[] for i in range(0,max(bestConf))] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
    for i in range(0,n):
        s2[bestConf[i]-1].append(Weights[i])

   
    return bestConf,s2,time()-start_time

def show_sol(c,sol,opt,details):
    a = max(sol[0])
    print('Nombre de bins : ',a)
    print('Optimum : ',opt)
    print('Gap : ',round( (a-opt)*100/a,2),' % ')
    print('temps : ', sol[2],' s')
    if(details == True):
        r = sol[1]
        for e in range(len(r)):
            print('bin ',e+1,' contient les objets : ',r[e],' | espace libre : ',c-sum(r[e]))