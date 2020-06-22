import numpy as np
from time import time
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
def TS_BPP(n, Wj, c, opt, N_tabu, Nb_iter, Init_FF=True):
    
    t = time()
    s, f, w, cs, tabu_list, lb = Initialize(n, Wj, c, Init_FF)
    lb = opt
    if f == lb :
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
                s2 = [[] for i in range(0,s.max())] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
                for i in range(0,n):
                    s2[s[i]-1].append(Wj[i])
                return s, s2, time()-t
        Add_Tabu(cs, tabu_list, N_tabu)
        i += 1
    s2 = [[] for i in range(0,s.max())] # s2 est la solution avec les poids  ex : [[20,731], [10,900], [10000]]                                          # une solution avec 3 bins avec le poids des objets dans les listes 
    for i in range(0,n):
        s2[s[i]-1].append(Wj[i])
    return s, s2, time()-t

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