from time import time
from gurobipy import *
import numpy as np
from copy import deepcopy

def model_bpp(c, w, opt, UB=None, bin_for_item=None, LogToConsole=False, TimeLimit=60):
    t_start = time()
    n = len(w)
    LB = opt
    if UB == None:
        UB = n
    if LogToConsole:
        print('c =', c, '| n =', n, '| LB =', LB, '| UB =', UB)
    model = Model()
    model.params.LogToConsole = LogToConsole
    model.params.TimeLimit = TimeLimit # seconds
    x = model.addVars(n, UB, vtype=GRB.BINARY)
    y = model.addVars(UB, vtype=GRB.BINARY)
    model.setObjective(quicksum(y[j] for j in range(UB)), GRB.MINIMIZE) # minimize the number of bins used
    model.addConstrs(quicksum(x[i,j] for j in range(UB)) == 1 for i in range(n)) # each item in exactly one bin
    model.addConstrs(quicksum(w[i] * x[i,j] for i in range(n)) <= c * y[j] for j in range(UB))
                                                                  # limit total weight in each bin; also, link $x_{ij}$ with $y_j$
    if bin_for_item != None:
        for i in range(n):
            x[i, bin_for_item[i]].start = 1
    model.optimize()
    bin_for_item = [-1 for i in range(n)]
    for i in range(n):
        for j in range(UB):
            if x[i,j].X > 0.5:
                bin_for_item[i] = j
    return model.ObjVal, model.ObjBound, time()-t_start,bin_for_item

def Initialize(Wj, n, c, InitBFD):
    sg = []
    if InitBFD:
        sg,Wjsort = BFD(n,c,Wj)
        Wj = Wjsort

    else:
        for i in range(n) : 
            sg.append([i])
    sm = sum(Wj)
    pile = [ {'n': 0, 'l': [[0]]} ]
    if sm%c == 0 : lbg = sm//c 
    else: lbg = sm//c + 1
    return len(sg),sg,lbg,pile,Wj


def ConcatBins(node):
    conc = []
    for i in node['l']:
        conc.extend(i)
    return conc


def NextObject(Wj, n, node, conc):
    minindex = None
    for i in range(n):
        if i not in conc:
            if not minindex: minindex = i
            if i > node['n']:
                return i,minindex
    return None ,minindex

def ExceedCapacity(Wj, c, node):
    sm = 0
    for i in node['l'][-1]:
        sm += Wj[i]
    if sm > c:
        return True
    return False

def Lb(Wj, n, c, node, conc):
    sm = 0
    for i in range(n):
        if i not in conc:
            sm += Wj[i]
    return len(node['l'])+sm/c

def BinNotFull(Wj, n, c, conc, node):
    sm = 0
    for i in node['l'][-1]:
        sm += Wj[i]
    for i in range(n):
        if i not in conc and sm+Wj[i]<=c :
            return True
    return False


#Best-Fit-Decreasing
def BFD(n,c,w):
    if n == 0:
        return 0

    wSorted = w.copy()
    wSorted.sort(reverse = True)
    return BF(n, c, wSorted),wSorted
# Best-Fit
def BF(n, c, w):
    if n == 0:
        return 0
    Result = [[]]
    Bins = [c]
    for i in range(n):
        RC = []
        RCind = []
        for j in range(len(Bins)):
            if w[i] <= Bins[j]:
                RC.append(Bins[j] - w[i])
                RCind.append(j)
        if len(RC) > 0:
            a = RCind[np.argmin(RC)]
            Bins[a] -= w[i]
            Result[a].append(i)
        else:
            Bins.append(c)
            Bins[len(Bins)-1] -= w[i]
            Result.append([])
            Result[-1].append(i)
    return (Result)

def BinPackingBB(Wj, n, c, opt, TimeLimit=200, InitBFD=False):
    t = time()
    ubg, sg, lbg ,pile, Wj = Initialize(Wj, n, c, InitBFD)
    i = 0
    print("=================================================================================")
    print("Les objets : ",Wj)
    print("\n=================================================================================")
    print("Solution Inital : {}  {}  {}".format(sg,ubg,round(time()-t,4)))
    if(ubg == lbg):
        pile = []
    print("\n=================================================================================")
    while pile :
        if(time()-t >= TimeLimit):
            print("=================================================================================")
            print("TL reached, meilleure solution trouvee : {} | {} | gap : {} % | temps : {} \n".format(sg,ubg,round( (ubg-opt)*100/ubg,3 ),round(time()-t,4)))
            return ubg,round(time()-t,4)
        elif(ubg == opt):
            print("=================================================================================")
            print("Solution optimale : {} | {} | gap : {} % | temps : {} \n".format(sg,ubg,round( (ubg-opt)*100/ubg,3 ),round(time()-t,4)))
            return ubg,round(time()-t,4)
        node = pile.pop()
        conc = ConcatBins(node)
        if ((ExceedCapacity(Wj, c, node)) or (Lb(Wj, n, c, node, conc) >= ubg)):
            continue
            
        else:
            if len(conc) == n:
                sg = node['l']
                ubg = len(node['l'])
                i += 1
                #print("Solution #{} : {}  {}  {} \n".format(i,sg,ubg,round(time()-t,4)))
                if ubg == lbg : 
                    break
            else:
                j, minindex = NextObject(Wj, n, node, conc)
                if not j:
                    if BinNotFull(Wj, n, c, conc, node):
                        continue
                    son = {
                        'n': minindex,
                        'l': node['l'],
                    }
                    son['l'].append([minindex])
                    pile.append(son)

                else :
                    sons = [{   
                        'n': j,
                        'l': deepcopy(node['l']),
                    },  
                    {
                        'n': j,
                        'l': node['l'],

                    }]

                    sons[1]['l'][-1].append(j)
                    pile.extend(sons)
    
    print("=================================================================================")
    print("Solution optimale : {} | {} | gap : {} % | temps : {} \n".format(sg,ubg,round( (ubg-opt)*100/ubg,3 ),round(time()-t,4)))
    return ubg,round( (ubg-opt)*100/ubg,3 ),round(time()-t,4)