from time import time
import numpy as np
    #First-Fit
def FF(n, c, w):
    start_time = time()
    if n == 0:
        return 0

    Bins = [c]
    result = [[]]
    for i in range(n):
        nFit = False
        for j in range(len(Bins)):
            if w[i] <= Bins[j]:
                Bins[j] -= w[i]
                result[j].append(w[i])
                break
            if j == len(Bins)-1:
                nFit = True
        if nFit is True:
            Bins.append(c)
            result.append([])
            result[-1].append(w[i])
            Bins[-1] -= w[i]
    return Bins,result,len(Bins),sum(Bins)/(c*len(Bins)),time()- start_time
#Best-Fit
def BF(n, c, w):
    start_time = time()
    if n == 0:
        return 0
    result = [[]]
    Bins = [c]
    for i in range(n):
        RC = []
        RCind = []
        for j in range(len(Bins)):
            if w[i] <= Bins[j]:
                RC.append(Bins[j] - w[i])
                RCind.append(j)
        if len(RC) > 0:
            Bins[RCind[np.argmin(RC)]] -= w[i]
            result[RCind[np.argmin(RC)]].append(w[i])
        else:
            Bins.append(c)
            Bins[len(Bins)-1] -= w[i]
            result.append([])
            result[-1].append(w[i])
    return Bins,result,len(Bins),sum(Bins)/(c*len(Bins)),time() - start_time
#Next-Fit
def NF(n, c, w):
    start_time = time()
    if n == 0:
        return 0

    curBin = c
    nBins = 1
    Bins = [c]
    result = [[]]
    j = 0
    for i in range(len(w)):
        if w[i] <= Bins[j]:
            Bins[j] -= w[i]
            result[j].append(w[j])
        else:
            Bins.append(c)
            j += 1
            result.append([])
            Bins[j] -= w[i]
            result[j].append(w[i])
    return Bins,result,len(Bins),sum(Bins)/(c*len(Bins)), time() - start_time

#Max-Rest <=>  Worst Fit
def MR(n, c, w):
    start_time = time()
    if n==0:
        return 0
    
    Bins = [c]
    result = [[]]
    for i in range(n):
        k = np.argmax(Bins)
        if w[i] <= Bins[k]:
            Bins[k] -= w[i]
            result[k].append(w[i])
        else:
            Bins.append(c)
            Bins[len(Bins)-1] -= w[i]
            result.append([])
            result[-1].append(w[i])
    return Bins,result,len(Bins),sum(Bins)/(c*len(Bins)),time() - start_time
#Max-Rest-Heap
def MRH(n,c,w):
    start_time = time()
    if n == 0:
        return 0
    nb_bins = 1
    import heapq
    Bins = [c]
    heapq.heapify(Bins)
    cap = heapq.heappop(Bins)
    for i in range(n-1):
        if w[i] + cap <= 2*c:
            cap = heapq.heappushpop(Bins,cap + w[i])
        else:
            heapq.heappush(Bins,cap)
            cap = heapq.heappushpop(Bins,c+w[i])
            nb_bins += 1
    i += 1
    if w[i] + cap <= 2*c:
        cap = heapq.heappush(Bins,cap + w[i])
    else:
        heapq.heappush(Bins,cap)
        cap = heapq.heappush(Bins,c+w[i])
        nb_bins += 1
    Bins = [ 2*c - x for x in Bins]
    return Bins,nb_bins,sum(Bins)/(c*len(Bins)),time() - start_time
#Last-Fit
def LF(n,c,w):
    start_time = time()
    if n == 0:
        return 0

    Bins = [c]
    result = [[]]
    for i in range(len(w)):
        nFit = False
        for j in range(len(Bins)-1,-1,-1):
            if w[i] <= Bins[j]:
                Bins[j] = Bins[j] - w[i]
                result[j].append(w[i])

                break
            if j == len(Bins)-1:
                nFit = True

        if nFit is True:
            Bins.append(c)
            result.append([])
            result[-1].append(w[i])
            Bins[len(Bins)-1] -= w[i]
    return Bins,result,len(Bins),sum(Bins)/(c*len(Bins)),time()-start_time

def show1(c,nb,op,t,result,detail=False):
    print('temps d execution : ',t, 'ms')
    print('nombre de bins utilisés : ',nb)
    print('solution optimale : ',op)
    print('gap : ',round((nb-op)*100/op,2), '%')
    if ( detail == True ):
        for e in range(len(result)):
            print('bin ',e+1,' contient les objets : ',result[e],' | espace libre : ',c-sum(result[e]))
            
def show2(c,nb,op,t,detail=False):
    print('temps d execution : ',t, 'ms')
    print('nombre de bins utilisés : ',nb)
    print('solution optimale : ',op)
    print('gap : ',round((nb-op)*100/op,2), '%')
    if ( detail == True ):
        print(' Même result que Max-Rest ')
    
    
    
    
    