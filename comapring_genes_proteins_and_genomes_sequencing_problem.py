#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import math
import numpy as np
from finding_hidden_messages_in_dna_motif_problem import FileToListSeperated, FileToList

Blosum62 = {'A': {'A': 4, 'C': 0, 'E': -1, 'D': -2, 'G': 0, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 0, 'W': -3, 'V': 0, 'Y': -2}, 'C': {'A': 0, 'C': 9, 'E': -4, 'D': -3, 'G': -3, 'F': -2, 'I': -1, 'H': -3, 'K': -3, 'M': -1, 'L': -1, 'N': -3, 'Q': -3, 'P': -3, 'S': -1, 'R': -3, 'T': -1, 'W': -2, 'V': -1, 'Y': -2}, 'E': {'A': -1, 'C': -4, 'E': 5, 'D': 2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': -2, 'L': -3, 'N': 0, 'Q': 2, 'P': -1, 'S': 0, 'R': 0, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'D': {'A': -2, 'C': -3, 'E': 2, 'D': 6, 'G': -1, 'F': -3, 'I': -3, 'H': -1, 'K': -1, 'M': -3, 'L': -4, 'N': 1, 'Q': 0, 'P': -1, 'S': 0, 'R': -2, 'T': -1, 'W': -4, 'V': -3, 'Y': -3}, 'G': {'A': 0, 'C': -3, 'E': -2, 'D': -1, 'G': 6, 'F': -3, 'I': -4, 'H': -2, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 'Q': -2, 'P': -2, 'S': 0, 'R': -2, 'T': -2, 'W': -2, 'V': -3, 'Y': -3}, 'F': {'A': -2, 'C': -2, 'E': -3, 'D': -3, 'G': -3, 'F': 6, 'I': 0, 'H': -1, 'K': -3, 'M': 0, 'L': 0, 'N': -3, 'Q': -3, 'P': -4, 'S': -2, 'R': -3, 'T': -2, 'W': 1, 'V': -1, 'Y': 3}, 'I': {'A': -1, 'C': -1, 'E': -3, 'D': -3, 'G': -4, 'F': 0, 'I': 4, 'H': -3, 'K': -3, 'M': 1, 'L': 2, 'N': -3, 'Q': -3, 'P': -3, 'S': -2, 'R': -3, 'T': -1, 'W': -3, 'V': 3, 'Y': -1}, 'H': {'A': -2, 'C': -3, 'E': 0, 'D': -1, 'G': -2, 'F': -1, 'I': -3, 'H': 8, 'K': -1, 'M': -2, 'L': -3, 'N': 1, 'Q': 0, 'P': -2, 'S': -1, 'R': 0, 'T': -2, 'W': -2, 'V': -3, 'Y': 2}, 'K': {'A': -1, 'C': -3, 'E': 1, 'D': -1, 'G': -2, 'F': -3, 'I': -3, 'H': -1, 'K': 5, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -1, 'S': 0, 'R': 2, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'M': {'A': -1, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': 0, 'I': 1, 'H': -2, 'K': -1, 'M': 5, 'L': 2, 'N': -2, 'Q': 0, 'P': -2, 'S': -1, 'R': -1, 'T': -1, 'W': -1, 'V': 1, 'Y': -1}, 'L': {'A': -1, 'C': -1, 'E': -3, 'D': -4, 'G': -4, 'F': 0, 'I': 2, 'H': -3, 'K': -2, 'M': 2, 'L': 4, 'N': -3, 'Q': -2, 'P': -3, 'S': -2, 'R': -2, 'T': -1, 'W': -2, 'V': 1, 'Y': -1}, 'N': {'A': -2, 'C': -3, 'E': 0, 'D': 1, 'G': 0, 'F': -3, 'I': -3, 'H': 1, 'K': 0, 'M': -2, 'L': -3, 'N': 6, 'Q': 0, 'P': -2, 'S': 1, 'R': 0, 'T': 0, 'W': -4, 'V': -3, 'Y': -2}, 'Q': {'A': -1, 'C': -3, 'E': 2, 'D': 0, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': 0, 'L': -2, 'N': 0, 'Q': 5, 'P': -1, 'S': 0, 'R': 1, 'T': -1, 'W': -2, 'V': -2, 'Y': -1}, 'P': {'A': -1, 'C': -3, 'E': -1, 'D': -1, 'G': -2, 'F': -4, 'I': -3, 'H': -2, 'K': -1, 'M': -2, 'L': -3, 'N': -2, 'Q': -1, 'P': 7, 'S': -1, 'R': -2, 'T': -1, 'W': -4, 'V': -2, 'Y': -3}, 'S': {'A': 1, 'C': -1, 'E': 0, 'D': 0, 'G': 0, 'F': -2, 'I': -2, 'H': -1, 'K': 0, 'M': -1, 'L': -2, 'N': 1, 'Q': 0, 'P': -1, 'S': 4, 'R': -1, 'T': 1, 'W': -3, 'V': -2, 'Y': -2}, 'R': {'A': -1, 'C': -3, 'E': 0, 'D': -2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 2, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -2, 'S': -1, 'R': 5, 'T': -1, 'W': -3, 'V': -3, 'Y': -2}, 'T': {'A': 0, 'C': -1, 'E': -1, 'D': -1, 'G': -2, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': 0, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 5, 'W': -2, 'V': 0, 'Y': -2}, 'W': {'A': -3, 'C': -2, 'E': -3, 'D': -4, 'G': -2, 'F': 1, 'I': -3, 'H': -2, 'K': -3, 'M': -1, 'L': -2, 'N': -4, 'Q': -2, 'P': -4, 'S': -3, 'R': -3, 'T': -2, 'W': 11, 'V': -3, 'Y': 2}, 'V': {'A': 0, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': -1, 'I': 3, 'H': -3, 'K': -2, 'M': 1, 'L': 1, 'N': -3, 'Q': -2, 'P': -2, 'S': -2, 'R': -3, 'T': 0, 'W': -3, 'V': 4, 'Y': -1}, 'Y': {'A': -2, 'C': -2, 'E': -2, 'D': -3, 'G': -3, 'F': 3, 'I': -1, 'H': 2, 'K': -2, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -3, 'S': -2, 'R': -2, 'T': -2, 'W': 2, 'V': -1, 'Y': 7}}
PAM250 = {'A': {'A': 2, 'C': -2, 'E': 0, 'D': 0, 'G': 1, 'F': -3, 'I': -1, 'H': -1, 'K': -1, 'M': -1, 'L': -2, 'N': 0, 'Q': 0, 'P': 1, 'S': 1, 'R': -2, 'T': 1, 'W': -6, 'V': 0, 'Y': -3}, 'C': {'A': -2, 'C': 12, 'E': -5, 'D': -5, 'G': -3, 'F': -4, 'I': -2, 'H': -3, 'K': -5, 'M': -5, 'L': -6, 'N': -4, 'Q': -5, 'P': -3, 'S': 0, 'R': -4, 'T': -2, 'W': -8, 'V': -2, 'Y': 0}, 'E': {'A': 0, 'C': -5, 'E': 4, 'D': 3, 'G': 0, 'F': -5, 'I': -2, 'H': 1, 'K': 0, 'M': -2, 'L': -3, 'N': 1, 'Q': 2, 'P': -1, 'S': 0, 'R': -1, 'T': 0, 'W': -7, 'V': -2, 'Y': -4}, 'D': {'A': 0, 'C': -5, 'E': 3, 'D': 4, 'G': 1, 'F': -6, 'I': -2, 'H': 1, 'K': 0, 'M': -3, 'L': -4, 'N': 2, 'Q': 2, 'P': -1, 'S': 0, 'R': -1, 'T': 0, 'W': -7, 'V': -2, 'Y': -4}, 'G': {'A': 1, 'C': -3, 'E': 0, 'D': 1, 'G': 5, 'F': -5, 'I': -3, 'H': -2, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 'Q': -1, 'P': 0, 'S': 1, 'R': -3, 'T': 0, 'W': -7, 'V': -1, 'Y': -5}, 'F': {'A': -3, 'C': -4, 'E': -5, 'D': -6, 'G': -5, 'F': 9, 'I': 1, 'H': -2, 'K': -5, 'M': 0, 'L': 2, 'N': -3, 'Q': -5, 'P': -5, 'S': -3, 'R': -4, 'T': -3, 'W': 0, 'V': -1, 'Y': 7}, 'I': {'A': -1, 'C': -2, 'E': -2, 'D': -2, 'G': -3, 'F': 1, 'I': 5, 'H': -2, 'K': -2, 'M': 2, 'L': 2, 'N': -2, 'Q': -2, 'P': -2, 'S': -1, 'R': -2, 'T': 0, 'W': -5, 'V': 4, 'Y': -1}, 'H': {'A': -1, 'C': -3, 'E': 1, 'D': 1, 'G': -2, 'F': -2, 'I': -2, 'H': 6, 'K': 0, 'M': -2, 'L': -2, 'N': 2, 'Q': 3, 'P': 0, 'S': -1, 'R': 2, 'T': -1, 'W': -3, 'V': -2, 'Y': 0}, 'K': {'A': -1, 'C': -5, 'E': 0, 'D': 0, 'G': -2, 'F': -5, 'I': -2, 'H': 0, 'K': 5, 'M': 0, 'L': -3, 'N': 1, 'Q': 1, 'P': -1, 'S': 0, 'R': 3, 'T': 0, 'W': -3, 'V': -2, 'Y': -4}, 'M': {'A': -1, 'C': -5, 'E': -2, 'D': -3, 'G': -3, 'F': 0, 'I': 2, 'H': -2, 'K': 0, 'M': 6, 'L': 4, 'N': -2, 'Q': -1, 'P': -2, 'S': -2, 'R': 0, 'T': -1, 'W': -4, 'V': 2, 'Y': -2}, 'L': {'A': -2, 'C': -6, 'E': -3, 'D': -4, 'G': -4, 'F': 2, 'I': 2, 'H': -2, 'K': -3, 'M': 4, 'L': 6, 'N': -3, 'Q': -2, 'P': -3, 'S': -3, 'R': -3, 'T': -2, 'W': -2, 'V': 2, 'Y': -1}, 'N': {'A': 0, 'C': -4, 'E': 1, 'D': 2, 'G': 0, 'F': -3, 'I': -2, 'H': 2, 'K': 1, 'M': -2, 'L': -3, 'N': 2, 'Q': 1, 'P': 0, 'S': 1, 'R': 0, 'T': 0, 'W': -4, 'V': -2, 'Y': -2}, 'Q': {'A': 0, 'C': -5, 'E': 2, 'D': 2, 'G': -1, 'F': -5, 'I': -2, 'H': 3, 'K': 1, 'M': -1, 'L': -2, 'N': 1, 'Q': 4, 'P': 0, 'S': -1, 'R': 1, 'T': -1, 'W': -5, 'V': -2, 'Y': -4}, 'P': {'A': 1, 'C': -3, 'E': -1, 'D': -1, 'G': 0, 'F': -5, 'I': -2, 'H': 0, 'K': -1, 'M': -2, 'L': -3, 'N': 0, 'Q': 0, 'P': 6, 'S': 1, 'R': 0, 'T': 0, 'W': -6, 'V': -1, 'Y': -5}, 'S': {'A': 1, 'C': 0, 'E': 0, 'D': 0, 'G': 1, 'F': -3, 'I': -1, 'H': -1, 'K': 0, 'M': -2, 'L': -3, 'N': 1, 'Q': -1, 'P': 1, 'S': 2, 'R': 0, 'T': 1, 'W': -2, 'V': -1, 'Y': -3}, 'R': {'A': -2, 'C': -4, 'E': -1, 'D': -1, 'G': -3, 'F': -4, 'I': -2, 'H': 2, 'K': 3, 'M': 0, 'L': -3, 'N': 0, 'Q': 1, 'P': 0, 'S': 0, 'R': 6, 'T': -1, 'W': 2, 'V': -2, 'Y': -4}, 'T': {'A': 1, 'C': -2, 'E': 0, 'D': 0, 'G': 0, 'F': -3, 'I': 0, 'H': -1, 'K': 0, 'M': -1, 'L': -2, 'N': 0, 'Q': -1, 'P': 0, 'S': 1, 'R': -1, 'T': 3, 'W': -5, 'V': 0, 'Y': -3}, 'W': {'A': -6, 'C': -8, 'E': -7, 'D': -7, 'G': -7, 'F': 0, 'I': -5, 'H': -3, 'K': -3, 'M': -4, 'L': -2, 'N': -4, 'Q': -5, 'P': -6, 'S': -2, 'R': 2, 'T': -5, 'W': 17, 'V': -6, 'Y': 0}, 'V': {'A': 0, 'C': -2, 'E': -2, 'D': -2, 'G': -1, 'F': -1, 'I': 4, 'H': -2, 'K': -2, 'M': 2, 'L': 2, 'N': -2, 'Q': -2, 'P': -1, 'S': -1, 'R': -2, 'T': 0, 'W': -6, 'V': 4, 'Y': -2}, 'Y': {'A': -3, 'C': 0, 'E': -4, 'D': -4, 'G': -5, 'F': 7, 'I': -1, 'H': 0, 'K': -4, 'M': -2, 'L': -1, 'N': -2, 'Q': -4, 'P': -5, 'S': -3, 'R': -4, 'T': -3, 'W': 0, 'V': -2, 'Y': 10}}

def RecursiveChange(money, Coins):
    print(money)
    if money == 0:
        return 0
    MinNumCoins = math.inf
    for coin in Coins:
        if money >= coin:
            NumCoins = RecursiveChange((money-coin), Coins)
            if (NumCoins + 1) < MinNumCoins:
                MinNumCoins = NumCoins+1
    return MinNumCoins            

def DPChange(money, Coins):
    MinNumCoins = [0]*(money+1)
    for m in range(1, money+1):
        MinNumCoins[m] = math.inf 
        for coin in Coins:
            if m >= coin:
                if MinNumCoins[m-coin]+1 < MinNumCoins[m]:
                    MinNumCoins[m] = MinNumCoins[m-coin]+1
    return MinNumCoins[-1]

def ManhattanTourist(n, m, Down, Right):
    s = np.zeros((n+1,m+1))
    for i in range(1,n+1):
        s[i][0] = s[i-1][0] + Down[i-1][0]
    for j in range(1,m+1):
        s[0][j] = s[0][j-1] + Right[0][j-1]
    for i in range(1,n+1):
        for j in range(1,m+1):
            go_down = s[i-1][j] + Down[i-1][j]
            go_right = s[i][j-1] + Right[i][j-1]
            s[i][j] = max(go_down, go_right)           
    return s[n][m]

def LCSBackTrack(stringA, stringB):
    print(len(stringA))
    print(len(stringB))
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            match = 0
            if stringA[i-1] == stringB[j-1]:
                match = 1
            s[i][j] = max(s[i-1][j], s[i][j-1], s[i-1][j-1]+match)
            if s[i][j] == s[i-1][j]:
                backtrack[i][j] = 'top(deletion)'
            elif s[i][j] == s[i][j-1]:
                backtrack[i][j] = 'left(insertion)'
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = 'match/mismatch'
    return backtrack                  

def OutputLCS(backtrack, v, i, j):
    LCS = []
    while (i>0 and j>0):
        if backtrack[i][j] == 'match/mismatch':
            LCS.append(v[i-1])
            i = i-1
            j = j-1
        elif backtrack[i][j] == 'top(deletion)':
            i = i-1
        else:
            j = j-1
    LCS.reverse()   
    print(len(LCS))     
    return ''.join(LCS)

def LongestPathOfDAG(graph, start_node, end_node):
    s = {}
    backtrack = {}
    curr_node = list(graph.keys())[0]
    while curr_node != end_node:
        if curr_node in s.keys():
            s[graph[curr_node][0]] = s[curr_node] + graph[curr_node][1]
            backtrack[graph[curr_node][0]] = curr_node
            curr_node = graph[curr_node][0]
        else:
            s[graph[curr_node][0]] = graph[curr_node][1]
            backtrack[graph[curr_node][0]] = curr_node
            curr_node = graph[curr_node][0]
    current_node = end_node
    path = []    
    print(backtrack)
    while current_node != start_node:
        path.append(current_node)    
        current_node = backtrack[current_node]
    path.append(start_node)   
    path.reverse() 
    print(s[end_node])
    return '->'.join(list(map(str,path)))

def GlobalAlignmentProblem(stringA, stringB):
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    for i in range(0, len(stringA)+1):
        s[i][0] = i*(-5)
    for j in range(0, len(stringB)+1):
        s[0][j] = j*(-5)
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            match = Blosum62[stringA[i-1]][stringB[j-1]]
            s[i][j] = max(s[i-1][j]-5, s[i][j-1]-5, s[i-1][j-1]+match)
            if s[i][j] == (s[i-1][j]-5):
                backtrack[i][j] = 'top(deletion)'
            elif s[i][j] == (s[i][j-1]-5):
                backtrack[i][j] = 'left(insertion)'
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = 'match/mismatch'
    for line in backtrack:
        print(line)            
    alignmentA = []
    alignmentB = []
    i = len(stringA)
    j = len(stringB)
    while (i>0 and j>0):
        if backtrack[i][j] == 'match/mismatch':
            alignmentA.append(stringA[i-1])
            alignmentB.append(stringB[j-1])
            i = i-1
            j = j-1
        elif backtrack[i][j] == 'top(deletion)':
            alignmentA.append(stringA[i-1])
            alignmentB.append('-')
            i = i-1
        else:
            alignmentA.append('-')
            alignmentB.append(stringB[j-1])
            j = j-1
    if i>0:
        alignmentA.append(stringA[i-1])
        alignmentB.append('-')
        i=i-1
    if j>0:
        alignmentA.append('-')
        alignmentB.append(stringB[j-1])
        j=j-1           
    alignmentA.reverse()
    alignmentB.reverse()        
    print(''.join(alignmentA))
    print(''.join(alignmentB))                   
    return s[len(stringA)][len(stringB)]

def LocalAlignmentProblem(stringA, stringB):
    highest_score = 0
    highest_location = [0,0]
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    for i in range(1, len(stringA)+1):
        backtrack[i][0] = 'SS'
    for j in range(1, len(stringB)+1):
        backtrack[0][j] = 'SS'    
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            match = PAM250[stringA[i-1]][stringB[j-1]]
            s[i][j] = max(0,s[i-1][j]-5, s[i][j-1]-5, s[i-1][j-1]+match)
            if s[i][j] > highest_score:
                highest_score = s[i][j]
                highest_location[0] = i
                highest_location[1] = j     
            if s[i][j] == 0:
                backtrack[i][j] = 'SS'    
            elif s[i][j] == (s[i-1][j]-5):
                backtrack[i][j] = 'TD'
            elif s[i][j] == (s[i][j-1]-5):
                backtrack[i][j] = 'LI'    
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = 'MM'     
    #print(s)              
    #for line in backtrack:
    #    print(line)            
    alignmentA = []
    alignmentB = []
    i = highest_location[0]
    j = highest_location[1]
    while (backtrack[i][j] != 'SS'):
        if backtrack[i][j] == 'MM':
            alignmentA.append(stringA[i-1])
            alignmentB.append(stringB[j-1])
            i = i-1
            j = j-1
        elif backtrack[i][j] == 'TD':
            alignmentA.append(stringA[i-1])
            alignmentB.append('-')
            i = i-1
        else:
            alignmentA.append('-')
            alignmentB.append(stringB[j-1])
            j = j-1          
    alignmentA.reverse()
    alignmentB.reverse()        
    print(''.join(alignmentA))
    print(''.join(alignmentB))                   
    return str(s[highest_location[0]][highest_location[1]])

def EditDistance(stringA, stringB):
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    for i in range(0, len(stringA)+1):
        s[i][0] = i
    for j in range(0, len(stringB)+1):
        s[0][j] = j
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            if stringA[i-1] == stringB[j-1]:
                match = 0
            else:
                match = 1    
            s[i][j] = min(s[i-1][j]+1, s[i][j-1]+1, s[i-1][j-1]+match)                   
    return s[len(stringA)][len(stringB)]
    
def FittingAlignment(stringA, stringB):
    highest_score = 0
    highest_location = [0,0]
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    for i in range(0, len(stringA)+1):
        s[i][0] = 0
    for j in range(0, len(stringB)+1):
        s[0][j] = -j
    for i in range(1, len(stringA)+1):
        backtrack[i][0] = 'TD'
    for j in range(1, len(stringB)+1):
        backtrack[0][j] = 'LI'    
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            if stringA[i-1] == stringB[j-1]:
                match = 1
            else:
                match = -1    
            s[i][j] = max(s[i-1][j]-1, s[i][j-1]-1, s[i-1][j-1]+match)
            if j == len(stringB):
                if s[i][j] > highest_score:
                    highest_score = s[i][j]
                    highest_location[0] = i
                    highest_location[1] = j     
            if s[i][j] == (s[i][j-1]-1):
                backtrack[i][j] = 'LI'        
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = 'MM'          
            elif s[i][j] == (s[i-1][j]-1):
                backtrack[i][j] = 'TD'         
    print(s)              
    #for line in backtrack:
    #    print(line)            
    alignmentA = []
    alignmentB = []
    i = highest_location[0]
    j = highest_location[1]
    while (j>0):
        if backtrack[i][j] == 'MM':
            alignmentA.append(stringA[i-1])
            alignmentB.append(stringB[j-1])
            i = i-1
            j = j-1
        elif backtrack[i][j] == 'TD':
            alignmentA.append(stringA[i-1])
            alignmentB.append('-')
            i = i-1
        else:
            alignmentA.append('-')
            alignmentB.append(stringB[j-1])
            j = j-1          
    alignmentA.reverse()
    alignmentB.reverse()        
    print(''.join(alignmentA))
    print(''.join(alignmentB))                   
    return highest_score

def OverlapAlignment(stringA, stringB):
    highest_score = 0
    highest_location = [0,0]
    s = np.zeros((len(stringA)+1, len(stringB)+1))
    backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    first_char_B = stringB[0]
    if first_char_B in stringA:
        start = stringA.index(first_char_B)
    else:
        start = 0    
    for i in range(0, len(stringA)+1):
        s[i][0] = 0
    for j in range(0, len(stringB)+1):
        s[0][j] = -j*2
    for i in range(1, len(stringA)+1):
        backtrack[i][0] = 'TD'
    for j in range(1, len(stringB)+1):
        backtrack[0][j] = 'LI'    
    for i in range(start+1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            if stringA[i-1] == stringB[j-1]:
                match = 1
            else:
                match = -2    
            s[i][j] = max(s[i-1][j]-2, s[i][j-1]-2, s[i-1][j-1]+match)
            if i == len(stringA):
                if s[i][j] > highest_score:
                    highest_score = s[i][j]
                    highest_location[0] = i
                    highest_location[1] = j   
            if s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = 'MM'          
            elif s[i][j] == (s[i][j-1]-2):
                backtrack[i][j] = 'LI'                  
            elif s[i][j] == (s[i-1][j]-2):
                backtrack[i][j] = 'TD'         
    print(s)              
    #for line in backtrack:
    #    print(line)            
    alignmentA = []
    alignmentB = []
    i = highest_location[0]
    j = highest_location[1]
    while (j>0):
        if backtrack[i][j] == 'MM':
            alignmentA.append(stringA[i-1])
            alignmentB.append(stringB[j-1])
            i = i-1
            j = j-1
        elif backtrack[i][j] == 'TD':
            alignmentA.append(stringA[i-1])
            alignmentB.append('-')
            i = i-1
        else:
            alignmentA.append('-')
            alignmentB.append(stringB[j-1])
            j = j-1          
    alignmentA.reverse()
    alignmentB.reverse()        
    print(''.join(alignmentA))
    print(''.join(alignmentB))                   
    return highest_score

def AlignmentWithAffineGaps(stringA, stringB):
    middle_s = np.zeros((len(stringA)+1, len(stringB)+1))
    lower_s = np.zeros((len(stringA)+1, len(stringB)+1))
    upper_s = np.zeros((len(stringA)+1, len(stringB)+1))
    middle_backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    lower_backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    upper_backtrack = [[None for _ in range(0, len(stringB)+1)] for _ in range(0, len(stringA)+1)]
    for i in range(1, len(stringA)+1):
        middle_s[i][0] = -11-i+1
        lower_s[i][0] = -11-i+1
        upper_s[i][0] = -math.inf
    for j in range(1, len(stringB)+1):
        middle_s[0][j] = -11-j+1
        lower_s[0][j] = -math.inf
        upper_s[0][j] = -11-j+1
    for d in range(1, min(len(stringA),len(stringB))+1):
        match = Blosum62[stringA[d-1]][stringB[d-1]]
        lower_s[d][d] = max(lower_s[d-1][d]-1, middle_s[d-1][d]-11)
        upper_s[d][d] = max(upper_s[d][d-1]-1, middle_s[d][d-1]-11)
        middle_s[d][d] = max(lower_s[d][d], upper_s[d][d], middle_s[d-1][d-1]+match)
        """ print(lower_s[d][d])
        print(upper_s[d][d])
        print(middle_s[d-1][d-1])
        print(match)
        print(middle_s[d][d]) """
        if lower_s[d][d] == middle_s[d-1][d]-11:
            lower_backtrack[d][d] = 'ML'
        else:
            lower_backtrack[d][d] = 'DL'
        if upper_s[d][d] == middle_s[d][d-1]-11:
            upper_backtrack[d][d] = 'MU'
        else:
            upper_backtrack[d][d] = 'RU'
        if middle_s[d][d] == middle_s[d-1][d-1]+match:
            middle_backtrack[d][d] = 'DM'
        elif middle_s[d][d] == lower_s[d][d]:
            middle_backtrack[d][d] = 'LM'
        else:
            middle_backtrack[d][d] = 'UM'     
        for i in range(d+1, min(len(stringA),len(stringB))+1):
            match = Blosum62[stringA[i-1]][stringB[d-1]]
            lower_s[i][d] = max(lower_s[i-1][d]-1, middle_s[i-1][d]-11)
            upper_s[i][d] = max(upper_s[i][d-1]-1, middle_s[i][d-1]-11)
            middle_s[i][d] = max(lower_s[i][d], upper_s[i][d], middle_s[i-1][d-1]+match)
            if lower_s[i][d] == middle_s[i-1][d]-11:
                lower_backtrack[i][d] = 'ML'
            else:
                lower_backtrack[i][d] = 'DL'
            if upper_s[i][d] == middle_s[i][d-1]-11:
                upper_backtrack[i][d] = 'MU'
            else:
                upper_backtrack[i][d] = 'RU'    
            if middle_s[i][d] == middle_s[i-1][d-1]+match:
                middle_backtrack[i][d] = 'DM'
            elif middle_s[i][d] == lower_s[i][d]:
                middle_backtrack[i][d] = 'LM'
            else:
                middle_backtrack[i][d] = 'UM'    
        for j in range(d+1, min(len(stringA),len(stringB))+1):
            lower_s[d][j] = max(lower_s[d-1][j]-1, middle_s[d-1][j]-11)
            if lower_s[d][j] == middle_s[d-1][j]-11:
                lower_backtrack[d][j] = 'ML'
            else:
                lower_backtrack[d][j] = 'DL'
        for j in range(d+1, min(len(stringA),len(stringB))+1):
            match = Blosum62[stringA[d-1]][stringB[j-1]]
            upper_s[d][j] = max(upper_s[d][j-1]-1, middle_s[d][j-1]-11)
            middle_s[d][j] = max(lower_s[d][j], upper_s[d][j], middle_s[d-1][j-1]+match)
            if upper_s[d][j] == middle_s[d][j-1]-11:
                upper_backtrack[d][j] = 'MU'
            else:
                upper_backtrack[d][j] = 'RU'
            if middle_s[d][j] == middle_s[d-1][j-1]+match:
                middle_backtrack[d][j] = 'DM'
            elif middle_s[d][j] == lower_s[d][j]:
                middle_backtrack[d][j] = 'LM'
            else:
                middle_backtrack[d][j] = 'UM'    
        for i in range(d+1, min(len(stringA),len(stringB))+1):
            upper_s[i][d] = max(upper_s[i][d-1]-1, middle_s[i][d-1]-11)
            if upper_s[i][d] == middle_s[i][d-1]-11:
                upper_backtrack[i][d] = 'MU'
            else:
                upper_backtrack[i][d] = 'RU'
    """ print(middle_s)    
    for line in middle_backtrack:
        print(line)        
    print(lower_s)
    for line in lower_backtrack:
        print(line)
    print(upper_s)
    for line in upper_backtrack:
        print(line)         """   
    print('done with first part of matrices')     
    if len(stringA) > len(stringB):
        for j in range(1, len(stringB)+1):
            for i in range(len(stringB)+1, len(stringA)+1):
                match = Blosum62[stringA[i-1]][stringB[j-1]]
                lower_s[i][j] = max(lower_s[i-1][j]-1, middle_s[i-1][j]-11)
                upper_s[i][j] = max(upper_s[i][j-1]-1, middle_s[i][j-1]-11)
                middle_s[i][j] = max(lower_s[i][j], upper_s[i][j], middle_s[i-1][j-1]+match)
                if lower_s[i][j] == middle_s[i-1][j]-11:
                    lower_backtrack[i][j] = 'ML'
                else:
                    lower_backtrack[i][j] = 'DL'
                if upper_s[i][j] == middle_s[i][j-1]-11:
                    upper_backtrack[i][j] = 'MU'
                else:
                    upper_backtrack[i][j] = 'RU'    
                if middle_s[i][j] == middle_s[i-1][j-1]+match:
                    middle_backtrack[i][j] = 'DM'
                elif middle_s[i][j] == lower_s[i][j]:
                    middle_backtrack[i][j] = 'LM'
                else:
                    middle_backtrack[i][j] = 'UM'  
    if len(stringB) > len(stringA):
        for j in range(len(stringA)+1, len(stringB)+1):
            for i in range(1, len(stringA)+1):
                match = Blosum62[stringA[i-1]][stringB[j-1]]
                lower_s[i][j] = max(lower_s[i-1][j]-1, middle_s[i-1][j]-11)
                upper_s[i][j] = max(upper_s[i][j-1]-1, middle_s[i][j-1]-11)
                middle_s[i][j] = max(lower_s[i][j], upper_s[i][j], middle_s[i-1][j-1]+match)
                if lower_s[i][j] == middle_s[i-1][j]-11:
                    lower_backtrack[i][j] = 'ML'
                else:
                    lower_backtrack[i][j] = 'DL'
                if upper_s[i][j] == middle_s[i][j-1]-11:
                    upper_backtrack[i][j] = 'MU'
                else:
                    upper_backtrack[i][j] = 'RU'    
                if middle_s[i][j] == middle_s[i-1][j-1]+match:
                    middle_backtrack[i][j] = 'DM'
                elif middle_s[i][j] == lower_s[i][j]:
                    middle_backtrack[i][j] = 'LM'
                else:
                    middle_backtrack[i][j] = 'UM' 
    print('----------------------')                                 
    print(middle_s)    
    for line in middle_backtrack:
        print(line)        
    """print(lower_s)
    for line in lower_backtrack:
        print(line)
    print(upper_s)
    for line in upper_backtrack:
        print(line)  """
    alignmentA = []
    alignmentB = []   
    i = len(stringA)              
    j = len(stringB)
    place = 'middle'
    print('backtracking now')
    while ((i > 0) and (j > 0)):
        """ print(place)
        print(f'i: {i}')
        print(f'j: {j}') """
        if place == 'middle':
            #print(middle_backtrack[i][j])
            if middle_backtrack[i][j] == 'DM':
                alignmentA.append(stringA[i-1])
                alignmentB.append(stringB[j-1])
                i = i-1
                j = j-1
            elif middle_backtrack[i][j] == 'LM':
                #alignmentA.append(stringA[i-1])
                #alignmentB.append('-')
                place = 'lower'
            elif middle_backtrack[i][j] == 'UM':
                #alignmentA.append('-')
                #alignmentB.append(stringB[j-1])
                place = 'upper'
        elif place == 'lower':
            if lower_backtrack[i][j] == 'DL':
                alignmentA.append(stringA[i-1])
                alignmentB.append('-')
                i = i-1 
            else:
                alignmentA.append(stringA[i-1])
                alignmentB.append('-')
                i = i-1
                place = 'middle'
        else:
            if upper_backtrack[i][j] == 'RU':
                alignmentA.append('-')
                alignmentB.append(stringB[j-1])
                j = j-1
            else:
                alignmentA.append('-')
                alignmentB.append(stringB[j-1])
                j = j-1
                place = 'middle'
        #print(alignmentA)
        #print(alignmentB)    
    while i>0:
        print(stringA[i-1])
        alignmentA.append(stringA[i-1])
        alignmentB.append('-')
        i=i-1
    while j>0:
        alignmentA.append('-')
        alignmentB.append(stringB[j-1])
        j=j-1           
    alignmentA.reverse()
    alignmentB.reverse()
    print(''.join(alignmentA))
    print(''.join(alignmentB))                       
    return middle_s[len(stringA)][len(stringB)]

def FindMiddleNode(stringA, stringB):
    print('¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨')
    #print(f'stringA: {stringA}')
    #print(f'stringB: {stringB}')
    if len(stringB) == 1:
        middle_column = 0
    else:    
        middle_column = math.floor(len(stringB)/2)
    current_column = []
    old_column = []
    from_source_middle = []
    next_column = []
    max_value = -math.inf
    max_index = 0
    max_match = -math.inf
    for i in range(0, len(stringA)+1):
        current_column.append(i*(-5))
    if middle_column == len(stringB):
        for j in range(1, middle_column+1):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column:       
                from_source_middle = current_column[::1]    
    elif middle_column >= 1:     
        for j in range(1, middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column:        
                from_source_middle = current_column[::1]
            if j == middle_column+1:
                next_column = current_column[::1]
    elif middle_column == 0:
        from_source_middle = current_column[::1]
        for j in range(1, middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column+1:
                next_column = current_column[::1]           
    stringA = stringA[::-1]
    stringB = stringB[::-1]
    max_to = -math.inf
    print(from_source_middle)
    print(next_column)
    for i in range(0, len(stringA)+1):
        current_column[i] = i*(-5)
    if middle_column != 0:    
        for j in range(1, len(stringB)-middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                    if j == len(stringB)-middle_column+1:
                        print(current_column[i])
                        if (current_column[i] + from_source_middle[-1]) > max_value:
                            horz = -math.inf
                            if from_source_middle[-1]-5 == next_column[-1]:
                                horz = next_column[-1]
                            if horz > -math.inf:
                                max_value = current_column[i] + from_source_middle[-i-1]
                                max_index = len(from_source_middle)-i-1
                                max_to = horz
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
                    if j == len(stringB)-middle_column+1:
                        print(f'max to : {max_to}')
                        print(f'from source middle (current) : {from_source_middle[-i-1]}')
                        if (current_column[i] + from_source_middle[-i-1]) > max_value:
                            """
                            if (current_column[i] + from_source_middle[-i-1]) == max_value:
                                if next_column[-i] - from_source_middle[-i-1] >= max_match:
                                    max_value = current_column[i] + from_source_middle[-i-1]
                                    max_index = len(from_source_middle)-i-1
                                    max_match = next_column[-i] - from_source_middle[-i-1]
                            else:  """
                            vert = -math.inf
                            diag = -math.inf
                            horz = -math.inf
                            if from_source_middle[-i-1]-5 == from_source_middle[-i]:
                                vert = from_source_middle[-i]
                            if from_source_middle[-i-1] + Blosum62[stringA[i-1]][stringB[-j]] == next_column[-i]:
                                diag = next_column[-i]
                            if from_source_middle[-i-1]-5 == next_column[-i-1]:
                                vert = next_column[-i-1]
                            max_dir = max(diag, vert, horz)
                            if max_dir > max_to:
                                max_value = current_column[i] + from_source_middle[-i-1]
                                max_index = len(from_source_middle)-i-1 
                                max_to = max_dir        
    else:
        for i in range(0, len(stringA)+1):
            if i == 0:
                current_column[i] = j*(-5)
                if (current_column[i] + from_source_middle[-1]) > max_value:
                    horz = -math.inf
                    if from_source_middle[-1]-5 == next_column[-1]:
                        horz = next_column[-1]
                    if horz > -math.inf:
                        max_value = current_column[i] + from_source_middle[-i-1]
                        max_index = len(from_source_middle)-i-1
                        max_to = horz
            else:    
                match = Blosum62[stringA[i-1]][stringB[0]]
                current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
                if (current_column[i] + from_source_middle[-i-1]) > max_value:
                    vert = -math.inf
                    diag = -math.inf
                    horz = -math.inf
                    if from_source_middle[-i-1]-5 == from_source_middle[-i]:
                        vert = from_source_middle[-i]
                    if from_source_middle[-i-1] + Blosum62[stringA[i-1]][stringB[0]] == next_column[-i]:
                        diag = next_column[-i]
                    if from_source_middle[-i-1]-5 == next_column[-i-1]:
                        vert = next_column[-i-1]
                    max_dir = max(diag, vert, horz)
                    if max_dir > max_to:
                        max_value = current_column[i] + from_source_middle[-i-1]
                        max_index = len(from_source_middle)-i-1 
                        max_to = max_dir 
        for i in range(0, len(stringA)+1):
            if i == 0:
                if (current_column[i] + from_source_middle[-1]) > max_value:
                    max_value = current_column[i] + from_source_middle[-1]
                    max_index = len(from_source_middle)-1    
            else:    
                if (current_column[i] + from_source_middle[-i-1]) > max_value:
                    """ if (current_column[i] + from_source_middle[-i-1]) == max_value:
                        if next_column[-i] - from_source_middle[-i-1] >= max_match:
                            max_value = current_column[i] + from_source_middle[-i-1]
                            max_index = len(from_source_middle)-i-1
                            max_match = next_column[-i] - from_source_middle[-i-1]
                    else:  """
                    max_value = current_column[i] + from_source_middle[-i-1]
                    max_index = len(from_source_middle)-i-1
    to_node =(0,0)
    direction = 0
    stringA = stringA[::-1]
    stringB = stringB[::-1]
    #print(f'Max Value: {max_value}')
    #print(f'middle column: {middle_column}')
    #print(f'middle row (before): {max_index}')
    if (max_index != len(from_source_middle)-1):
        if middle_column != len(stringB):
            vert = -math.inf
            diag = -math.inf
            horz = -math.inf
            if from_source_middle[max_index]-5 == from_source_middle[max_index+1]:
                vert = from_source_middle[max_index+1]
            if from_source_middle[max_index] + Blosum62[stringA[max_index]][stringB[middle_column]] == next_column[max_index+1]:
                diag = next_column[max_index+1]
            if from_source_middle[max_index]-5 == next_column[max_index]:
                horz = next_column[max_index]
            max_dir = max(diag, vert, horz)
            if max_dir == diag:
                to_node = (max_index+1, middle_column+1)
            elif max_dir == vert:
                to_node = (max_index+1, middle_column)
            else:
                to_node = (max_index, middle_column+1)                   
            #if ((from_source_middle[max_index] + Blosum62[stringA[max_index]][stringB[middle_column]] == next_column[max_index+1]) and (Blosum62[stringA[max_index]][stringB[middle_column]] > -5)):
            #    to_node = (max_index+1, middle_column+1)    
        else:
            if from_source_middle[max_index]-5 == from_source_middle[max_index+1]:
                to_node = (max_index+1, middle_column)   
        """ if from_source_middle[max_index]-5 == next_column[max_index]:
            to_node = (max_index, middle_column+1) """                       
    elif max_index == len(from_source_middle)-1:
        to_node = (max_index, middle_column+1)
    if ((to_node[0] == max_index) and (to_node[1] == middle_column+1)):
        direction = 'H'
    elif ((to_node[0] == max_index+1) and (to_node[1] == middle_column)):
        direction = 'V'
    else:
        direction = 'D'                  
    return (max_index, middle_column), to_node, direction

def FindMiddleNodeFromEdges(stringA, stringB):
    print('¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨')  
    middle_column = math.floor(len(stringB)/2)
    current_column = []
    old_column = []
    from_source_middle = []
    from_sink_middle = []
    next_column = []
    next_column_sink = []
    diff_from_source = []
    diff_from_sink = []
    max_value = -math.inf
    max_index = 0
    max_match = -math.inf
    for i in range(0, len(stringA)+1):
        current_column.append(i*(-5))
    if middle_column == len(stringB):
        for j in range(1, middle_column+1):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column:       
                from_source_middle = current_column[::1]    
    elif middle_column >= 1:     
        for j in range(1, middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column:        
                from_source_middle = current_column[::1]
            if j == middle_column+1:
                next_column = current_column[::1]
    elif middle_column == 0:
        from_source_middle = current_column[::1]
        for j in range(1, middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == middle_column+1:
                next_column = current_column[::1]
    for i in range(0, len(stringA)+1):
        vert = -math.inf
        diag = -math.inf
        horz = -math.inf
        if i == len(stringA):
            if from_source_middle[i]-5 == next_column[i]:
                horz = -5
            max_dir = max(diag, vert, horz)
        else:    
            if from_source_middle[i]-5 == from_source_middle[i+1]:
                vert = -5
            #print(stringB[middle_column])    
            if from_source_middle[i] + Blosum62[stringA[i]][stringB[middle_column]] == next_column[i+1]:
                diag = Blosum62[stringA[i]][stringB[middle_column]]
            if from_source_middle[i]-5 == next_column[i]:
                horz = -5
            max_dir = max(diag, vert, horz)
        diff_from_source.append(max_dir)                    
    stringA = stringA[::-1]
    stringB = stringB[::-1]
    max_to = -math.inf
    print(f' diff from source: {diff_from_source}')
    #print(from_source_middle)
    #print(next_column)
    for i in range(0, len(stringA)+1):
        current_column[i] = i*(-5)
    if middle_column != 0:    
        for j in range(1, len(stringB)-middle_column+2):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == len(stringB)-middle_column:
                from_sink_middle = current_column[::1]
            if j == len(stringB)-middle_column+1:
                next_column_sink = current_column[::1]            
    else:
        from_sink_middle = current_column[::1]
        for j in range(1, len(stringB)-middle_column+1):
            old_column = current_column[::1]
            for i in range(0, len(stringA)+1):
                if i == 0:
                    current_column[i] = j*(-5)
                else:    
                    match = Blosum62[stringA[i-1]][stringB[j-1]]
                    current_column[i] = max(old_column[i-1]+match, old_column[i]-5, current_column[i-1]-5)
            if j == len(stringB)-middle_column:
                next_column_sink = current_column[::1]
    for i in range(0, len(stringA)+1):
        vert = -math.inf
        diag = -math.inf
        horz = -math.inf
        if i == len(stringA):
            if from_sink_middle[i]-5 == next_column_sink[i]:
                horz = next_column_sink[i]
            max_dir = max(diag, vert, horz)
        else:    
            if from_sink_middle[i]-5 == from_sink_middle[i+1]:
                vert = -5      
            #print(f'Next Cloumn sink: {next_column_sink}')
            if middle_column == 0:
                new_middle = 0
            else:    
                new_middle =len(stringB)-middle_column
            if from_sink_middle[i] + Blosum62[stringA[i]][stringB[new_middle]] == next_column_sink[i+1]:
                diag = Blosum62[stringA[i]][stringB[new_middle]]
            if from_sink_middle[max_index]-5 == next_column_sink[max_index]:
                horz = -5
            max_dir = max(diag, vert, horz)
        diff_from_sink.append(max_dir)
        #print(max_value)
        if diff_from_sink[i]+diff_from_source[-i-1] > max_value:
            max_value = diff_from_sink[i]+diff_from_source[-i-1]
            max_index = len(from_source_middle)-i-1 
            #print(f'max index: {max_index}')         
    to_node =(0,0)
    direction = 0
    stringA = stringA[::-1]
    stringB = stringB[::-1]
    print(f' middle sink: {from_sink_middle}')
    print(f' next sink: {next_column_sink}')
    print(f' diff from sink: {diff_from_sink}')
    #print(f'Max Value: {max_value}')
    #print(f'middle column: {middle_column}')
    #print(f'middle row (before): {max_index}')
    if (max_index != len(from_source_middle)-1):
        if middle_column != len(stringB):
            if diff_from_source[max_index] == -5:
                if diff_from_source[max_index] == from_source_middle[max_index+1]-from_source_middle[max_index]:
                    to_node = (max_index+1, middle_column)
                elif diff_from_source[max_index] == next_column[max_index]-from_source_middle[max_index]:
                    to_node = (max_index, middle_column+1) 
            elif diff_from_source[max_index] == next_column[max_index+1] - from_source_middle[max_index]:
                to_node = (max_index+1, middle_column+1)             
            #if ((from_source_middle[max_index] + Blosum62[stringA[max_index]][stringB[middle_column]] == next_column[max_index+1]) and (Blosum62[stringA[max_index]][stringB[middle_column]] > -5)):
            #    to_node = (max_index+1, middle_column+1)    
        else:
            if from_source_middle[max_index]-5 == from_source_middle[max_index+1]:
                to_node = (max_index+1, middle_column)                         
    elif max_index == len(from_source_middle)-1:
        to_node = (max_index, middle_column+1)
    if ((to_node[0] == max_index) and (to_node[1] == middle_column+1)):
        direction = 'H'
    elif ((to_node[0] == max_index+1) and (to_node[1] == middle_column)):
        direction = 'V'
    else:
        direction = 'D'                  
    return (max_index, middle_column), to_node, direction    

""" LinearSpaceAlignment(v, w, top, bottom, left, right)
    if left = right
        output path formed by bottom − top vertical edges
    if top = bottom
        output path formed by right − left horizontal edges
    middle ← ⌊ (left + right)/2⌋
    midEdge ← MiddleEdge(v, w, top, bottom, left, right)
    midNode ← vertical coordinate of the initial node of midEdge 
    LinearSpaceAlignment(v, w, top, midNode, left, middle)
    output midEdge
    if midEdge = "→" or midEdge = "↘"
        middle ← middle + 1
    if midEdge = "↓" or midEdge ="↘"
        midNode ← midNode + 1 
    LinearSpaceAlignment(v, w, midNode, bottom, middle, right) """

def LinearSpaceAlignment(stringA, stringB, top=0, bottom=0, left=0, right=0, path = []):
    if left == right:
        print('------------')
        print(f'left=right={left}, TOP, BOTTTOM:{top}, {bottom}')
        print(f'PATH: {path}')
        for _ in range(0, (bottom-top)):
            path.append('V')    
            #print('V')
    elif top == bottom:
        print('------------')
        print(f'top=bottom={top}, LEFT, RIGHT: {left}, {right}')
        print(f'PATH: {path}')
        for _ in range(0, (right-left)):
            path.append('H')
            #print('H')    
    else:
        if left + right == 1:
            middle_column = 0
        else:    
            middle_column = math.floor((left+right)/2)
        #print('---------------')
        print(f'TOP, BOTTOM, LEFT, RIGHT: {top}, {bottom}, {left}, {right}')
        #print(f'A: {stringA[top:bottom]}')
        #print(f'B: {stringB[left:right]}')
        #print(f'Path: {path}')
        middle_node = FindMiddleNodeFromEdges(stringA[top:bottom], stringB[left:right])
        mid_edge = middle_node[2]
        #if top == 0:
        #if middle_node[0][0] == 1:
        #    mid_row = top
        #else:  
        if top+middle_node[0][0] == 0:
            mid_row = top+middle_node[0][0]
        else:       
            mid_row = top+middle_node[0][0]
        #else:
        #    mid_row = top+middle_node[0][0]-1
        print(f'Middle column: {middle_column}')         
        print(f'Middle edge: {mid_edge}')
        print(f'Middle row: {mid_row}')
        LinearSpaceAlignment(stringA, stringB, top, mid_row, left, middle_column, path)
        path.append(mid_edge)
        #print(mid_edge)
        if ((mid_edge == 'H') or (mid_edge == 'D')):
            if middle_column == 1:
                middle_column = middle_column+1
            else:    
                middle_column = middle_column+1
        if ((mid_edge == 'V') or (mid_edge == 'D')):
            if mid_row == 0:
                mid_row = mid_row+1
            else:
                mid_row = mid_row+1      
        LinearSpaceAlignment(stringA, stringB, mid_row, bottom, middle_column, right, path)  
    return path       

def AlignmentScore(stringA, stringB, alignment):
    #print('**************************')
    #print(len(stringA))
    #print(len(stringB))
    #print(len(path))
    alignmentA = []
    alignmentB = []
    score = 0
    num_V = 0
    num_H = 0
    for direction in range(0,len(alignment)):
        if alignment[direction] == 'V':
            score = score-5
            alignmentA.append(stringA[direction-num_H])
            alignmentB.append('-')
            num_V += 1
        elif alignment[direction] == 'H':
            score = score-5
            alignmentB.append(stringB[direction-num_V])
            alignmentA.append('-')
            num_H += 1    
        elif alignment[direction] == 'D':
            alignmentA.append(stringA[direction-num_H])
            alignmentB.append(stringB[direction-num_V])
            score = score + Blosum62[stringA[direction-num_H]][stringB[direction-num_V]]
    print(''.join(alignmentA))
    print(''.join(alignmentB))    
    return score        

def FloydWarshall(graph, num_leaves):
    num_vertex = graph.shape[0]
    distance_matrix = graph
    for k in range(num_vertex):
        for i in range(num_vertex):
            for j in range(num_vertex):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]        
    return distance_matrix[:num_leaves,:num_leaves]   

def DistanceReversal(Perm):
    num_of_rev = 0
    for i in range(1, len(Perm)+1):
        if Perm[i-1] != i:
            if Perm[i-1] == -i:
                Perm[i-1] = i
                num_of_rev += 1
                tmp_perm = []
                for syn in Perm:
                    if syn > 0:
                        tmp_perm.append("+"+str(syn))
                    else: 
                        tmp_perm.append(str(syn))
                print(' '.join(tmp_perm))
            else:
                if i in Perm:
                    index = Perm.index(i)+1
                else:
                    index = Perm.index(-i)+1
                Perm[i-1:index] = reversed(Perm[i-1:index])   
                Perm[i-1:index] = map(lambda a: -a, Perm[i-1:index])
                num_of_rev += 1
                tmp_perm = []
                for syn in Perm:
                    if syn > 0:
                        tmp_perm.append("+"+str(syn))
                    else: 
                        tmp_perm.append(str(syn))
                print(' '.join(tmp_perm))
                if Perm[i-1] == -i:
                    Perm[i-1] = i
                    num_of_rev += 1 
                    tmp_perm = []
                    for syn in Perm:
                        if syn > 0:
                            tmp_perm.append("+"+str(syn))
                        else: 
                            tmp_perm.append(str(syn))
                    print(' '.join(tmp_perm))            
    return Perm                

def NumOfBreakingPoints(Perm):
    NumOfBreakingPoints = 0
    if Perm[0] != 1:
        NumOfBreakingPoints += 1
    for i in range(1,len(Perm)):
        if Perm[i]-Perm[i-1] != 1:
            NumOfBreakingPoints += 1      
    if (len(Perm)+1)-Perm[-1] != 1:
        NumOfBreakingPoints += 1
    return NumOfBreakingPoints            


def FileToListSeperatedInt(filename, skip=0, sep=' '):
    """
    Returns list form file. Each entry is sepearted by a seperator. Works for integer entries.
    """
    f = open('./' + filename)
    for _ in range(0,skip):
        next(f)
    l = []  
    for line in f:
        l.append(list(map(int,line.strip().split(sep))))
    return l

def FileToSortedGraph(filename, skip=0):
    f = open('./' + filename)
    graph = dict()
    for _ in range(0,skip):
        next(f)
    for line in f:
        (key, val) = line.strip().split('->')
        graph[int(key)] = list(map(int, list(val.strip().split(':'))))
    graph_sorted = {}
    entries = list(graph.keys())
    entries.sort()
    for node in entries:
        graph_sorted[node] = graph[node]
    return graph_sorted

def FileToGraphMatrix(filename, skip=0):
    f = open('./' + filename)
    all_vertices = []
    for _ in range(0,skip):
        next(f)
    for line in f:
        all_vertices.append(int(line.strip().split('->')[0]))
    num_vertices = max(all_vertices)
    graph = np.zeros((num_vertices+1, num_vertices+1))
    for i in range(num_vertices+1):
        for j in range(num_vertices+1):
            if j == i:
                graph[i][j] = 0
            else:    
                graph[i][j] = math.inf
    f.seek(0)
    for _ in range(0,skip):
        next(f)
    for line in f:
        vert0 = int(line.strip().split('->')[0])
        vert1 = int(line.strip().split('->')[1].split(':')[0])
        weight = int(line.strip().split('->')[1].split(':')[1])
        graph[vert0][vert1] = weight
    return graph

def FileToPermutation(filename):
    f = open('./' + filename)
    l = []  
    for line in f:
        l = list(map(int,line.strip().split(' ')))
    return l

stringA_and_StringB = FileToList('dataset_250_14.txt')
stringA = stringA_and_StringB[0]
stringB = stringA_and_StringB[1]
#graph = FileToSortedGraph('dataset_245_7.txt', 2)

Permutation = FileToPermutation('dataset_287_6.txt')
print(NumOfBreakingPoints(Permutation))

#print(FindMiddleNode('FP', 'P'))
""" graph = FileToGraphMatrix('dataset_10328_12.txt',1)
distance_matrix = FloydWarshall(graph,32)
f = open('output.txt','w')
for dist in distance_matrix:
    row = []
    for i in dist:
        row.append(str(int(i)))
    f.write(f"{' '.join(row)} \n")   """  
#f.write(LocalAlignmentProblem(stringB,stringA))
#f.close()
#print(LongestPathOfDAG(graph, 0, 49))
#print(OutputLCS(LCSBackTrack(stringA, stringB), stringA, len(stringA), len(stringB)))  