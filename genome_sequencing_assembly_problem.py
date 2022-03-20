#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import random
import copy
import math
import itertools
from finding_hidden_messages_in_dna_motif_problem import FileToList, FileToListSeperated

def Composition(k, text):
    comp = []
    for i in range(0, len(text)-k+1):
        comp.append(text[i:i+k])
    #comp.sort()
    return comp 

def PathToGenome(Path):
    genome = []
    genome.append(Path[0])
    for read_index in range(1,len(Path)):
        genome.append(Path[read_index][-1])
    return genome   

def Suffix(kmer):
    return kmer[1:]

def Prefix(kmer):
    return kmer[:-1]

def Overlap(Patterns):
    overlap = dict()
    for pat in Patterns:
        for p in Patterns:
            if Suffix(pat) == Prefix(p):
                if not pat in overlap:
                    overlap[pat] = [p]
                else:
                    overlap[pat].append(p)
    return overlap                  

def DebruijnFromText(k, text):
    graph = dict()
    sorted_graph = dict()
    for kmer_start in range(0, len(text)-k+1):
        kmer = text[kmer_start:kmer_start+k]
        if not Prefix(kmer) in graph:
            graph[Prefix(kmer)] = [Suffix(kmer)]
        else:
            graph[Prefix(kmer)].append(Suffix(kmer))
    for key in sorted(graph):
        sorted_graph[key] = graph[key]
    for key, value in sorted_graph.items():
        value.sort()
        sorted_graph[key] = value
    return sorted_graph

def DebruijnFromPatterns(patterns):
    graph = dict()
    sorted_graph = dict()
    for kmer in patterns:
        if Prefix(kmer) not in graph:
            graph[Prefix(kmer)] = [Suffix(kmer)]
        else:
            graph[Prefix(kmer)].append(Suffix(kmer))
    for key in sorted(graph):
        sorted_graph[key] = graph[key]
    for key, value in sorted_graph.items():
        value.sort()
        sorted_graph[key] = value
    return sorted_graph

def EulerianCycle(graph):
    i = 0
    nodes = list(graph.keys())
    num_edges = 0
    for keys in graph:
        for item in graph[keys]:
            num_edges += 1
    while True:
        stack = []
        circuit = []
        test_graph = copy.deepcopy(graph)
        if i < len(nodes):
            start = nodes[i]
            current_node = start 
            while True:
                if current_node in test_graph.keys():
                    if bool(test_graph[current_node]):
                        stack.append(current_node)
                        current_node = test_graph[current_node].pop()
                    else:
                        circuit.append(current_node)
                        test_graph.pop(current_node, None)
                        if bool(stack):
                            current_node = stack.pop()
                else:
                    circuit.append(current_node)
                    if bool(stack):
                        current_node = stack.pop()   
                if (not bool(test_graph)) and  (not bool(stack)):
                    break                                             
            if len(circuit) == num_edges:       
                circuit.append(start)
                break
            else:
                i+=1       
        else:
            return '0'  
    return circuit[::-1]   

def EulerianPath(graph):
    start_found = 0
    end_found = 0
    starting_node = 0
    end_node = 0
    nodes = list(graph)
    num_edges = 0
    for keys in graph:
        for item in graph[keys]:
            num_edges += 1
    i = 0
    for i in range(0, len(graph)): # find the starting node, where out degree minus in degree is one
        key = nodes[i]
        out_deg = len(graph[key])
        in_deg = 0
        for keys in graph:
            for item in graph[keys]:
                if item == key:
                    in_deg += 1           
        if (out_deg-in_deg) == 1:
            starting_node = key
            start_found = 1
        elif (out_deg-in_deg) == -1:
            end_node = key
            end_found = 1
        if bool(start_found and end_found):
            break
    if not end_node:
        for keys in graph:
            for item in graph[keys]:
                if not (item in graph):
                    end_node = item                    
    stack = []
    circuit = []
    test_graph = copy.deepcopy(graph)
    current_node = starting_node
    while True:
        if current_node in test_graph.keys():
            if bool(test_graph[current_node]):
                stack.append(current_node)
                current_node = test_graph[current_node].pop()
            else: 
                circuit.append(current_node)
                test_graph.pop(current_node, None)
                if bool(stack):
                    current_node = stack.pop()
        else:
            circuit.append(current_node)
            if bool(stack):
                current_node = stack.pop()  
        if not bool(test_graph):
            break                                                         
    if len(circuit) == num_edges:      
        circuit.append(starting_node)            
    return circuit[::-1]

def StringReconstruction(patterns):
    db = DebruijnFromPatterns(patterns)
    path = EulerianPath(db)
    text = PathToGenome(path)
    return text

def KUniversalCycle(k):
    set =[]
    for tub in list(itertools.product([0, 1], repeat=k)):
        set.append(''.join(map(str, tub)))
    db = DebruijnFromPatterns(set)
    path = EulerianCycle(db)
    text = PathToGenome(path)
    return text[1:]

def SuffixPair(KmerPair):
    tup_suffix = ()
    tup_suffix = tup_suffix + (KmerPair[0][1:],)
    tup_suffix = tup_suffix + (KmerPair[1][1:],)
    return tup_suffix


def PrefixPair(KmerPair):
    tup_pre = ()
    tup_pre = tup_pre + (KmerPair[0][:-1],)
    tup_pre = tup_pre + (KmerPair[1][:-1],)
    return tup_pre

def DebruijnPair(allKmerpairs):
    graph = dict()
    sorted_graph = dict()
    for kmerpair in allKmerpairs:
        if PrefixPair(kmerpair) in graph.keys():
            graph[PrefixPair(kmerpair)].append(SuffixPair(kmerpair))
        else:
            graph[PrefixPair(kmerpair)] = [SuffixPair(kmerpair)]
    for key in sorted(graph):
        sorted_graph[key] = graph[key]
    for key, value in graph.items():
        value.sort() 
        sorted_graph[key] = value
    return sorted_graph         

def EulerianPathPair(graph, d):
    start_found = 0
    end_found = 0
    starting_node = 0
    end_node = 0
    nodes = list(graph)
    num_edges = 0
    for keys in graph:
        for item in graph[keys]:
            num_edges += 1
    i = 0
    for i in range(0, len(graph)): # find the starting node, where out degree minus in degree is one
        key = nodes[i]
        out_deg = len(graph[key])
        in_deg = 0
        for keys in graph:
            for item in graph[keys]:
                if item == key:
                    in_deg += 1           
        if (out_deg-in_deg) == 1:
            starting_node = key
            start_found = 1
        elif (out_deg-in_deg) == -1:
            end_node = key
            end_found = 1
        if bool(start_found and end_found):
            break
    if not end_node:
        for keys in graph:
            for item in graph[keys]:
                if not (item in graph):
                    end_node = item                    
    stack = []
    circuit = []
    test_graph = copy.deepcopy(graph)
    current_node = starting_node
    i = 0
    while True:
        if current_node in test_graph.keys():
            if bool(test_graph[current_node]):
                stack.append(current_node)
                if i > d:
                    for pair in range(0, len(test_graph[current_node])):
                        if test_graph[current_node][pair][0][-1] == stack[-(d+2)][1][0]:
                            prev_node = current_node
                            current_node = test_graph[current_node][pair]
                            del test_graph[prev_node][pair]
                            break
                else:
                    current_node = test_graph[current_node].pop()       
            else: 
                circuit.append(current_node)
                test_graph.pop(current_node, None)
                if bool(stack):
                    current_node = stack.pop()
        else:
            circuit.append(current_node)
            if bool(stack):
                current_node = stack.pop()  
        if not bool(test_graph):
            break 
        i += 1                                                        
    if len(circuit) == num_edges:      
        circuit.append(starting_node)            
    return circuit[::-1]

def PathToGenomePair(Path, d):
    genome = []
    for pair in Path:
        genome.append(pair[0][0])
    for i in range(1, len(Path[-1][0])):
        genome.append(Path[-1][0][i])
    for j in range((d+2),0, -1):
        genome.append(Path[-j][1][0])
    for k in range(1, len(Path[-1][1])):    
        genome.append(Path[-1][1][k])
    return genome

def StringReconstructionPair(allKmerpairs, d):
    graph = DebruijnPair(allKmerpairs)
    print('graph done.')
    path = EulerianPathPair(graph, d)
    print('path done.')
    genome = PathToGenomePair(path, d)
    return genome

def IsOneInOneOut(graph, node):
    out_deg = 0
    if node in graph:
        out_deg = len(graph[node])
    in_deg = 0
    for keys in graph:
        for item in graph[keys]:
            if item == node:
                in_deg += 1
    return ((in_deg == 1) and (out_deg == 1))             

def MaximalNonBranchingPaths(graph):
    paths = []
    nodes = list(graph.keys())
    for i in range(0, len(graph)):
        key = nodes[i]
        out_deg = len(graph[key])
        in_deg = 0
        for keys in graph:
            for item in graph[keys]:
                if item == key:
                    in_deg += 1
        if not((in_deg == 1) and (out_deg == 1)):
            if out_deg > 0:
                for item in graph[key]:
                    nonBranchingPath = []
                    nonBranchingPath.append(key)
                    nonBranchingPath.append(item)
                    while IsOneInOneOut(graph,item):
                        nonBranchingPath.append(graph[item][0])
                        item = graph[item][0]
                    if len(nonBranchingPath) > 1:    
                        paths.append(nonBranchingPath)
    test_graph = copy.deepcopy(graph)                    
    for key in test_graph:
        cycle = []
        while IsOneInOneOut(graph,key):
            cycle.append(key)
            prev_key = key
            key = graph[key][0]
            if bool(test_graph[prev_key]):
                del test_graph[prev_key][0]
            else:
                break    
            if key == cycle[0]:
                cycle.append(key)
                break
        if (len(cycle)>1) and (cycle[0] == cycle[-1]):    
            paths.append(cycle)
    return paths

def contigs(patterns):
    contigs = []
    graph = DebruijnFromPatterns(patterns)
    paths = MaximalNonBranchingPaths(graph)
    for path in paths:
        genome = PathToGenome(path)
        contigs.append(genome)
    return contigs    



def FileToGraph(filename):
    f = open('./' + filename)
    graph = dict()
    for line in f:
        (key, val) = line.strip().split('->')
        graph[int(key)] = list(map(int, list(val.strip().split(','))))
    return graph  

def FileToPairs(filename, skip=0):
    f = open('./' + filename)
    ListOfPairs = []
    for _ in range(0,skip):
        next(f)
    for line in f:
        (kmer1, kmer2) = line.strip().split('|')
        ListOfPairs.append((kmer1, kmer2))
    return ListOfPairs    


patterns = FileToList('dataset_205_5.txt')
contigs = contigs(patterns)
#patterns = FileToList('dataset_203_7.txt', 1)

#overlap = Overlap(patterns)

#graph = DebruijnFromPatterns(patterns)

#graph = FileToGraph('dataset_203_6.txt')

#print('->'.join(list(map(str,EulerianCycle(graph)))))

allKmerpairs = FileToPairs('dataset_6206_4.txt', 1)

print(''.join(StringReconstructionPair(allKmerpairs, 200)))

#f = open('output.txt','w')
#for contig in contigs:
#    f.write(key)
#    f.write(' -> ')
#    f.write(', '.join(value))
#    f.write('\n')
#    f.write(' -> '.join(list(map(str, path))))
#   f.write(''.join(contig))
#    f.write(' ')
#    f.write('\n')
# f.write(''.join(StringReconstructionPair(allKmerpairs, 200)))
#f.write('->'.join(list(map(str,EulerianPath2(graph)))))
#f.close()
