#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import math
import random
from random import choices
import numpy as np
from finding_hidden_messages_in_dna_ori_problem import Neighbors, nucleotides, ApproxPatternCount, NumberToPattern, HammingDistance, PatternToNumber

def MotifEnumerationUseless(DNA, k, d):
  """
  Returns a list of patterns with length k (considering patterns with distance d) that all subDNAs in DNA (list) have common.
  """  
  patterns = []
  for i in range(0, len(DNA[0])-k+1):
    pattern = DNA[0][i:i+k]
    neighbors = Neighbors(pattern, d)
    if neighbors == pattern:
      count = 0
      for subdna in DNA:
        if ApproxPatternCount(pattern, subdna, d) > 0:
          count += 1
      if count >= len(DNA):
        patterns.append(pattern)
    else:   
      for neigh in neighbors:
        count = 0
        for subdna in DNA:
          if ApproxPatternCount(neigh, subdna, d) > 0:
            count += 1
        if count >= len(DNA):
          patterns.append(neigh)
  patterns = list(dict.fromkeys(patterns))
  return patterns

def Count(motifs):
  """
  Returns matrix 4x # of motifs which contains the count of each nucleotide in each column.
  """
  lengthOfMotifs = len(motifs[0])
  # count = [[0 for x in motifs] for y in nucleotides]
  count = np.zeros( (len(nucleotides), lengthOfMotifs))
  for nuc in nucleotides:
    for letterindex in range(0, lengthOfMotifs):
      for motif in motifs:
        if motif[letterindex] == nuc:
          count[nucleotides.index(nuc)][letterindex] += 1
  return count 

def PseudoCount(motifs):
  """
  Returns matrix 4x # of motifs which contains the pseudocount of each nucleotide in each column (1 is added to all entries).
  """
  lengthOfMotifs = len(motifs[0])
  # count = [[0 for x in motifs] for y in nucleotides]
  count = np.ones( (len(nucleotides), lengthOfMotifs))
  for nuc in nucleotides:
    for letterindex in range(0, lengthOfMotifs):
      for motif in motifs:
        if motif[letterindex] == nuc:
          count[nucleotides.index(nuc)][letterindex] += 1
  return count   

def Profile(motifs):
  """
  Returns matrix 4x # of motifs which contains the relative number of each nucleotide in each column from the count.
  """
  count = Count(motifs)
  t = len(motifs)
  profile = count/t
  return profile

def PseudoProfile(motifs):
  """
  Returns matrix 4x # of motifs which contains the relative number of each nucleotide in each column from the pseudocount.
  (no zero probability)
  """
  count = PseudoCount(motifs)
  t = len(motifs)+4
  profile = count/t
  return profile 

def Consensus(motifs):
  """
  Returns a list (string) of a DNA string that is the most probably form the motifs.
  """
  profile = Profile(motifs)
  consense = []
  for letterindex in range(0, len(motifs[0])):
    maxIndex = np.where(profile[:,letterindex] == max(profile[:,letterindex]))
    maxIndexInt = maxIndex[0][0].item()
    consense.append(NumberToPattern(maxIndexInt, 1))
  return consense

def Entropy(motifs):
  """
  Returns the combined entropy of each column in the motifs compared to the consensus.
  """
  profile = Profile(motifs)
  entropy = 0
  for nucIndex in range(0, len(nucleotides)):
    for letterIndex in range(0, len(motifs[0])):
      value = profile[nucIndex][letterIndex]
      if value != 0:
        entropy += (math.log(value)/math.log(2))*value
  return -entropy 

def MedianString(DNA, k):
  """
  Returns k-mer which had the least distance averaged to all DNA strings.
  """
  dist = math.inf
  first_pattern = ''.join(NumberToPattern(0, k))
  all_patterns = Neighbors(first_pattern, k)
  for pattern in all_patterns:
    d = Distance(pattern, DNA)
    if dist > d:
      dist = d
      median = pattern
  return median    


def Distance(pattern, DNA):
  """
  Looks for mininum distance to pattern in all DNA strings and returns the sum of the minimum distances.
  """
  dist = 0
  k = len(pattern)
  for string in DNA:
    for index in range(0, len(string)-k+1):
      if index == 0:
        min_dist = HammingDistance(pattern, string[index:index+k])
      elif HammingDistance(pattern, string[index:index+k]) < min_dist:
        min_dist = HammingDistance(pattern, string[index:index+k])
    dist += min_dist
  return dist 

def ProfilMostProbableKmer(text, k, profile): 
  """
  Returns the most probable k-mer in text, given the profile.
  """
  maxProb = 1
  mostProbPattern = text[:k]
  for i in range(0,k):
      nucIndex = PatternToNumber(mostProbPattern[i])
      probNuc = profile[nucIndex][i]
      maxProb *= probNuc
  for startingIndex in range(1, len(text)-k+1):
    pattern = text[startingIndex:startingIndex+k]
    prob = 1
    for i in range(0,k):
      nucIndex = PatternToNumber(pattern[i])
      probNuc = profile[nucIndex][i]
      prob *= probNuc
    if prob > maxProb:
      maxProb = prob
      mostProbPattern = pattern
  return mostProbPattern

def ProfileMostProbableKmer_withDict(text, k, profile):
  """
  Returns the most probable k-mer in text, given the profile as a dictionary.
  """
  maxProb = 1
  mostProbPattern = text[:k]
  for i in range(0,k):
      probNuc = profile[mostProbPattern[i]][i]
      maxProb *= probNuc
  for startingIndex in range(1, len(text)-k+1):
    pattern = text[startingIndex:startingIndex+k]
    prob = 1
    for i in range(0,k):
      probNuc = profile[pattern[i]][i]
      prob *= probNuc
    if prob > maxProb:
      maxProb = prob
      mostProbPattern = pattern
  return mostProbPattern  

def GreedyMotifSearch(DNA, k, t):
  """
  Returns list of best matching k-mers in all strings in DNA. (t= number of strings)
  """
  BestMotifs = []
  for subDNA in DNA:
    BestMotifs.append(subDNA[0:k])
  for firstIndex in range(0, len(DNA[0])-k+1):
    Motifs = []
    Motifs.append(DNA[0][firstIndex:firstIndex+k])
    for otherDNAindex in range(1, t):
      profile = PseudoProfile(Motifs[:otherDNAindex])
      Motifs.append(ProfilMostProbableKmer(DNA[otherDNAindex], k, profile))
    if Score(Motifs) < Score(BestMotifs):
      BestMotifs = Motifs
  return BestMotifs    

def Score(motifs):
  """
  Returns int for score of motif matrix. The more conserved the motifs the smaller the score.
  """
  score = 0
  consense = Consensus(motifs)
  for motif in motifs:
    for nucIndex in range(0, len(motifs[0])):
      if motif[nucIndex] != consense[nucIndex]:
        score += 1
  return score  

def Motifs(Profile, DNA):
  motifs = []
  for text in DNA:
    motifs.append(ProfilMostProbableKmer(text, len(Profile[0]), Profile))
  return motifs  

def RandomizedMotifSearch(DNA, k, t, N):
  """
  Starting with random Motifs it finds the best matching k-mers in DNA.
  """
  Result = []
  motifs = []
  BestScore = math.inf
  for i in range(0,N):
    print(i)
    BestMotifs = []
    for string in DNA:
      startingPoint = random.randint(0,len(DNA[0])-k)
      BestMotifs.append(string[startingPoint:startingPoint+k])
    while True:
      profile = PseudoProfile(BestMotifs)
      motifs = Motifs(profile, DNA)
      CurrentScore = Score(BestMotifs)
      if Score(motifs) < CurrentScore:
        BestMotifs = motifs[:]
      else:
        break 
    if CurrentScore < BestScore:
      Result = BestMotifs[:]
      BestScore = CurrentScore
      print(BestScore)
  return Result

def Random(prob):
  integers = list(range(0,len(prob)))
  sum = 0
  for p in prob:
    sum += p
  if sum != 1:
    prob = list(map(lambda x: x/sum, prob))
  value = choices(integers, prob)
  return value[0]

def Pr(pattern, profile, k):
  probabilities = []
  for i in range(0, len(pattern)-k):
      kmer = pattern[i:i+k]
      prob = 1
      for j in range(0,len(kmer)):
        nuc = kmer[j]
        num = PatternToNumber(nuc)
        prob *= profile[num,j]
      probabilities.append(prob)
  return probabilities      

def GibbsSampler(DNA, k, t, N):
  BestMotifs = []
  list_t = []
  for _ in range(0,t):
    list_t.append(1)
  #for string in DNA:
  #  startingPoint = random.randint(0,len(DNA[0])-k-1)
  #  BestMotifs.append(string[startingPoint:startingPoint+k])
  BestMotifs = RandomizedMotifSearch(DNA, k, t, 100)
  for j in range(0,N):
    print(j)
    i = Random(list_t)
    Motifs = BestMotifs[:]
    del Motifs[i]
    profile = PseudoProfile(Motifs)
    new_index = Random(Pr(dna[i],profile,k))
    new_motifi = dna[i][new_index:new_index+k]
    Motifs.insert(i, new_motifi)
    if Score(Motifs) < Score(BestMotifs):
      BestMotifs = Motifs[:]
      print(Score(BestMotifs))
  return BestMotifs    



def FileToList(filename, skip=0):
  """
  Returns list from file. Each line is a new entry.
  """
  f = open('./' + filename)
  for _ in range(0,skip):
    next(f)
  return f.read().splitlines()

def FileToListSeperated(filename, skip=0, sep=' '):
  """
  Returns list form file. Each entry is sepearted by a seperator.
  """
  f = open('./' + filename)
  for _ in range(0,skip):
    next(f)
  l = []  
  for line in f:
    l.append(list(line.strip().split(sep)))
  return l


#dna = FileToList('dataset_163_4.txt', 1)
#dna = ['CGCCCCTCTCGGGGGTGTTCAGTAACCGGCCA','GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG','TAGTACCGAGACCGAAAGAAGTATACAGGCGT','TAGATCAAGTTTCAGGTGCACGTCGGTGAACC','AATCCACCAGCTCCACGTGCAATGTTGGCCTA']

#print(FileToListSepBySpace('dataset_5164_1.txt', 1))
#print(Distance('TTTTGTG', dna))

#prof = {'A': [0.2, 0.4, 0.6], 'C': [0.1, 0.5, 0.3], 'T': [0.7, 0.2, 0.3], 'G': [0.1, 0.3, 0.8]}
#num = 3
#tex='ATCTGTTAAACTGA'
#print('\n'.join(GibbsSampler(dna, 15, 20, 10000)))

""" [[0.237, 0.25, 0.276, 0.316, 0.237, 0.211, 0.289, 0.329, 0.355, 0.276, 0.211, 0.289, 0.276],
[0.276, 0.171, 0.158, 0.171, 0.224, 0.25, 0.276, 0.158, 0.25, 0.224, 0.276, 0.145, 0.211],
[0.303, 0.276, 0.289, 0.276, 0.329, 0.263, 0.25, 0.316, 0.197, 0.25, 0.276, 0.276, 0.263],
[0.184, 0.303, 0.276, 0.237, 0.211, 0.276, 0.184, 0.197, 0.197, 0.25, 0.237, 0.289, 0.25]]  """