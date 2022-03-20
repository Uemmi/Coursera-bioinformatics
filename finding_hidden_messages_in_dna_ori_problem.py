#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import math

nucleotides = ['A', 'C', 'G', 'T']

def PatternCount(text, pattern):
  """
  Returns number how often pattern appears in text.
  """
  count = 0
  for startingIndex in range(0, len(text)-len(pattern)+1):
    if text[startingIndex:startingIndex+len(pattern)] == pattern:
      count += 1
  return count

def SymbolToNumber(symbol):
  if symbol == 'A':
    return 0
  elif symbol == 'C':
    return 1
  elif symbol == 'G':
    return 2
  else:
    return 3  

def PatternToNumber(pattern):
  if not pattern:
    return 0
  symbol = pattern[len(pattern)-1]
  prefix = pattern[0:len(pattern)-1]
  return 4 * PatternToNumber(prefix) + SymbolToNumber(symbol)  

def NumberToPattern(i, k):
  seq = []
  while i > 0:
    rest = i%4
    seq.append(rest)
    i = i//4
  while len(seq) != k:
    seq.insert(len(seq), 0)
  string = ''.join(map(str, seq))  
  pattern = str(string).translate(str.maketrans('0123', 'ACGT'))
  return pattern[::-1]

def ComputingFrequencies(text, k):
  """
  Returns a list with frequencies of all lexiographically ordered k-mers in text.
  """
  frequencyArray = [0]*int(math.pow(4, k))
  for i in range(0, len(text)-k+1):
    pattern = text[i:i+k]
    j = PatternToNumber(pattern)
    frequencyArray[j] += 1
  return frequencyArray

def FrequentWords(text, k):
  """
  Returns list with most frequent k-mers in text.
  """
  frequentPatterns = []
  count = []
  for startingIndex in range(0, len(text)-k+1):
    pattern = text[startingIndex:startingIndex+k]
    count.append(PatternCount(text, pattern))
  maxCount = max(count)
  for i in range(0, len(text)-k+1):
    if count[i] == maxCount:
      frequentPatterns.append(text[i:i+k])
  frequentPatterns = list(dict.fromkeys(frequentPatterns))
  return frequentPatterns

def FasterFrequentWords(text, k):
  """
  Faster version of FrequentWords(). 
  """
  frequentPatterns = []
  frequencyArray = ComputingFrequencies(text, k)
  maxCount = max(frequencyArray)
  for i in range(0, int(math.pow(4,k))-1):
    if frequencyArray[i] == maxCount:
      pattern = NumberToPattern(i, k)
      frequentPatterns.append(pattern)
  return frequentPatterns 

def FindingFrequentWordsBySorting(text , k):
  """
  Returns list with most frequent k-mers in text.
  """
  frequentPatterns = []
  index = [0]*(len(text)-k + 1)
  count = [0]*(len(text)-k + 1)
  for i in range(0, len(text)-k+1):
    pattern = text[i:i+k]
    index[i] = PatternToNumber(pattern)
    count[i] = 1
  index.sort()
  sortedIndex = index
  for i in range(1, len(text)-k+1):
    if sortedIndex[i] == sortedIndex[i-1]:
      count[i] = count[i-1] + 1
  maxCount = max(count)
  for i in range(0, len(text)-k+1):
    if count[i] == maxCount:
      pattern = NumberToPattern(sortedIndex[i], k)
      frequentPatterns.append(pattern)
  return frequentPatterns    

def ReverseCompliment(DNA):
  """
  Returns the reverse compliment of DNA.
  """
  mapping = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
  Compliment = []
  for nuc in DNA:
    Compliment.append(mapping[nuc])
  return ''.join(Compliment[::-1])

def PatternOccuranceInGenome(genome, pattern):
  """
  Returns the index of occurances of pattern in genome.
  """
  places = []
  for i in range(0, len(genome)-len(pattern)+1):
    if genome[i:i+len(pattern)] == pattern:
      places.append(i)
  return places  

def FindingClumps(genome, k, L, t):
  """
  Returns k-mers that form clumps of at least t occurances in a subpart of genome of length L.
  """
  frequentPatterns = []
  clump = [0] * int(math.pow(4,k))
  for i in range(0, len(genome)-L):
    text = genome[i:i+L]
    frequencyArray = ComputingFrequencies(text, k)
    for index in range(0, int(math.pow(4,k)) - 1):
      if frequencyArray[index] >= t:
        clump[index] = 1
  for index in range(0, int(math.pow(4,k))-1):
    if clump[index] == 1:
      pattern = NumberToPattern(index, k)
      frequentPatterns.append(pattern)
  return frequentPatterns

def BetterClumpFinding(genome, k, L, t):
  """
  Better version of FindingCLumps(). 
  Returns k-mers that form clumps of at least t occurances in a subpart of genome of length L.
  """
  frequentPatterns = []
  clump = [0] * int(math.pow(4,k))
  text = genome[0:L]
  frequencyArray = ComputingFrequencies(text, k)
  for i in range(0, int(math.pow(4,k))-1):
    if frequencyArray[i] >= t:
      clump[i] = 1           
  for i in range(1, len(genome)-L+1):
    firstPattern = genome[i-1:i-1+k]
    index = PatternToNumber(firstPattern)
    frequencyArray[index] -= 1
    lastPattern = genome[i+L-k:i+L]
    index = PatternToNumber(lastPattern)
    frequencyArray[index] += 1
    if frequencyArray[index] >= t:
      clump[index] = 1         
  for i in range(0, int(math.pow(4,k))):
    if clump[i] == 1:
      pattern = NumberToPattern(i, k)
      frequentPatterns.append(pattern)
  return frequentPatterns 

def Skewi(genome):
  """
  Returns list of GC ratio for every point in genome.
  """
  skewLsit = []
  skew = 0
  skewLsit.append(skew)
  for i in genome:
    if i == 'G':
      skew += 1
      skewLsit.append(skew)
    elif i == 'C':
      skew -= 1
      skewLsit.append(skew)
    else:
      skew = skew
      skewLsit.append(skew)
  return skewLsit 

def MinSkew(genome):
  """
  Returns list of all indexes in genome where GC ratio is minimum.
  """
  skewList = Skewi(genome)
  minIndex = []
  minValue = min(skewList)
  minFirstIndex = skewList.index(minValue)
  minIndex.append(minFirstIndex)
  restList = skewList[minFirstIndex+1:len(skewList)]
  for i in restList:
    if i == minValue:
      minIndex.append(minFirstIndex+restList.index(i)+1)
  return minIndex 

def HammingDistance(p, q):
  """
  Returns number of differences in p and q.
  """
  differences = 0
  if len(p) != len(q):
    return 'k-mers not same length!'     
  else:
    for i in range(0,len(p)):
      if p[i] != q[i]:
        differences += 1
  return differences

def ApproxOccuranceInText(pattern, text, d):
  """
  Returns list of indexes of all approx. occurances of pattern in text with a max. difference of d.
  """
  occurance = []
  for index in range(0,len(text)-len(pattern)+1):
    if HammingDistance(text[index:index+len(pattern)], pattern) <= d:
      occurance.append(index)
  return occurance

def ApproxPatternCount(pattern, text, d):
  """
  Returns number of occurances of pattern in text with tolerance d.
  """
  count = 0
  for index in range(0,len(text)-len(pattern)+1):
    if HammingDistance(text[index:index+len(pattern)], pattern) <= d:
      count += 1
  return count    

def Neighbors(pattern, d):
  """
  Returns list of all patterns that are within distance d of pattern.
  """
  if d == 0:
    return pattern
  if len(pattern) == 1:
    return nucleotides
  neighborhood = []
  SuffixNeighbors = Neighbors(pattern[1:], d)
  for text in SuffixNeighbors:
    if HammingDistance(pattern[1:], text) < d:
      for x in nucleotides:
        neighborhood.append(x+text)
    else:
      neighborhood.append(pattern[0]+text)
  return neighborhood   

def FrequentWordsWithMismatches(text, k, d):
  """
  Returns list of frequent pattern in text with length k including patterns that are within d distance of the k-mer.
  """
  frequentPatterns = []
  neighborhoods = []
  for i in range(0, len(text)-k+1):
    neighborhoods.append(Neighbors(text[i:i+k], d))
  neighborhoodsConcat = []
  for sublist in neighborhoods:
    for item in sublist:
      neighborhoodsConcat.append(item)  
  index = [0]*len(neighborhoodsConcat)
  count = [0]*len(neighborhoodsConcat)    
  for i in range(0, len(neighborhoodsConcat)):
    pattern = neighborhoodsConcat[i]
    index[i] = PatternToNumber(pattern)
    count[i] = 1 
  index.sort()
  for i in range(0, len(neighborhoodsConcat)-1):
    if index[i] == index[i+1]:
      count[i+1] = count[i] + 1
  maxCount = max(count)
  for i in range(0, len(neighborhoodsConcat)):
    if count[i] == maxCount:
      frequentPatterns.append(NumberToPattern(index[i], k))
  return frequentPatterns

def FrequentWordsWithMismatchesAndReverseComp(text, k, d):
  """
  Returns list of frequent pattern in text with length k including patterns that are within d distance or a reverse compliment of the k-mer.
  """
  frequentPatterns = []
  neighborhoods = []
  for i in range(0, len(text)-k+1):
    neighborhoods.append(Neighbors(text[i:i+k], d)) 
  neighborhoodsConcat = []
  for sublist in neighborhoods:
    for item in sublist:
      neighborhoodsConcat.append(item)
  reverseNeighborhoodConcat = map(ReverseCompliment, neighborhoodsConcat)
  fullNeighborhoodConcat = list(reverseNeighborhoodConcat) + neighborhoodsConcat
  index = [0]*2*len(neighborhoodsConcat)
  count = [0]*2*len(neighborhoodsConcat)    
  for i in range(0, 2*len(neighborhoodsConcat)):
    pattern = fullNeighborhoodConcat[i]
    index[i] = PatternToNumber(pattern)
    count[i] = 1
  index.sort()
  for i in range(0, 2*len(neighborhoodsConcat)-1):
    if index[i] == index[i+1]:
      count[i+1] = count[i] + 1
  maxCount = max(count)
  for i in range(0, 2*len(neighborhoodsConcat)):
    if count[i] == maxCount:
      frequentPatterns.append(NumberToPattern(index[i], k))
  return frequentPatterns          
  

""" with open('./E_coli.txt') as f:
  genome = f.readline() """

#result = MotifEnumerationUseless(['AATTCATAAAATTTGGGCTGCCCGT', 'ATTCGGTTTACACCCCGCCGGGATA', 'CCTGACGCCGACGACAGAACTGACT', 'TGCCGGCCTACTATCTTGTTTCACC', 'AGCGTACCCGCAGAAGGCTGTTTGA', 'TGCAGTATCTGGACGAAGGCCTGGC'], 5, 2)

#print(result)
#print(' '.join(map(str, result)))
