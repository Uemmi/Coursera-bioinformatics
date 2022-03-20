#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import copy
from collections import OrderedDict
from finding_hidden_messages_in_dna_ori_problem import ReverseCompliment
from finding_hidden_messages_in_dna_motif_problem import FileToListSeperated

aminoAcidMass = {
    'G': 57,
    'A': 71,
    'S': 87,
    'P': 97,
    'V': 99,
    'T': 101,
    'C': 103,
    'I': 113,
    'L': 113,
    'N': 114,
    'D': 115,
    'K': 128,
    'Q': 128,
    'E': 129,
    'M': 131,
    'H': 137,
    'F': 147,
    'R': 156,
    'Y': 163,
    'W': 186
}
Extended_Amino_Acid_Mass = dict()
for i in range(57,201):
    Extended_Amino_Acid_Mass[str(i)] = i

Extended_AAs = []
for key in Extended_Amino_Acid_Mass.keys():
    ls = key.split()
    Extended_AAs.append(ls)

AAs = list(map(list, aminoAcidMass.keys()))
AAs.remove(['L'])
AAs.remove(['Q'])

brevis_AAs_int = [147, 128, 113, 114, 97, 99, 186, 163]
brevis_AAs = []
for AA in brevis_AAs_int:
    ls = str(AA).split()
    brevis_AAs.append(ls)

#CodonToAA = {'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAU': 'N', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T', 'AGA': 'R', 'AGC': 'S', 'AGG': 'R', 'AGU': 'S', 'AUA': 'I', 'AUC': 'I', 'AUG': 'M', 'AUU': 'I', 'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAU': 'H', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R', 'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L', 'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAU': 'D', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G', 'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V', 'UAA': 'STOP', 'UAC': 'Y', 'UAG': 'STOP', 'UAU': 'Y', 'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S', 'UGA': 'STOP', 'UGC': 'C', 'UGG': 'W', 'UGU': 'C', 'UUA': 'L', 'UUC': 'F', 'UUG': 'L', 'UUU': 'F'}
CodonToAA = dict()
code = open('./RNA_codon_table_1.txt')
allCodons = code.readlines()
for codon in allCodons:
    ls = codon.strip().split(' ')
    if len(ls) > 1:
        key = ls[0]
        val = ls[1]
        CodonToAA[key] = val
    else:
        key = ls[0]
        val = 'STOP'
        CodonToAA[key] = val

def RNAToProtein(RNA):
    seq = []
    for i in range(0, len(RNA), 3):
        codon = RNA[i:i+3]
        AA = CodonToAA[codon]
        if AA != 'STOP':
            seq.append(AA)
        else:
            break    
    return seq

def Transcription(String):
    mapping = {'A':'A', 'T':'U', 'C':'C', 'G':'G'}
    Compliment = []
    for nuc in String:
        Compliment.append(mapping[nuc])
    return str(''.join(Compliment))

def FindSubstringInDNAForAASeq(AAseq, DNA):
    strings = []
    for i in range(0, len(DNA)-3*len(AAseq)+1):
        stringDNA = []
        stringCompliment = []
        for j in range(0, len(AAseq)):
            AACodonFromDNA = DNA[i+3*j:i+3*j+3]
            AACodonFromReverse = str(''.join(ReverseCompliment(AACodonFromDNA)))
            AAFromDNA = CodonToAA[Transcription(AACodonFromDNA)]
            AAFromReverse = CodonToAA[Transcription(AACodonFromReverse)]
            if AAFromDNA == AAseq[j]:
                stringDNA.append(AACodonFromDNA)
            if AAFromReverse == AAseq[len(AAseq)-1-j]:
                stringCompliment.append(AACodonFromDNA)
        if len(stringDNA) == len(AAseq):
            strings.append(stringDNA)
        if len(stringCompliment) == len(AAseq):
            strings.append(stringCompliment)
        for string_index in range(0, len(strings)):
            strings[string_index] = ''.join(strings[string_index])
    return strings

def LinearSpectrum(Peptide):
    PrefixMass = [0]
    linearspectrum = [0]
    if Peptide == '0':
        return linearspectrum
    for AA_i in range(0, len(Peptide)):
        #PrefixMass.append(PrefixMass[AA_i]+aminoAcidMass[Peptide[AA_i]])
        PrefixMass.append(PrefixMass[AA_i]+Extended_Amino_Acid_Mass[Peptide[AA_i]])
    for AA_i in range(0,len(PrefixMass)-1):
        for AA_j in range(AA_i+1, len(PrefixMass)):
            Mass = PrefixMass[AA_j] - PrefixMass[AA_i]
            linearspectrum.append(Mass)
    linearspectrum.sort()     
    return linearspectrum  

def CyclicSpectrum(Peptide):
    PrefixMass = [0]
    cyclicspectrum = [0]
    if Peptide == '0':
        return cyclicspectrum
    for AA_i in range(0, len(Peptide)):
        #PrefixMass.append(PrefixMass[AA_i]+aminoAcidMass[Peptide[AA_i]])
        PrefixMass.append(PrefixMass[AA_i]+Extended_Amino_Acid_Mass[Peptide[AA_i]])
    peptidemass = PrefixMass[len(Peptide)]
    for AA_i in range(0,len(PrefixMass)-1):
        for AA_j in range(AA_i+1, len(PrefixMass)):
            Mass = PrefixMass[AA_j] - PrefixMass[AA_i]
            cyclicspectrum.append(Mass)
            if ((AA_i > 0) and (AA_j < len(PrefixMass)-1)):
                cyclicspectrum.append(peptidemass - Mass)
    cyclicspectrum.sort()     
    return cyclicspectrum

def Mass(peptide):
    mass = 0
    for AA in peptide:
        #mass += aminoAcidMass[AA]
        mass += Extended_Amino_Acid_Mass[AA]
    return mass

def Expand(Peptides, AAs_list = Extended_AAs):
    new_peptides = []
    for peptide_i in range(0, len(Peptides)):
        for AA in AAs_list:
            peptide = Peptides[peptide_i]
            if peptide == ['0']:
                new_peptides.append(AA)
            else:    
                new_peptide = list(peptide) + AA
                new_peptides.append(new_peptide)
    return new_peptides        

def CyclopeptideSequencing(Spectrum):
    CandidatePeptides = [['0']]
    FinalPeptides = []
    i = 0
    while bool(CandidatePeptides):
        print(i)
        CandidatePeptides = Expand(CandidatePeptides)
        copy_CandidatePeptides = copy.deepcopy(CandidatePeptides)
        for peptide in copy_CandidatePeptides:
            if Mass(peptide) == int(Spectrum[-1]):
                if ((list(map(str,CyclicSpectrum(peptide))) == Spectrum) and (peptide not in FinalPeptides)):
                    FinalPeptides.append(peptide)
                CandidatePeptides.remove(peptide)    
            elif not all(elem in Spectrum for elem in list(map(str,LinearSpectrum(peptide)))):
                CandidatePeptides.remove(peptide)      
        i += 1     
    num_finalpeptides = []    
    for peptide in FinalPeptides:
        num_peptide = []
        for AA in peptide:
            num_peptide.append(str(aminoAcidMass[AA]))
        num_finalpeptides.append(num_peptide)          
    return num_finalpeptides[::-1]

Alphabet = {57: 'G', 71: 'A', 87: 'S', 97: 'P',
            99: 'V', 101: 'T', 103: 'C', 113:'I/L',
            114: 'N', 115: 'D', 128: 'K/Q', 129: 'E',
            131: 'M', 137: 'H', 147: 'F', 156: 'R', 163: 'Y', 186: 'W'}

def CountingMass(Mass, masslist):
    if Mass == 0: return 1, masslist
    if Mass < 57: return 0, masslist
    if Mass in masslist: return masslist[Mass], masslist
    n = 0
    for i in Alphabet:
        k, masslist = CountingMass(Mass - i, masslist)
        n += k
    masslist[Mass] = n
    return n, masslist

#print(CountingMass(2000, {})[0])

def ScoringCyclic(Peptide, Spectrum):
    theoretical_spec = CyclicSpectrum(Peptide)
    copy_Spectrum = copy.deepcopy(Spectrum)
    score = 0
    for mass in theoretical_spec:
        if str(mass) in copy_Spectrum:
            copy_Spectrum.remove(str(mass))
            score += 1
    return score                     

""" def Trim(Spectrum, PeptideBoard, N):
    Score_dict = dict()
    for peptide in PeptideBoard:
        peptide = str(''.join(peptide))
        Score_dict[peptide] = ScoringLinear(peptide, Spectrum)
    Sort_dict = {k: v for k, v in sorted(Score_dict.items(), key=lambda item: item[1], reverse=True)} 
    peptide_list = list(Sort_dict.keys())
    Leader_peptides = peptide_list[:N]
    if bool(Leader_peptides):
        last_N = Leader_peptides[-1]
    while ((N+1 < len(peptide_list)) and (Sort_dict[peptide_list[N+1]] == Sort_dict[last_N])):
        Leader_peptides.append(peptide_list[N+1])
        N += 1
    return Leader_peptides   """

def Trim(Spectrum, PeptideBoard, N):
    Score_dict = dict()
    for peptide in PeptideBoard:
        #peptide = str(''.join(peptide))
        key_name = str(' '.join(peptide))
        Score_dict[key_name] = ScoringLinear(peptide, Spectrum)
    #print(Score_dict)    
    Sort_dict = OrderedDict(sorted(Score_dict.items(), key=lambda t: t[1], reverse=True))
    peptide_list = []
    for key in Sort_dict.keys():
        peptide_list.append(key.split())
    Leader_peptides = peptide_list[:N]
    if bool(Leader_peptides):
        last_N = Leader_peptides[-1]
    while ((N < len(peptide_list)) and (Sort_dict[str(' '.join(peptide_list[N]))] == Sort_dict[str(' '.join(last_N))])):
        Leader_peptides.append(peptide_list[N])
        N += 1
    return Leader_peptides      

def ScoringLinear(Peptide, Spectrum):
    theoretical_spec = LinearSpectrum(Peptide)
    copy_Spectrum = copy.deepcopy(Spectrum)
    score = 0
    for mass_t in theoretical_spec:
        if str(mass_t) in copy_Spectrum:
            copy_Spectrum.remove(str(mass_t))
            score += 1
    return score

def LeaderboardCyclopeptideSequencing(Spectrum, N):
    Leaderboard = [['0']]
    LeaderPeptide = ['0']
    Max_Score = 0
    i = 0
    mass_spec = int(Spectrum[-1])
    while bool(Leaderboard) :
        Leaderboard = Expand(Leaderboard)
        print(len(Leaderboard))
        copy_leaderboard = copy.deepcopy(Leaderboard)
        for peptide in copy_leaderboard:
            mass = Mass(peptide)
            if mass == mass_spec:
                score = ScoringCyclic(peptide, Spectrum)
                #if ScoringCyclic(peptide, Spectrum) > ScoringCyclic(LeaderPeptide, Spectrum):
                if score > Max_Score:
                    LeaderPeptide = [peptide]
                    Max_Score = score
                    #LeaderPeptide = peptide
                elif score == Max_Score:
                    LeaderPeptide.append(peptide)
            elif mass > mass_spec:
                Leaderboard.remove(peptide) 
        print(len(Leaderboard))             
        Leaderboard = Trim(Spectrum, Leaderboard, N)
        print(len(Leaderboard))
        print("--------------")
        i += 1
    #num_peptide = []    
    #for AA in LeaderPeptide:
    #    num_peptide.append(str(aminoAcidMass[AA]))
    """ num_finalpeptides = []    
    for peptide in LeaderPeptide[1:]:
        num_peptide = []
        for AA in peptide:
            num_peptide.append(str(aminoAcidMass[AA]))
        num_finalpeptides.append(num_peptide)          
    return num_finalpeptides """
    return LeaderPeptide
    #return num_peptide  

def Convolution(Spectrum):
    conv = []
    for i in range(len(Spectrum)-1, 0,-1):
        for j in range(i-1, -1, -1):
            diff = int(Spectrum[i])-int(Spectrum[j])
            if ((diff > 56) and (diff < 201)):
                conv.append(diff)
    return conv    

def Filter(ConvSpectrum, M):
    conv = sorted(ConvSpectrum)
    frequency_dict = dict()
    for AA in conv:
        if int(AA) > 200:
            break
        if AA in frequency_dict:
            frequency_dict[AA] += 1
        else:
            frequency_dict[AA] = 1
    Sorted_frequ = OrderedDict(sorted(frequency_dict.items(), key=lambda t: t[1], reverse=True))
    new_AAs_all = list(Sorted_frequ.keys())
    new_AAs_filtered = new_AAs_all[:M]
    last_M = new_AAs_filtered[-1]
    while ((M < len(new_AAs_all)) and (Sorted_frequ[new_AAs_all[M]] == Sorted_frequ[last_M])):
        new_AAs_filtered.append(new_AAs_all[M])
        M += 1
    new_AAs_filtered_list = []
    for AA in new_AAs_filtered:
        ls = AA.split()
        new_AAs_filtered_list.append(ls)    
    return new_AAs_filtered_list

def ConvolutionCyclopeptodeSequencing(Spectrum, N, M):
    conv = list(map(str,Convolution(Spectrum)))
    new_AAs = Filter(conv, M)
    Leaderboard = [['0']]
    LeaderPeptide = ['0']
    LeaderPeptides = []
    list_of_scores = []
    Max_score = 0
    lowest_score_in_list = len(Spectrum)
    i = 0
    mass_spec = int(Spectrum[-1])
    print(mass_spec)
    while bool(Leaderboard) :
        Leaderboard = Expand(Leaderboard, new_AAs)
        print(len(Leaderboard))
        copy_leaderboard = copy.deepcopy(Leaderboard)
        for peptide in copy_leaderboard:
            mass = Mass(peptide)
            if mass == mass_spec:
                score = ScoringCyclic(peptide, Spectrum)
                if len(LeaderPeptides) != 86:
                    LeaderPeptides.append(peptide)
                    list_of_scores.append(score)
                elif len(LeaderPeptides) == 86:
                    min_score = min(list_of_scores)
                    if score > min_score:
                        ind = list_of_scores.index(min_score)
                        list_of_scores.append(score)
                        LeaderPeptides.append(peptide)
                        list_of_scores.remove(min_score)
                        del LeaderPeptides[ind]
            elif mass > mass_spec:
                Leaderboard.remove(peptide) 
        print(len(Leaderboard))             
        Leaderboard = Trim(Spectrum, Leaderboard, N)
        print(len(Leaderboard))
        print("--------------")
        i += 1
    #num_peptide = []    
    #for AA in LeaderPeptide:
    #    num_peptide.append(str(aminoAcidMass[AA]))
    """ num_finalpeptides = []    
    for peptide in LeaderPeptide[1:]:
        num_peptide = []
        for AA in peptide:
            num_peptide.append(str(aminoAcidMass[AA]))
        num_finalpeptides.append(num_peptide)          
    return num_finalpeptides """
    return LeaderPeptides
    #return num_peptide

def CyclopeptodeSequencing_special(Spectrum, N, M):
    Leaderboard = [['0']]
    LeaderPeptide = '0'
    Max_score = 0
    LeaderPeptides = []
    list_of_scores = []
    Max_score = 0
    lowest_score_in_list = len(Spectrum)
    i = 0
    new_spectrum = []
    for mass in Spectrum:
        clean_mass = int(mass[:-2])-1
        new_spectrum.append(str(clean_mass))  
    conv = list(map(str,Convolution(new_spectrum)))
    new_AAs = Filter(conv, M) 
    print(new_AAs)    
    print(brevis_AAs)
    while i<10 :
        print(i)
        Leaderboard = Expand(Leaderboard, new_AAs)
        copy_leaderboard = copy.deepcopy(Leaderboard)
        for peptide in copy_leaderboard:
            mass = Mass(peptide)
            if i == 9:
                score = ScoringCyclic(peptide, new_spectrum)
                if len(LeaderPeptides) != 10:
                    LeaderPeptides.append(peptide)
                    list_of_scores.append(score)
                elif len(LeaderPeptides) == 10:
                    min_score = min(list_of_scores)
                    if score > min_score:
                        ind = list_of_scores.index(min_score)
                        list_of_scores.append(score)
                        LeaderPeptides.append(peptide)
                        list_of_scores.remove(min_score)
                        del LeaderPeptides[ind]
            if mass > int(new_spectrum[-1]):
                Leaderboard.remove(peptide)
        print(len(Leaderboard))             
        Leaderboard = Trim(new_spectrum, Leaderboard, N)
        print(len(Leaderboard))
        print("--------------")    
        i += 1
    #num_peptide = []    
    #for AA in LeaderPeptide:
    #    num_peptide.append(str(aminoAcidMass[AA]))
    """ num_finalpeptides = []    
    for peptide in LeaderPeptide[1:]:
        num_peptide = []
        for AA in peptide:
            num_peptide.append(str(aminoAcidMass[AA]))
        num_finalpeptides.append(num_peptide)          
    return num_finalpeptides """
    return LeaderPeptides
    #return num_peptide    

def FileToSingleString(Filename):
    f = open('./'+Filename)
    Lines = f.readlines()
    string = []
    for line in Lines:
        string.append(line.strip())
    return str(''.join(string)) 

def FileToOneListSeperated(filename, skip=0, sep=' '):
  """
  Returns list form file. Each entry is sepearted by a seperator.
  """
  f = open('./' + filename)
  for _ in range(0,skip):
    next(f)
  l = []  
  for line in f:
    ls = list(line.strip().split(sep))
    for item in ls:
        l.append(item)
  return l

#DNA = FileToSingleString('Bacillus_brevis.txt')
spectrum = FileToOneListSeperated('real_spectrum.txt', 0, ' ')
Peptide= FileToOneListSeperated('Example_peptide.txt', 0, ' ')

#print(' '.join('-'.join(peptide) for peptide in LeaderboardCyclopeptideSequencing(spectrum,1000)))
print('----'.join(' '.join(peptide) for peptide in CyclopeptodeSequencing_special(spectrum, 1600, 20)))
#print(' '.join(CyclopeptodeSequencing_special(spectrum, 1500, 100)))
#print(CyclicSpectrum(Peptide))

#print(CyclicSpectrum(['M', 'I', 'I', 'D', 'S', 'V', 'M', 'N', 'S']))
#print(Mass(['M', 'I', 'I', 'D', 'S', 'V', 'M', 'N', 'S']))