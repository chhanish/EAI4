###################################
# CS B551 Spring 2021, Assignment #3
#
#  Your names and user ids: gsandeep-hachid-hadank
#
# (Based on skeleton code by D. Crandall)
#


import random
from collections import Counter
import math



# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.CounterOfWords = {}
        self.POSnumber = {}
        self.initProbability = {}
        self.CollectionRevPOS = {}
        self.CollectionBayesProbability = {}
        self.CollectionPos = {}
        self.CollectionEmissionProbability = {}
        self.CollectionTranProbability = {}
       
        
# Calculate the log of the posterior probability of a given sentence
#  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        postCutValue=0.00000000000000000001
        if model == "Simple":
            posterierProbability , lenSentence , i = 0.0 , len(sentence) , 0
            while (i < lenSentence):
                if (sentence[i], label[i]) not in self.CollectionEmissionProbability:
                    posterierProbability =posterierProbability+ math.log(postCutValue)
                else:
                    posterierProbability =posterierProbability+ math.log(self.CollectionEmissionProbability[(sentence[i], label[i])])
                i+=1
            return posterierProbability
        elif model == "HMM":
            posterierProbability , lenSentence , i = 0.0 , len(sentence) , 0
            while (i < lenSentence):
                if (sentence[i], label[i]) not in self.CollectionEmissionProbability:
                    posterierProbability =posterierProbability+ math.log(postCutValue)
                else:
                    posterierProbability =posterierProbability+ math.log(self.CollectionEmissionProbability[(sentence[i], label[i])])
                i+=1
            previous_pos , i = label[0] , 1
            while (i < len(label)):
                posterierProbability += math.log(self.CollectionTranProbability[(previous_pos, label[i])])
                previous_pos = label[i]
                i+=1
            return posterierProbability
        elif model == "Complex":          
            posterierProbability = 0.0
            l , i= len(sentence), 0
            while (i < l):
                if (sentence[i], label[i]) not in self.CollectionEmissionProbability:
                    posterierProbability =posterierProbability+ math.log(0.00000000000000000001)
                else:
                    posterierProbability =posterierProbability+ math.log(self.POSnumber[label[i]] * self.CollectionEmissionProbability[(sentence[i], label[i])])
                i+=1
                
            previous_pos = label[0]
            i = 1
            while (i < len(label)):
                posterierProbability += math.log(self.CollectionTranProbability[(previous_pos, label[i])])
                previous_pos = label[i]
                i+=1
            return posterierProbability
        else:
            print("Unknown algo!")


    # Do the training!
    def train(self, data):
        self.initProbability = {'adj': 0,'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
        self.POSnumber = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
        self.CollectionPos = {'adj': [], 'adv': [], 'adp': [], 'conj': [], 'det': [], 'noun': [], 'num': [], 'pron': [], 'prt': [], 'verb': [], 'x': [], '.': []}
        self.CollectionTranProbability , count = {} , 0
        for meta in data :
            if (len(meta[ 1 ]) > 1) :
                for i in range(0,len( meta[ 1 ])-1) :
                    prev, nextt = meta[ 1 ][ i ], meta[ 1 ][ i + 1 ]
                    self.initProbability[ prev ] += 1
                    if ( prev , nextt ) in self.CollectionTranProbability:
                        self.CollectionTranProbability[ ( prev , nextt ) ] += 1
                    else:
                        self.CollectionTranProbability.update( { ( prev , nextt ) : 1 } )
                    self.CollectionPos[ meta[ 1 ][ i ] ].append( meta[ 0 ][ i ] )
                self.CollectionPos[meta[1][i + 1]].append(meta[0][i + 1])
                self.initProbability[meta[1][i + 1]] += 1
        total = sum(self.initProbability.values())
        for meta in self.initProbability.keys():
            self.initProbability[meta] /= total
        for meta in data:
            for pos in meta[1]:
                self.POSnumber[pos] += 1
                count += 1
            for word in meta[0]:
                if word not in self.CounterOfWords:
                    self.CounterOfWords[word] = 1
                else:
                    self.CounterOfWords[word] += 1
        for meta in data:
            i = 0
            while (i < len(meta[0])):
                if meta[0][i] in self.CollectionRevPOS:
                    self.CollectionRevPOS[meta[0][i]].append(meta[1][i])
                else:
                    self.CollectionRevPOS[meta[0][i]] = [meta[1][i]]
                i+=1
        for pos in self.POSnumber:
            self.POSnumber[pos] /=count
        for word in self.CounterOfWords:
            self.CounterOfWords[word] /=count
        for pos, words in self.CollectionPos.items():
            total , count = 0 , Counter(words)
            count = Counter(th for th in count.elements())
            for elem, co in count.items():
                total += co
            for elem, co in count.items():
                self.CollectionEmissionProbability.update({(elem, pos): co / total})
        total = sum(self.CollectionTranProbability.values())
        for i in self.CollectionTranProbability:
            self.CollectionTranProbability[i] = (float(self.CollectionTranProbability[i] / total ))
        pos , tem = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.'] , []
        for j in pos:
            for i in pos:
                tem.append((i,j))
        lis = list(self.CollectionTranProbability)
        tp_min = min(self.CollectionTranProbability.values())
        for i in list(set(tem)-set(lis)):
            self.CollectionTranProbability[i] = tp_min/20.0
        for word, pos in self.CollectionRevPOS.items():
            total , counts = 0 , Counter(pos)
            counts = Counter(th for th in counts.elements())
            for elem, co in counts.items():
                total += co
            for elem, co in counts.items():
                if word in self.CollectionBayesProbability:
                    self.CollectionBayesProbability[word].append([co / total, elem])
                else:
                    self.CollectionBayesProbability.update({word: [[co / total, elem]]})




    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    def simplified(self, sentence):
        res = []
        for word in list(sentence):
            if (word not in self.CollectionRevPOS):
                res.append('x')
            else:
                tem = []
                for mod in self.CollectionBayesProbability[word]:
                    tem.append(mod[0])
                res.append(self.CollectionBayesProbability[word][tem.index(max(tem))][1])
        return res

    def hmm_viterbi(self, sentence):
        cur_dist  = {'adj': [], 'adv': [], 'adp': [], 'conj': [], 'det': [], 'noun': [], 'num': [], 'pron': [], 'prt': [], 'verb': [], 'x': [], '.': []}
        partsOfSpeech = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        res , previous_pos = [] ,''
        dictonaryOfPOS = {'adj': None, 'adv': None, 'adp': None, 'conj': None, 'det': None, 'noun': None, 'num': None, 'pron': None, 'prt': None, 'verb': None, 'x': None, '.': None}
        word = list(sentence)[0]
        p=0.0
        for meta in partsOfSpeech:
            if (word, meta) not in self.CollectionEmissionProbability:
                cur_dist[meta].append(("noun", 0.000000000001))
            else:
                cur_dist[meta].append(("noun", self.CollectionEmissionProbability[(word, meta)] * self.initProbability[meta]))
        res.append(previous_pos)

        for word in list(sentence)[1:]:
            tempDictonaryOfPOS = dictonaryOfPOS
            for meta in partsOfSpeech:
                max_p = 0.0
                for previous_pos, prev_prob in cur_dist.items():
                    if (word, meta) in self.CollectionEmissionProbability:
                        tem = prev_prob[-1][1] * self.CollectionEmissionProbability[(word, meta)] * self.CollectionTranProbability[(previous_pos, meta)]
                        if (max_p < tem):
                            max_p, new_pos = tem, previous_pos
                if max_p == 0.0:
                    for previous_pos, prev_prob in cur_dist.items():
                        tem = prev_prob[-1][1] * self.CollectionTranProbability[(previous_pos, meta)] * 0.00000005
                        if (max_p < tem):
                            max_p, new_pos = tem, previous_pos
                tempDictonaryOfPOS[meta] = (new_pos, max_p)
            for i, j in tempDictonaryOfPOS.items():
                cur_dist[i].append(j)
        max_p = 0.0
        t_pos = temp = ""
        for meta, word in cur_dist.items():
            if (max_p < word[-1][1]):
                temp = meta
                max_p, t_pos = word[-1][1], word[-1][0]
        max_p = 0.0
        res = []
        if not temp:
            temp = 'noun'
        res.append(temp)
        if not t_pos:
            max_p = 0.0
            for meta in partsOfSpeech:
                if max_p < self.CollectionTranProbability[(res[-1], meta)]:
                    max_p, t_pos = self.CollectionTranProbability[(res[-1], meta)], meta
        res.append(t_pos)
        for i in range(len(sentence) - 2, 0, -1):
            if t_pos:
                t_pos = cur_dist[t_pos][i][0] 
            else:
                max_p = 0.0
                for meta in partsOfSpeech:
                    if max_p < self.CollectionTranProbability[res[-1]][meta]:
                        max_p, t_pos = self.CollectionTranProbability[res[-1]][meta], partsOfSpeech
            res.append(t_pos)
        res.reverse()
        return res[:len(sentence)]




    def complex_mcmc(self, sentence):
        result = ["noun"] * len(sentence)
        pos = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        biasProbability = []
        for word in list(sentence):
            temp = []
            for mod in pos:
                if (word, mod) not in self.CollectionEmissionProbability:
                    temp.append(2.0)
                else:
                    temp.append(self.CollectionEmissionProbability[(word, mod)])
            min_temp , i = min(temp) ,0
            while (i < len(temp)):
                if temp[i] == 2.0:
                    temp[i] = min_temp * 0.000000000001
                i+=1  
            total , i= sum(temp) ,0
            while (i < len(temp)):
                temp[i] /= total
                i+=1 
            x , i = 0.0 , 0
            while i < len(temp):
                x += temp[i]
                temp[i] = x
                i+=1
            biasProbability.append(temp)
        result , l = [] , 0
        for l in range(len(sentence)):
            result.append([])
            for num in range(20000,0,-1):
                for i in range(12):
                    if (random.random() <= biasProbability[l][i]):
                        result[l].append(pos[i])
                        break
        sol = []
        for mod in result:
            sol.append(mod[-1])
        return sol


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

