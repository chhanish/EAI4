#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Harsh Dankhara, Sandeepkumar Gaddam, Hanish Chidipothu
# (based on skeleton code by D. Crandall, Oct 2020)
#
#We are using two different methods 
#1]simplified Bayes Net 
#2]Viterbi algorithm on HMM 
#to recognize the text in image to see which performs better than the other. 
#The second character recognition in the test image depends on the first character recognized and also the observed characters and 
#therefore we form a HMM of characters where test image are the observed values. 
#we are using bc.train as a training data file, which we clean by removing the foreign characters and 
#also replacing space followed by period with period so that we can improve the transition probability of the ending letter, 
#after which we calculate the initial probabilities with and without log, as viterbi requires log, and 
#also we calculate the transition probabilities by the first letter present. 
#We then calculate emission probabilities by comparing the observed character's pixels data with the 
#pixels information of each character in the training image and if they match we multiply by 0.8 else we multiply by 0.2. 
#After we compute all the probabilities we implement the above two methods to recognize the text in an image which we pass as argument as test image file. 
#For, Simple method we just see which training letter has the highest probability to be the text in test image.

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

def load_letters(fname):
        im = Image.open(fname)
        px = im.load()
        (x_size, y_size) = im.size
        print(im.size)
        print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
        result = []
        for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
            result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
        return result
        
def load_training_letters(fname):
        TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        letter_images = load_letters(fname)
        return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def ras(s):
    res = ""
    for i in s:
        for j in i:
            res += str(j)
    return res

def remissionP(a, b):
    pp = 0.8
    p = 1
    for i in range(len(a)):
        if a[i] != b[i]:
            p = p * (1 - pp)
        else:
            p = p * pp
    return p

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

tel = [i for i in test_letters[0]]

trl = [[0 for i in range(2)] for j in range((len(train_letters)))]

c = 0
for i in train_letters:
    trl[c][0] = i
    trl[c][1] = ras(train_letters[i])
    c += 1

tel = [i for i in test_letters[15]]
op = ""

for tel in test_letters:
    minl = trl[0][0]
    minv = remissionP(ras(tel), ras(trl[0][1]))
    for p in range(len(trl)):
        up = remissionP(ras(tel), ras(trl[p][1]))
        if minv < up:
            minv = up
            minl = trl[p][0]

    op += minl
print("Simple: " + op)

TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
c = 1

def rd(fname):
    ex = []
    f = open(fname, 'r')
    for l in f:
        temp = tuple([i for i in l.split()])
        ex += [(temp[0::2]), ]
    return ex

def rcd():
    trd = rd(train_txt_fname)
    cd = ""

    for l1 in trd:
        sp = ""
        for l in l1:
            sp += " " + ''.join(c for c in l if c in TRAIN_LETTERS)
        sp = sp.replace(" .", ".")
        cd += sp + "\n"
    return cd.strip()

ifd = dict()
ipnld = dict()
ipd = dict()
tcd = dict()
tc = 0
con = rcd().splitlines()

for l in con:
    for i in l:
        currv = tcd.get(i, 0)
        tcd[i] = currv + 1
        tc += 1

for l in con:
        if len(list(l))>1:
            if list(l)[0] == " ":
                i=list(l)[1]
            else:
                i=list(l)[0]
            currv = ifd.get(i, 0)
            ifd[i] = currv + 1

ti=0
for k in ifd:
    ti += ifd[k]

for k in ifd:
    ipnld[k] = (float(ifd[k]) / ti)
    ipd[k] = -math.log(float(ifd[k]) / ti)

tf = dict()
for l in con:
    for ch_i in range(len(l) - 1):
        currv = tf.get(l[ch_i] + l[ch_i + 1], 0)
        tf[l[ch_i] + l[ch_i + 1]] = currv + 1

transitionP = dict()
transitionnlP = dict()

for k in tf:
    minV = tc
    v = ifd.get(k[0], minV)
    transitionP[k] = -math.log(float(tf[k]) / v)
    transitionnlP[k] = (float(tf[k]) / tcd[k[0]])

lv = -math.log(1.0 / tc)
lnlv = (1.0 / tc)
c = 0
tpd = dict()
tek = dict()

def pcemission(tlp):
    for i in range(len(tlp)):
        temissiontel = dict()
        for j in TRAIN_LETTERS:
            temissiontel[j] = remissionP(ras(train_letters[j]), ras(test_letters[i]))
        tek[i] = temissiontel


pcemission(test_letters)

c = 0
tpd = dict()

def v_hmm():
    fd = dict()
    sd = dict()

    op = ""
    c = 0
    tcv = 0
    fdn = False
    mtransitionD = dict()
    for tel in test_letters:
        fdn = not fdn
        if (fdn):
            fd = dict()
        else:
            sd = dict()
        for l in TRAIN_LETTERS:
            emissionP = -math.log(remissionP(ras(train_letters[l]), ras(tel)))
            if c == 0:
                fd[l] = (ipd.get(l, -math.log(1.0 / tc)) + emissionP)
            else:
                tvitD = dict()
                for i in TRAIN_LETTERS:
                    tpv = transitionP.get(i + l, -math.log(1.0 / tc))
                    tDk = str(c) + str(i) + str(l)
                    if fdn:
                        tvitD[tDk] = (tpv + emissionP + sd.get(str(i), -math.log(1.0 / tc)))
                    else:
                        tvitD[tDk] = (tpv + emissionP + fd.get(str(i), -math.log(1.0 / tc)))

                minDsk = min(tvitD, key=tvitD.get)
                mtransitionD[str(c) + l] = minDsk[-2]
                if fdn:
                    fd[l] = tvitD[minDsk]
                else:
                    sd[l] = tvitD[minDsk]
        if fdn:
            m = min(fd, key=fd.get)
        else:
            m = min(sd, key=sd.get)

        c += 1
        op += str(m)
        tcv += 1
    tempc = c - 1

    fop = m
    while tempc > 0:
        m = mtransitionD[str(tempc) + str(m)]
        tempc -= 1
        fop = m + fop

    print("   HMM: " + op)

v_hmm()
