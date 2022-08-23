# Assignment 3: EAI Assignment
## Part1:

### Posterior Probability:
We normalize the probabilities for POS. For Simplified, we calculate log of sum of emission prob for all words.
For HMM, we calculate log of sum of emission and transition prob 
For Complex, like HMM but we additionally include gibbs sampling normalized POS prob.

### Training:
Created dictionaries storing transition, emission and initial probabilities.
Calculated Bayes probability for all words.

Formulae used for calculations:

Transition probability: P(s_(i+1)| s_i)

Emission Probability: P(w_i|s_i)

### Algorithms:
1. Simplified:
For each word We have created dictionary to store all POS and normalized. We have used most probable POS if testing label exists else ‘x’ as default.

2. HMM:
We use emission, transition and initial probability and we consider pos with max probability by iterating 12 POS
Formulae: (s∗1, . . . , s∗N ) = arg maxs1,...,sNP (Si = si|W ).

3. MCMC Complex:
We use gibbs sampling by randomly generating POS from existing POS emission probability. We do this random iteration for 20000 times and return the last POS.

##part 2:


Goal: To find out the boundaries at both the air-ice and the ice-bedrock .

The edge strength gives the values for the relative intensity of the particular pixels which is helpful in finding out the boundaries. 

Simple bayes net method:


Each pixel of the given image has both the row index value and column index value. By considering that in mind we have found the pizel in a column with maximum edge strength and have written its corresponding row value. By using this we have plotted the boundaries. 

for the Ice rock as per the assumptions given in the question we have added a value 10 to the air_ice_simple and used the same previous method in order to find out the boundary at the Ice rock layer.   


Viterbi method :

Even thought we didn't write the program for the viterbi method. we are writing the methodology to find boundaries.Please look through 

The viterbi method finds the boundary line in correspondance with the previous line such that the boundary mark at the previous pixel is closer to the next one adding to the edge strength. 
we multiply the maximum value of the previous state along eith the transition value to find the maximum emission probability and we use the backtracking method
to get the boundary.

