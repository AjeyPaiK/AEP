from scipy.stats import bernoulli
from collections import Counter
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import log


def gen_bernoulli(X, n, p):
    bernoulli_sequences = []
    for i in range(0, X):
        sequence = bernoulli.rvs(size=n, p=p)
        sequence = tuple(sequence)
        bernoulli_sequences.append(sequence)
    # print("The sequences are:\n")
    # print(*bernoulli_sequences, sep="\n")
    return bernoulli_sequences



def entropy(p):
    H = p*(log(1/p)/log(2))+(1-p)*(log(1/(1-p))/log(2))
    return H


def get_prob(sequences, n, X, epsilon, H):
    frequency_count = Counter(sequences)
    labels, values = zip(*frequency_count.items())
    values = list(values)
    indices = np.arange(0, len(labels))
    print(indices)
    width = 0.8
    probabilities = [a/X for a in values]
    typical_prob = [0] * len(labels)
    typical_set = []
    for i in range(0, len(labels)):
      if (probabilities[i] >= pow(2, -n*(H + epsilon)) and probabilities[i] <= pow(2, -n*(H - epsilon))):
        typical_set.append(labels[i]) 
        typical_prob[i] = probabilities[i]
    print("Total number of typical sequences are:", len(typical_set))
    #print("The sum of probabilities of all sequences is: %f" % sum(probabilities))
    #print("\n")
    # print("The sequences that belong the the typical set are:\n")
    # print(*typical_set, sep  ="\n")
    plt.bar(indices, typical_prob, align='center', width=width)
    plt.xlabel("Sequence number")
    plt.ylabel("Probability of occurence")
    # for i, v in enumerate(probabilities):
    # plt.text(i-0.5, v, str(v), color='blue')
    plt.show()
    plt.savefig("baba")
    return probabilities, H


if __name__ == '__main__':
    try:
        n = int(sys.argv[1])  # length of sequence
        X = int(sys.argv[2])  # No.of Sequences
        p = float(sys.argv[3])  # Bernoulli parameter p
    except Exception as e:
        n = 8
        X = 50000
        p = 0.25
    finally:
        print("\n")
        print("Length of sequence is:", n)
        print("No.of Sequences is:", X)
        print("Bernoulli parameter is:", p)
        print("\n")
H = entropy(p)
print("The entropy H(X) is: %f bits" % H)
sequences = gen_bernoulli(X, n, p)
print("The size of the sample space is: 2 ^", n)
epsilon = 0.1
probabilities, H = get_prob(sequences, n, X, epsilon, H)
