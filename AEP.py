from scipy.stats import bernoulli
from collections import Counter
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import log
from decimal import Decimal


def gen_bernoulli(X, n, p):
    bernoulli_sequences = []
    for i in range(0, X):
        sequence = bernoulli.rvs(size=n, p=p)
        sequence = tuple(sequence)
        bernoulli_sequences.append(sequence)
    # print("The sequences are:\n")
    # print(*bernoulli_sequences, sep="\n")
    return bernoulli_sequences


def multiply(numbers):
    product = 1
    for x in numbers:
        product *= x
    return product


def entropy(probabilities):
    H = 0
    for i in range(0, len(probabilities)):
        H += probabilities[i]*(log(1/probabilities[i])/log(2))
    return H


def get_prob(sequences, X):
    frequency_count = Counter(sequences)
    labels, values = zip(*frequency_count.items())
    values = list(values)
    indices = np.arange(0, len(labels))
    width = 0.5
    probabilities = [a/X for a in values]
    H = entropy(probabilities)
    joint_prob = multiply([pow(a, b) for a, b in zip(probabilities, values)])
    print("The entropy of all sequences H(X) is: %f bits" % H)
    print("The joint probability of all the sequences that have been sampled is:%2E" % joint_prob)
    print("\n")
    print("Total number of typical sequences are:", len(labels))
    print("The sum of probabilities of all sequences is: %f" % sum(probabilities))
    print("\n")
    # print("The sequences that belong the the typical set are:\n")
    # print(*labels, sep="\n")
    plt.figure(figsize=(len(labels)*5, 7))
    plt.bar(indices, probabilities, align='center', width=width)
    plt.xlabel("Sequence number")
    plt.ylabel("Probability of occurence")
    plt.xticks(indices, indices)
    # for i, v in enumerate(probabilities):
    # plt.text(i-0.5, v, str(v), color='blue')
    plt.show()
    return probabilities, H, joint_prob


if __name__ == '__main__':
    try:
        n = int(sys.argv[1])  # length of sequence
        X = int(sys.argv[2])  # No.of Sequences
        p = float(sys.argv[3])  # Bernoulli parameter p
    except Exception as e:
        n = 5
        X = 10
        p = 0.5
    finally:
        print("\n")
        print("Length of sequence is:", n)
        print("No.of Sequences is:", X)
        print("Bernoulli parameter is:", p)
        print("\n")
sequences = gen_bernoulli(X, n, p)
print("The size of the sample space is: 2 ^", X)
probabilities, H, joint_prob = get_prob(sequences, X)
epsilon = 0.1
if joint_prob >= pow(2, -X*(H + epsilon)) and joint_prob <= pow(2, -X*(H - epsilon)):
    print("Since %2E <= %2E <= %2E, the asymptotic equipartition property is verified!" %
          (pow(2, -X*(H + epsilon)), joint_prob, pow(2, -X*(H - epsilon))))
    print("\n")
else:
    print("Error!")
    print("\n")
