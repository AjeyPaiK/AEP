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

def get_prob(sequence):
  return pow(p,sequence.count(1))*pow((1-p),sequence.count(0))

def entropy(p):
    H = p*(log(1/p)/log(2))+(1-p)*(log(1/(1-p))/log(2))
    return H

def get_list(sequences, n, X, epsilon, H):
  frequency_count = Counter(sequences)
  labels, values = zip(*frequency_count.items())
  indices = np.arange(0, len(labels))
  width = 0.8
  probabilities = [a/X for a in values]
  plt.bar(indices, probabilities, align='center', width=width)
  plt.xlabel("Sequence number")
  plt.ylabel("Probability of occurence")
  plt.legend(['All Sequences'])
  plt.show()
  plt.savefig("histogram")
  # for i, v in enumerate(probabilities):
  # plt.text(i-0.5, v, str(v), color='blue')
  plt.show()
  return labels, indices
  
def get_typ_seq(indices, frequent, joint_PMF, epsilon, n, H):
  typical_prob = [0] * len(frequent)
  typical_set = []
  for i in range(0, len(frequent)):
    if (joint_PMF[i] >= pow(2, -n*(H + epsilon)) and joint_PMF[i] <= pow(2, -n*(H - epsilon))):
      typical_set.append(frequent[i])
      typical_prob[i] = joint_PMF[i]
  plt.bar(indices, typical_prob, align='center', width=0.8)
  plt.xlabel("Typical sequence number")
  plt.ylabel("Probability of occurence")
  plt.legend(['All Sequences','Typical Sequences'])
  plt.show()
  plt.savefig("Typical set")
  print("Total number of typical sequences are:", len(typical_set))
  print("The sequences that belong the the typical set are:\n")
  print(*typical_set, sep  ="\n")
  return


if __name__ == '__main__':
  possible_sequences = []
  try:
      n = int(sys.argv[1])  # length of sequence
      X = int(sys.argv[2])  # No.of Sequences
      p = float(sys.argv[3])  # Bernoulli parameter p
  except Exception as e:
      n = 10
      X = 50000
      p = 0.3
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
frequent, indices = get_list(sequences, n, X, epsilon, H)
joint_PMF = [get_prob(sequence) for sequence in frequent]
get_typ_seq(indices, frequent, joint_PMF, epsilon, n, H)
