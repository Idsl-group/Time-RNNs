from settings import *
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class Markov(torch.nn.Module):
    def __init__(self, transition_matrix, device):
        super(Markov, self).__init__()
        self.transition_matrix = transition_matrix
        self.linear_tm = nn.Linear(len(transition_matrix), len(transition_matrix[0]), bias=False).to(device)
        self.linear_tm.weight.data = torch.nn.Parameter(torch.transpose(torch.from_numpy(transition_matrix), 0, 1)).to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.linear_tm(x).to(device)
        return x

class Markov_Chain(nn.Module):
    def __init__(self, transition_matrix, device):
        super(Markov_Chain, self).__init__()
        self.transition_matrix = transition_matrix
        self.markov = Markov(self.transition_matrix, device)

    def forward(self, x):
        outputs = []
        for i in range(0, INPUT_LENGTH):
            state = x[i]
        x = state
        for i in range(0, OUTPUT_LENGTH):
            x = self.markov(x).squeeze(0)
            outputs += [x]
        outputs = torch.stack(outputs, 0).to(device)
        return outputs

def generate_table_of_frequencies(filenames):
    single_events = []
    for i in range(0, len(filenames)):
      data = pd.read_csv(filenames[i])
      single_events.append(data['class'] + data['event'])

    # Event encoding
    le = LabelEncoder()
    le.fit(single_events[0])
    single_events = pd.concat(single_events)
    single_events = le.transform(single_events).reshape(-1)
    uniques = pd.DataFrame(single_events).value_counts()
    unique_titles = uniques.index.values.reshape(-1,1)
    transition_matrix = np.zeros((len(unique_titles), len(unique_titles)))
    print(len(single_events))
    for i in range(0, len(single_events)-1):
      if i%1000 == 0:
        print(f'\r{round(i/len(single_events), 4)}', end = "")
      ev1 = single_events[i]
      ev2 = single_events[i+1]
      transition_matrix[ev1][ev2] += 1
    for i in range(0, len(transition_matrix)):
      transition_matrix[i] /= transition_matrix[i].sum()
    return uniques, unique_titles, transition_matrix