import random
import json
import torch
import torch.nn as nn
from nltk_utils import bag_of_words, tokenize
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = torch.load("cse_chatbot.pth", weights_only=True)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]
with open('cse_chatbot_dataset.json', 'r') as f:
    intents = json.load(f)

def get_response(sentence):
    tokenized = tokenize(sentence)
    bag = bag_of_words(tokenized, all_words)
    bag = bag.reshape(1, bag.shape[0])
    bag = torch.from_numpy(bag)

    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "Sorry, I don't know about that."
