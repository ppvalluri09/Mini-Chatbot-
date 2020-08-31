import numpy as np
import json
from nltk.tokenize.regexp import regexp_tokenize
from nltk.stem import WordNetLemmatizer
import torch

def get_embeddings():
    embeddings_dict = {}
    vectors = []
    with open("../../Word Embeddings/Pretrained/glove.6B.200d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
            vectors.append(vector)
    return embeddings_dict, np.asarray(vectors)

def get_intents():
    with open("./dataset/intents.json", "r") as f:
        intents = json.load(f)
    return intents["intents"]

def prepare_training_data():
    intents = get_intents()
    responses = {}
    questions = {"patterns": [], "tags": []}

    for intent in intents:
        responses[intent["tag"]] = intent["responses"]
        for question in intent["patterns"]:
            questions["patterns"].append(question)
            questions["tags"].append(intent["tag"])
    return questions, responses

def clean_text(df, lemmatize=False):
    df["patterns"] = df["patterns"].str.lower()
    if lemmatize:
        df["patterns"] = df["patterns"].apply(lambda x: " ".join(WordNetLemmatizer().lemmatize(x) for x in x.split()))
    df["patterns"] = df["patterns"].str.replace("*", "")
    df["patterns"] = df["patterns"].str.replace(r"[^\w\s]", "")
    return df

class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, df, questions, max_len=None):
        df = df.sample(frac=1).reset_index(drop=True)
        self.x = df["patterns"].values
        self.y = df["tags"].values
        self.y_dict = {v: k for k, v in dict(enumerate(sorted(set(self.y)))).items()}
        self.questions = list(questions)
        if max_len is None:
            self.max_len = max([len(val) for val in x])
        else:
            self.max_len = max_len
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = regexp_tokenize(self.x[idx], pattern="\s+", gaps=True)
        y = self.y_dict[self.y[idx]]
        x = [self.questions.index(word) for word in x]
        if len(x) > self.max_len:
            x = x[:self.max_len]
        else:
            x = x + [self.questions.index("padding")] * (self.max_len - len(x))

        return torch.tensor(x), torch.tensor(y)