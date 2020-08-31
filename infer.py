from prep_data import *
from models import *
from nltk.tokenize.regexp import regexp_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import random
import pandas as pd

embeddings_dict, embeddings = get_embeddings()
model = ChatBot(embeddings, 7)

model.load_state_dict(torch.load("./models/chatbot-1000.pt", map_location=torch.device("cpu")))

def clean_query(text, lemmatize=False):
    text = text.lower()
    if lemmatize:
        text = " ".join([WordNetLemmatizer().lemmatize(x) for x in text.split()])
    text = text.replace("*", "")
    text = text.replace(r"[^\w\s]", "")
    text = regexp_tokenize(text, pattern="\s+", gaps=True)
    return text

questions, responses = prepare_training_data()
df = pd.DataFrame(questions)
y_dict = dict(enumerate(sorted(set(df["tags"].values.tolist()))))

query = ""
while query != "quit":
    query = str(input(">>> "))
    query = clean_query(query, True)
    query = [list(embeddings_dict.keys()).index(word) for word in query]
    if len(query) > 64:
        query = query[:64]
    else:
        query = query + [list(embeddings_dict.keys()).index("padding")] * (64 - len(query))
    query = torch.tensor(query)
    yhat = model(query.view(1, -1))
    yhat = yhat.argmax(1)
    answer = random.choice(responses[y_dict[yhat.item()]])
    print("Eliza>>>", answer)