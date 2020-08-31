from prep_data import *
import pandas as pd
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

questions, responses = prepare_training_data()
questions_df = pd.DataFrame(questions)
questions_df = clean_text(questions_df, lemmatize=True)

embeddings_dict, embeddings = get_embeddings()

dataset = ChatbotDataset(questions_df, embeddings_dict.keys(), max_len=64)
loader = DataLoader(dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChatBot(embeddings, len(set(questions_df["tags"].values.tolist()))).to(device)
print(model)

optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=2e-4)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loop = tqdm(loader, total=len(loader), leave=False)
    train_loss = 0.0
    for batch in loop:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    
    if epoch % 250 == 0:
        torch.save(model.state_dict(), f"./models/chatbot-{epoch}.pt")