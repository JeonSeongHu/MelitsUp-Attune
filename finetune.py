import torch
import pandas as pd
from transformers import AutoTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Load data
train_df = pd.read_csv("dataset.csv", index_col=0)
train_texts = train_df["lines"].to_list()
train_labels = train_df["label"].to_list()
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2,
                                                                    random_state=0)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


# Define Dataset
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Define DataLoader
train_dataset = MyDataset(train_encodings, train_labels)
val_dataset = MyDataset(val_encodings, val_labels)

# Define model
model = RobertaForSequenceClassification.from_pretrained("BM-K/KoSimCSE-roberta-multitask", num_labels=9)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define optimizer and loss function
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

num_epochs = 5
batch_size =
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Set up learning rate scheduler
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# Define EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


from tqdm import tqdm

callback_earlystop = EarlyStopping(patience=2)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} -- Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        _, predicted_labels = torch.max(logits, 1)
        train_acc += torch.sum(predicted_labels == labels).item()

    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} -- Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            val_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            val_acc += torch.sum(predicted_labels == labels).item()
        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{num_epochs} -- Train Loss: {train_loss:.3f} -- Train Acc: {train_acc:.3f} -- Val Loss: {val_loss:.3f} -- Val Acc: {val_acc:.3f}")
    callback_earlystop(val_loss)
    if callback_earlystop.early_stop:
        print("Early stopping")
        break

