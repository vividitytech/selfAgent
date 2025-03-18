import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm  # For progress bars
import logging
import torch.nn.functional as F

from behaviorclassifier import BehaviorClassifier, TextClassificationDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1. Prepare your data (example using a CSV file)
DATA_PATH = "behavior_data.csv"  # Replace with your actual data path

try:
    df = pd.read_csv(DATA_PATH)
    # Check if the necessary columns exist
    if not all(col in df.columns for col in ['text', 'label']):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()



# 2. Choose a Pre-trained Model
MODEL_NAME = "bert-base-uncased"  # Or any other suitable model


# 5. Load Tokenizer and Model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    NUM_CLASSES = len(df['label'].unique())  #Dynamically determine the number of classes
    model = BehaviorClassifier(MODEL_NAME, NUM_CLASSES+1)
    print(f"Model '{MODEL_NAME}' loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 6. Prepare Data
MAX_LENGTH = 128 #Adjust as needed
texts = df['text'].tolist()
labels = df['label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 7. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use CUDA if available
model.to(device) #Move model to device

optimizer = AdamW(model.parameters(), lr=2e-5) #Optimizer

# Cross Entropy Loss (for classification)
criterion = torch.nn.CrossEntropyLoss()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train() #Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, desc="Training", unit="batch") as progress_bar:  # Use tqdm for progress bar
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad() #Zero the gradients

            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels) #Calculate the loss

            loss.backward() #Backpropagate the loss
            optimizer.step()  #Update the weights

            total_loss += loss.item()

            # Calculate metrics
            _, predicted = torch.max(outputs, dim=1) #Get the predicted class
            correct_predictions += torch.sum(predicted == labels).item()
            total_samples += labels.size(0) #Count total number of samples

            progress_bar.set_postfix({"loss": loss.item()}) #Update progress bar



    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval() #Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): #Disable gradient calculation
        with tqdm(dataloader, desc="Evaluating", unit="batch") as progress_bar:
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                # Calculate metrics
                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(predicted == labels).item()
                total_samples += labels.size(0)
                progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# Number of training epochs
NUM_EPOCHS = 3

try:
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device) # Evaluate
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print("Training complete.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()


# 8. Save the Trained Model (optional)
try:
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model") #Save tokenizer
    print("Model saved to ./trained_model")
except Exception as e:
    print(f"Error saving model: {e}")


# 9. Example Inference/Prediction (after training)
def predict(text):
    """
    Predicts the label for a given text using the trained model.
    """

    #Load trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./trained_model")
    model = BehaviorClassifier.from_pretrained("./trained_model",num_classes = NUM_CLASSES+1) #load model for prediction

    model.to(device)  # Move the model to the correct device

    model.eval() #Set model to evaluation mode
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad(): #Disable gradient calculation for inference
        outputs = model(input_ids, attention_mask)
        _, predicted_class = torch.max(outputs, dim=1) #Get the predicted class

    return F.softmax(outputs).cpu().numpy(), predicted_class.item() #Return the class


try:
    example_text = "This is a positive example."
    probability, predicted_class = predict(example_text)
    print(f"Text: {example_text}")
    print(f"Predicted Class: {predicted_class}")
except Exception as e:
    print(f"Error during prediction: {e}")