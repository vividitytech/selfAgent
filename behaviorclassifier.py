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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_LENGTH = 256 #Adjust as needed
# 3. Define the Custom Model
class BehaviorClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, num_classes) #Specify num_classes

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output #Use pooler output for classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_directory):
        """Saves the model to a directory, including the pretrained model and custom classifier."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the pretrained model
        self.pretrained_model.save_pretrained(os.path.join(save_directory, 'pretrained'))

        # Save the custom classifier's state dictionary
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, 'classifier.pth'))

    @classmethod
    def from_pretrained(cls, pretrained_directory, num_classes):
        """Loads the model from a directory."""
        # Load the pretrained model
        pretrained_model = AutoModel.from_pretrained(os.path.join(pretrained_directory, 'pretrained'))

        # Create the CustomClassifier instance
        model = cls(pretrained_model.name_or_path, num_classes)
        model.pretrained_model = pretrained_model  # Assign the loaded pretrained model

        # Load the custom classifier's state dictionary
        classifier_state_dict = torch.load(os.path.join(pretrained_directory, 'classifier.pth'))
        model.classifier.load_state_dict(classifier_state_dict)

        return model
    

    def predict(self, text, tokenizer, device):
        """
        Predicts the label for a given text using the trained model.
        """
        #model.to(device)  # Move the model to the correct device

        # model.eval() #Set model to evaluation mode
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
            outputs = self.forward(input_ids, attention_mask)
            _, predicted_class = torch.max(outputs, dim=1) #Get the predicted class

        return F.softmax(outputs, dim=-1).cpu().numpy(), predicted_class.item() #Return the class

# 4. Create a Custom Dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Use torch.long for cross-entropy loss
        }


