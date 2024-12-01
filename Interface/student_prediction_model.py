import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StudentPredictionModel:
    def __init__(self, model_path):
        """
        Initialize the model and load its parameters from the specified path.
        """
        self.model = self._build_model()
        self._load_model(model_path)
        self.model.eval()  # Set model to evaluation mode
        self.label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        self.input_mean = None  # Placeholder for input normalization mean
        self.input_std = None   # Placeholder for input normalization std

    def _build_model(self):
        """
        Define the model architecture. This should match the training architecture.
        """
        return nn.Sequential(
            nn.Linear(34, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def _load_model(self, model_path):
        """
        Load the saved model weights from the file.
        """
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, input_features):
        """
        Make a prediction using the model.
        Args:
            input_features (list or numpy array): Feature vector of size 34.
        Returns:
            str: Predicted label ('Dropout', 'Enrolled', or 'Graduate').
        """
        with torch.no_grad():
            # Convert input to a tensor and normalize using L2 normalization
            input_tensor = torch.tensor(input_features, dtype=torch.float32)
            input_tensor = F.normalize(input_tensor, p=2, dim=0).unsqueeze(0)  # Normalize and add batch dimension
            
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)  # Calculate probabilities
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_score = torch.max(probabilities).item()  # Get the highest probability
            
            predicted_label = self.label_map[predicted_class]
            return predicted_label, confidence_score
