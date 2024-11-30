import torch
import torch.nn as nn
import numpy as np

class StudentOutcomeModel:
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
            nn.Linear(36, 256),
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

    def set_normalization_params(self, mean, std):
        """
        Set normalization parameters for inputs.
        """
        self.input_mean = np.array(mean)
        self.input_std = np.array(std)

    def predict(self, input_features):
        """
        Make a prediction using the model.
        Args:
            input_features (list or numpy array): Feature vector of size 36.
        Returns:
            str: Predicted label ('Dropout', 'Enrolled', or 'Graduate').
        """
        if len(input_features) != 36:
            raise ValueError("Input features must be of length 36.")
        
        # Normalize the input
        input_array = np.array(input_features, dtype=np.float32)
        if self.input_mean is not None and self.input_std is not None:
            input_array = (input_array - self.input_mean) / self.input_std

        # Convert to tensor and make prediction
        input_tensor = torch.tensor(input_array).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()

        return self.label_map[predicted_index]
