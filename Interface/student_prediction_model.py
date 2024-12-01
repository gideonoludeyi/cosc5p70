import csv
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StudentPredictionModel:
    def __init__(self, model_path, input_mean=None, input_std=None, logfile=None):
        """
        Initialize the model and load its parameters from the specified path.
        """
        self.model = self._build_model()
        self._load_model(model_path)
        self.model.eval()  # Set model to evaluation mode
        self.label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        self.input_mean = input_mean  # Placeholder for input normalization mean
        self.input_std = input_std  # Placeholder for input normalization std
        self.logfile = logfile

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
            nn.Linear(64, 3),
        )

    def _load_model(self, model_path):
        """
        Load the saved model weights from the file.
        """
        self.model.load_state_dict(torch.load(model_path))

    def _preprocess_input(self, input_features):
        """
        Normalize the input features.
        """
        if self.input_mean is not None and self.input_std is not None:
            return (input_features - self.input_mean) / self.input_std
        else:
            return input_features

    def _create_logfile_if_not_exists(self):
        """
        Create a csv file to record confidence probabilities for each prediction
        """
        if self.logfile is None:
            return
        logfile = pathlib.Path(self.logfile)
        if not logfile.exists():
            logfile.touch()  # Create the log file
            with open(logfile, "w") as f:
                writer = csv.writer(f)
                inputfields = [f"feature_{i}" for i in range(34)]
                outputfields = [f"label_logproba_{i}" for i in range(3)]
                writer.writerow(inputfields + outputfields)

    def _record_confidence(self, input_features, probabilities):
        """
        Write the confidence (probabilities) of the model for each label onto a csv file.
        The probabilities are recorded as `y = log(1 + x_l)` where `x_l` is the model's confidence
        that the input features `x` correspond to a label `l`.

        Args:
            input_features (list or numpy array): Feature vector of size 34.
            probabilities (list or numpy array): Confidence scores for ('Dropout', 'Enrolled', 'Graduated') respectively.
        """
        if self.logfile is not None:
            logprobas = np.log1p(probabilities)
            record = np.concatenate([input_features, logprobas[0]])

            self._create_logfile_if_not_exists()
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(record.tolist())

    def predict(self, input_features):
        """
        Make a prediction using the model.
        Args:
            input_features (list or numpy array): Feature vector of size 34.
        Returns:
            tuple: Predicted label ('Dropout', 'Enrolled', or 'Graduate') and confidence score.
        """
        with torch.no_grad():
            processed_input_features = self._preprocess_input(input_features)
            input_tensor = torch.tensor(
                processed_input_features, dtype=torch.float32
            ).unsqueeze(0)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)  # Calculate probabilities

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_score = torch.max(
                probabilities
            ).item()  # Get the highest probability

            predicted_label = self.label_map[predicted_class]
            self._record_confidence(input_features, probabilities)
            return predicted_label, confidence_score
