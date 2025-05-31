import torch
import torch.nn as nn # Added for MLPModel
import numpy as np # For predict_proba slicing
import pathlib
import os
import sys # Added for the workaround
# We might need sklearn.preprocessing.StandardScaler and catboost.CatBoostClassifier
# if type hints are needed or for isinstance checks, but not strictly for loading/operating
# if the objects are pickled correctly.

# Define MLPModel class here so torch.load can find it
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        # Calculate layer sizes for a 3-layer network
        # (all-mpnet-base-v2 is typically 768 dimensions)
        # Assuming input_dim is 768
        layer1_size = int(input_dim * 1.5)
        layer2_size = int(layer1_size * 0.66)
        layer3_size = int(layer2_size * 0.5)
        
        # Define network layers (3 layers)
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, layer1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(layer1_size),
            
            # Second layer
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(layer2_size),
            
            # Third layer
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(layer3_size),
            
            # Output layer
            nn.Linear(layer3_size, 2)
        )
        
        # Store layer sizes for printing/info
        self.layers_info = {
            'input': input_dim,
            'layer1': layer1_size,
            'layer2': layer2_size,
            'layer3': layer3_size,
            'output': 2
        }
        
    def forward(self, x):
        return self.network(x)

# Global variables for the model components
MODEL_DATA = None
V_SCALER = None # This will be our StandardScaler
C_CLASSIFIER = None # This will be our CatBoostClassifier (or other)

# Get the directory where this script (api/model.py) is located
current_script_directory = pathlib.Path(__file__).resolve().parent

# Construct the absolute path to the model file
# model_v1.pkl is expected to be in a 'models' subdirectory
# The filename 'model_v1.pkl' might be misleading now, as it's a torch.save file.
# Consider renaming it if it causes confusion, e.g., 'highlight_model.pth'
model_file_name = "model_v1.pkl" # Or the actual name of your torch.save file
model_file_path = current_script_directory / "models" / model_file_name

# Workaround for PyTorch unpickling issue with classes defined in __main__ of training script
# The error message usually indicates the module name it's looking for (e.g., '__mp_main__')
# This code attempts to make the locally defined MLPModel available under that name.
_target_module_name_from_error = '__mp_main__' # Based on the last error message
if _target_module_name_from_error in sys.modules:
    if 'MLPModel' in globals() and isinstance(globals()['MLPModel'], type):
        setattr(sys.modules[_target_module_name_from_error], 'MLPModel', MLPModel)
        print(f"INFO: Injected MLPModel into sys.modules['{_target_module_name_from_error}'] for torch.load.")
    else:
        print(f"WARNING: MLPModel class not found in api/model.py globals or not a class. Cannot inject into '{_target_module_name_from_error}'.")
elif '__main__' in sys.modules: # Fallback to common '__main__' if specific one not found or typo
    if 'MLPModel' in globals() and isinstance(globals()['MLPModel'], type):
        setattr(sys.modules['__main__'], 'MLPModel', MLPModel)
        print(f"INFO: Injected MLPModel into sys.modules['__main__'] (fallback) for torch.load.")
    else:
        print(f"WARNING: MLPModel class not found in api/model.py globals or not a class. Cannot inject into '__main__' (fallback).")
else:
    print(f"WARNING: Neither '{_target_module_name_from_error}' nor '__main__' found in sys.modules. MLPModel injection failed.")

print(f"Current Working Directory (from model.py perspective): {os.getcwd()}")
print(f"Calculated absolute model path: {model_file_path}")
print(f"Does model file exist at calculated path? {model_file_path.exists()}")

try:
    if not model_file_path.exists():
        print(f"ERROR: Model file not found at {model_file_path}")
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    # Load the model data using torch.load, ensuring it runs on CPU if no GPU
    # Added weights_only=False as per PyTorch 2.6+ recommendation for trusted files
    # containing custom classes.
    MODEL_DATA = torch.load(model_file_path, map_location=torch.device('cpu'), weights_only=False)
    print("Model data loaded successfully with torch.load (weights_only=False).")

    if isinstance(MODEL_DATA, dict):
        # Extract the scaler and a chosen classifier
        if 'scaler' in MODEL_DATA:
            V_SCALER = MODEL_DATA['scaler']
            print("Scaler extracted successfully.")
        else:
            print("ERROR: 'scaler' not found in the loaded MODEL_DATA.")
            V_SCALER = None

        # Example: Using the CatBoost model from the first fold
        # Adjust this if you want a different model or the ensemble
        if 'fold_models' in MODEL_DATA and MODEL_DATA['fold_models']:
            first_fold = MODEL_DATA['fold_models'][0]
            if 'cat' in first_fold: # Assuming 'cat' holds the CatBoost model
                C_CLASSIFIER = first_fold['cat']
                print("CatBoost classifier from the first fold extracted successfully.")
            else:
                print("ERROR: CatBoost model ('cat') not found in the first fold of 'fold_models'.")
                C_CLASSIFIER = None
        else:
            print("ERROR: 'fold_models' not found or empty in MODEL_DATA.")
            C_CLASSIFIER = None
        
        if V_SCALER is None or C_CLASSIFIER is None:
            print("CRITICAL: One or more essential model components (scaler, classifier) failed to load/extract.")
            # Potentially raise an error or ensure score() handles this
    else:
        print(f"ERROR: Model data format is not as expected. MODEL_DATA is not a dictionary.")
        V_SCALER, C_CLASSIFIER = None, None

except FileNotFoundError: # Should be caught by model_file_path.exists() but good to have
    print(f"ERROR: Model file not found at '{model_file_path}'. Ensure it's in api/models/.")
    V_SCALER, C_CLASSIFIER = None, None
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading model '{model_file_path.name}' with torch.load: {e}")
    V_SCALER, C_CLASSIFIER = None, None

def score(embeddings_list: list) -> list[float]:
    """
    Scores sentences based on their embeddings.
    Expects a list of embedding arrays, NOT raw sentences.
    The embedding part should be done in main.py before calling this.
    """
    if V_SCALER is None or C_CLASSIFIER is None:
        print("ERROR in score(): Model (scaler or classifier) not loaded.")
        return [0.0] * len(embeddings_list)

    try:
        if not embeddings_list:
            return []
        
        embeddings_array = np.array(embeddings_list)
        if embeddings_array.ndim == 1:
             embeddings_array = embeddings_array.reshape(1, -1)
        
        scaled_embeddings = V_SCALER.transform(embeddings_array)
        # predict_proba returns probabilities for each class (e.g., [[P(class_0), P(class_1)], ...])
        # Based on user's definition: class 0 is 'important', class 1 is 'not important'.
        # We want to return P(important), which is P(class_0).
        probabilities = C_CLASSIFIER.predict_proba(scaled_embeddings)
        
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            important_scores = probabilities[:, 0].tolist() # Changed from [:, 1] to get P(class_0)
        else:
            print("Warning: Classifier predict_proba output shape is not as expected. Defaulting scores.")
            important_scores = [0.5] * len(embeddings_list) # Fallback

        return important_scores
    except Exception as e:
        print(f"Error during scoring: {e}")
        # Fallback to returning a list of default scores or re-raise
        return [0.0] * len(embeddings_list)

# Ensure V (Scaler) and C (Classifier) are defined for main.py if it expects them directly.
# For now, main.py will call model.score() which uses the global V_SCALER and C_CLASSIFIER.
# If main.py was trying to import V and C directly, that needs to be changed.
# It seems main.py imports "from . import model as scoring_model" and then calls scoring_model.score()
# OR it was trying to access scoring_model.V and scoring_model.C.
# With the current setup, scoring_model.V and scoring_model.C are not the loaded components.
# The loaded components are V_SCALER and C_CLASSIFIER.
# To avoid breaking main.py if it used scoring_model.V, we can assign them,
# but it's better if main.py just uses the score function.

V = V_SCALER # For potential backward compatibility if main.py accessed model.V
C = C_CLASSIFIER # For potential backward compatibility if main.py accessed model.C 