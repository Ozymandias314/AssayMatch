import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Union, Optional


class TorchMLPClassifier(nn.Module):
    """
    PyTorch MLP Classifier with frozen encoder for molecular property prediction.
    Encoder processes tokenized SMILES, MLP head is trained on encoded features.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_layer_sizes: Union[int, List[int]] = 100,
        activation: str = 'relu',
        max_iter: int = 1000,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        verbose: bool = False,
        random_state: Optional[int] = None,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        encoder_on_cpu: bool = True,
    ):
        """
        Parameters:
        -----------
        encoder : nn.Module
            Pretrained encoder (e.g., TrfmSeq2seq) with .encode() method.
            Will be frozen during training.
        hidden_layer_sizes : int or list of ints, default=100
            Number of neurons in each hidden layer. If int, creates one hidden layer.
        activation : str, default='relu'
            Activation function ('relu', 'tanh', 'logistic')
        max_iter : int, default=1000
            Maximum number of training iterations
        batch_size : int, default=200
            Size of minibatches for optimization
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer
        verbose : bool, default=False
            Whether to print progress messages
        random_state : int, optional
            Random seed for reproducibility
        early_stopping : bool, default=True
            Whether to use early stopping based on training loss
        n_iter_no_change : int, default=10
            Number of iterations with no improvement to wait before stopping
        tol : float, default=1e-4
            Tolerance for optimization improvement
        """
        super(TorchMLPClassifier, self).__init__()
        print("initializing classifier")
        
        # Store encoder and freeze it
        self.encoder = encoder
        self._freeze_encoder()
        
        # Store hyperparameters
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_name = activation
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.encoder_on_cpu = encoder_on_cpu
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Keep encoder on CPU to avoid numpy conversion issues in its encode method
        # We'll move encoded outputs to GPU after encoding
        if encoder_on_cpu:
            self.encoder.to('cpu')
        #print device of encoder modules
        self.encoder.eval()  # Keep encoder in eval mode
        
        # These will be initialized in fit()
        self.mlp = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.best_loss_ = float('inf')
        self.n_iter_ = 0
        
    def _freeze_encoder(self):
        """Freeze encoder parameters but keep gradients flowing"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
    
    def _build_mlp(self, input_dim: int, output_dim: int):
        """Build the MLP network architecture"""
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.mlp.to(self.device)
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation_name == 'relu':
            return nn.ReLU()
        elif self.activation_name == 'tanh':
            return nn.Tanh()
        elif self.activation_name == 'logistic':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def _encode_batch(self, xid_batch):
        """Encode a batch of tokenized SMILES using the frozen encoder"""
        # Move to CPU for encoding (encoder has numpy conversion issues on GPU)
        if self.encoder_on_cpu:
            xid_batch = xid_batch.cpu()
        
        # Transpose for encoder (expects [seq_len, batch_size])
        xid_transposed = torch.t(xid_batch)
        
        # Encode with frozen encoder (no grad needed for encoder)
        with torch.no_grad():
            encoded = self.encoder.encode(xid_transposed)
            
            # Handle case where encoder returns numpy array
            if isinstance(encoded, np.ndarray):
                encoded = torch.FloatTensor(encoded)
            
            # Move back to target device
            encoded = encoded.to(self.device)
        
        return encoded
    
    def forward(self, xid):
        """
        Forward pass: encode then classify
        
        Parameters:
        -----------
        xid : torch.Tensor of shape (batch_size, seq_len)
            Tokenized SMILES sequences
            
        Returns:
        --------
        logits : torch.Tensor of shape (batch_size, n_classes)
            Classification logits
        """
        # Encode
        encoded = self._encode_batch(xid)
        
        # MLP classification
        logits = self.mlp(encoded)
        
        return logits
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, seq_len)
            Tokenized SMILES sequences (train_xid)
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns:
        -------
        self : object
            Fitted classifier
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        X = np.array(X)
        y = np.array(y)
        
        # Store dataset info
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Encode all data once before training (encoder is frozen)
        if self.verbose:
            print("Encoding training data...")
        if self.encoder_on_cpu: 
            X_tensor = torch.LongTensor(X).to('cpu')  # Keep on CPU for encoding
        else:
            X_tensor = torch.LongTensor(X).to(self.device)
        
        # Encode in batches to avoid memory issues
        encoded_batches = []
        n_samples = X_tensor.shape[0]
        for i in range(0, n_samples, self.batch_size):
            batch = X_tensor[i:i+self.batch_size]
            encoded_batch = self._encode_batch(batch)
            encoded_batches.append(encoded_batch.cpu())  # Keep on CPU temporarily
        
        X_encoded = torch.cat(encoded_batches, dim=0)
        self.n_features_in_ = X_encoded.shape[1]
        
        # Build MLP if not already built
        if self.mlp is None:
            self._build_mlp(self.n_features_in_, self.n_classes_)
        
        # Now move encoded data to target device
        X_encoded = X_encoded.to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create DataLoader with pre-encoded features
        dataset = TensorDataset(X_encoded, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        # Setup optimizer and loss (only for MLP parameters)
        optimizer = optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.mlp.train()
        
        best_loss = float('inf')
        no_improvement_count = 0
        
        if self.verbose:
            print("Training MLP...")
        
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X_encoded, batch_y in dataloader:
                optimizer.zero_grad()
                
                # MLP forward (trainable) - data is already encoded
                outputs = self.mlp(batch_X_encoded)
                loss = criterion(outputs, batch_y)
                
                # Backward (only updates MLP)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Verbose output
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Iteration {epoch + 1}/{self.max_iter}, loss = {avg_loss:.6f}")
            
            # Early stopping check
            if self.early_stopping:
                if avg_loss < best_loss - self.tol:
                    best_loss = avg_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at iteration {epoch + 1}")
                    break
        
        self.n_iter_ = epoch + 1
        self.best_loss_ = best_loss
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, seq_len)
            Tokenized SMILES sequences (test_xid)
            
        Returns:
        --------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        self.mlp.eval()
        self.encoder.eval()
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.array(X)
        
        # Convert to tensor
        X_tensor = torch.LongTensor(X).to(self.device)
        
        # Create DataLoader for batched prediction
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
        )
        
        # Predict in batches
        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                # Encode
                encoded = self._encode_batch(batch_X)
                
                # MLP forward
                outputs = self.mlp(encoded)
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.vstack(predictions)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, seq_len)
            Tokenized SMILES sequences
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def save_weights(self, path: str):
        """
        Save only the MLP weights (encoder is frozen and shared).
        
        Parameters:
        -----------
        path : str
            Path to save the MLP weights
        """
        state = {
            'mlp_state_dict': self.mlp.state_dict(),
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'classes_': self.classes_,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation_name': self.activation_name
        }
        torch.save(state, path)
        if self.verbose:
            print(f"Model weights saved to {path}")
    
    def load_weights(self, path: str):
        """
        Load MLP weights.
        
        Parameters:
        -----------
        path : str
            Path to load the MLP weights from
        """
        state = torch.load(path, map_location=self.device)
        
        # Restore metadata
        self.n_classes_ = state['n_classes_']
        self.n_features_in_ = state['n_features_in_']
        self.classes_ = state['classes_']
        
        # Build MLP if needed
        if self.mlp is None:
            self._build_mlp(self.n_features_in_, self.n_classes_)
        
        # Load weights
        self.mlp.load_state_dict(state['mlp_state_dict'])
        self.mlp.to(self.device)
        
        if self.verbose:
            print(f"Model weights loaded from {path}")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override to handle custom checkpoint format with only MLP weights.
        
        This method intercepts the standard PyTorch load_state_dict to handle
        the custom checkpoint format that only contains MLP weights and metadata,
        not encoder weights (since encoder is frozen and shared).
        
        Parameters:
        -----------
        state_dict : dict
            Either:
            - Custom checkpoint dict with 'mlp_state_dict' key
            - Direct MLP state_dict 
        strict : bool, default=True
            Whether to strictly enforce that keys match (passed to MLP)
        assign : bool, default=False
            PyTorch parameter (ignored for compatibility)
            
        Returns:
        --------
        NamedTuple with missing_keys and unexpected_keys (PyTorch format)
            
        Examples:
        ---------
        # Load custom checkpoint
        ckpt = torch.load("weights.pth", map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt)
        """
        from torch.nn.modules.module import _IncompatibleKeys
        
        # Check if this is our custom checkpoint format
        if 'mlp_state_dict' in state_dict:
            # Custom checkpoint - extract metadata and MLP weights
            self.n_classes_ = state_dict['n_classes_']
            self.n_features_in_ = state_dict['n_features_in_']
            self.classes_ = state_dict['classes_']
            mlp_state_dict = state_dict['mlp_state_dict']
            
            # Build MLP if it doesn't exist yet
            if self.mlp is None:
                self._build_mlp(self.n_features_in_, self.n_classes_)
            
            # Load the MLP weights and get the result
            result = self.mlp.load_state_dict(mlp_state_dict, strict=strict)
            self.mlp.to(self.device)
            
            if self.verbose:
                print(f"MLP weights loaded successfully")
                
            # Return the _IncompatibleKeys from MLP load
            return result
        
        # If it looks like a direct MLP state dict (has Linear layer keys)
        elif any(key.startswith('0.') or key.startswith('mlp.') for key in state_dict.keys()):
            # Build MLP if it doesn't exist yet
            if self.mlp is None:
                raise RuntimeError(
                    "Cannot load MLP state dict without n_features_in_ and n_classes_. "
                    "Either fit the model first or load a full checkpoint with metadata."
                )
            
            # Load the MLP weights and get the result
            result = self.mlp.load_state_dict(state_dict, strict=strict)
            self.mlp.to(self.device)
            
            if self.verbose:
                print(f"MLP weights loaded successfully")
                
            # Return the _IncompatibleKeys from MLP load
            return result
        
        else:
            # Not our format - call parent's load_state_dict
            # This would be for loading full model state dicts including encoder
            return super().load_state_dict(state_dict, strict=strict, assign=assign)