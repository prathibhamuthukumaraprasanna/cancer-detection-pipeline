"""
Cancer Detection Pipeline

This script implements a complete pipeline for classifying biomedical text into three 
cancer types: Thyroid Cancer, Colon Cancer, and Lung Cancer.

Based on the project proposal by Prathibha Muthukumara Prasanna.

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - transformers
    - torch
    - tqdm

Run this script with Python 3.x after installing the required dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tqdm
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import nltk
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
nltk.download('punkt')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Download NLTK data for word tokenization
nltk.download('punkt', quiet=True)

class CancerDataLoader:
    """
    Handles loading and preprocessing of cancer dataset from parquet files.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the parquet files
        """
        self.data_dir = data_dir
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.label_to_id = None
        self.id_to_label = None
        
    def load_data(self):
        """Load train, validation and test data from parquet files."""
        try:
            # Load data from parquet files
            self.train_data = pd.read_parquet(os.path.join(self.data_dir, "train.parquet"))
            self.val_data = pd.read_parquet(os.path.join(self.data_dir, "validation.parquet"))
            self.test_data = pd.read_parquet(os.path.join(self.data_dir, "test.parquet"))
            
            # Create label mappings
            unique_labels = sorted(self.train_data['label'].unique())
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            
            print(f"Loaded train data: {len(self.train_data)} samples")
            print(f"Loaded validation data: {len(self.val_data)} samples")
            print(f"Loaded test data: {len(self.test_data)} samples")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_combined_data(self):
        """Return combined dataset for EDA purposes."""
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
    
    def preprocess_text(self, text):
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_preprocessing(self):
        """Apply preprocessing to all datasets."""
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Add processed text column to each dataset
        self.train_data['processed_text'] = self.train_data['input'].apply(self.preprocess_text)
        self.val_data['processed_text'] = self.val_data['input'].apply(self.preprocess_text)
        self.test_data['processed_text'] = self.test_data['input'].apply(self.preprocess_text)
        
        # Convert labels to numeric
        self.train_data['label_id'] = self.train_data['label'].map(self.label_to_id)
        self.val_data['label_id'] = self.val_data['label'].map(self.label_to_id)
        self.test_data['label_id'] = self.test_data['label'].map(self.label_to_id)
        
        print("Preprocessing completed.")


class ExploratoryDataAnalysis:
    """
    Performs exploratory data analysis on the cancer dataset.
    """
    
    def __init__(self, data):
        """
        Initialize the EDA class.
        
        Args:
            data: Pandas DataFrame containing the dataset
        """
        self.data = data
    
    def analyze_label_distribution(self):
        """Analyze and visualize the label distribution."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='label', data=self.data)
        plt.title("Distribution of Cancer Types")
        plt.xlabel("Cancer Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("label_distribution.png")
        plt.close()
        
        # Print label counts
        label_counts = self.data['label'].value_counts()
        print("Label Distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples ({count/len(self.data)*100:.2f}%)")
    
    def analyze_text_length(self):
        """Analyze and visualize the text length distribution using simple splitting."""
        # Calculate word counts using simple split instead of NLTK tokenizer
        self.data['word_count'] = self.data['input'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Plot histogram of word counts
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['word_count'], bins=50, kde=True)
        plt.title("Distribution of Text Length (Word Count)")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.axvline(x=self.data['word_count'].median(), color='red', linestyle='--', 
                    label=f"Median: {self.data['word_count'].median()}")
        plt.axvline(x=512, color='green', linestyle='--', label="BERT Limit: 512 tokens")
        plt.legend()
        plt.tight_layout()
        plt.savefig("text_length_distribution.png")
        plt.close()
        
        # Print statistics
        print("\nText Length Statistics (Word Count):")
        print(f"  Mean: {self.data['word_count'].mean():.2f}")
        print(f"  Median: {self.data['word_count'].median():.2f}")
        print(f"  Min: {self.data['word_count'].min()}")
        print(f"  Max: {self.data['word_count'].max()}")
        print(f"  Samples exceeding 512 words: {(self.data['word_count'] > 512).sum()} " +
            f"({(self.data['word_count'] > 512).sum()/len(self.data)*100:.2f}%)")
    def display_sample_texts(self, n=5):
        """Display sample texts from each category."""
        for label in self.data['label'].unique():
            print(f"\nSample texts for {label}:")
            samples = self.data[self.data['label'] == label]['input'].sample(n, random_state=42).values
            for i, sample in enumerate(samples, 1):
                if isinstance(sample, str):
                    if len(sample) > 300:
                        sample = sample[:300] + "..."
                    print(f"  Sample {i}: {sample}")
    
    def perform_full_eda(self):
        """Perform complete exploratory data analysis."""
        print("Starting Exploratory Data Analysis...")
        self.analyze_label_distribution()
        self.analyze_text_length()
        self.display_sample_texts()
        print("EDA completed. Visualization files saved.")


class BaselineModel:
    """
    Implements the TF-IDF + Logistic Regression baseline model.
    """
    
    def __init__(self):
        """Initialize the baseline model components."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        self.trained = False
    
    def train(self, X_train, y_train):
        """
        Train the baseline model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        print("Vectorizing training data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"Train data shape after vectorization: {X_train_vec.shape}")
        
        print("Training logistic regression model...")
        self.classifier.fit(X_train_vec, y_train)
        self.trained = True
        print("Baseline model training completed.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the baseline model.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Vectorizing test data...")
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Predicting on test data...")
        y_pred = self.classifier.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, 
            y_pred, 
            average='macro'
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        print("\nBaseline Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Baseline Model")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("baseline_confusion_matrix.png")
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }


class TransformerModel:
    """
    Implements a transformer-based model for cancer text classification.
    """
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", num_labels=3):
        """
        Initialize the transformer model.
        
        Args:
            model_name: Pre-trained model to use
            num_labels: Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.max_length = 512  # Maximum sequence length for BERT models
    
    def setup_model(self):
        """Set up the transformer model and tokenizer."""
        print(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        print("Model and tokenizer loaded successfully.")
    
    def preprocess_for_transformer(self, examples):
        """
        Tokenize and prepare data for transformer model.
        
        Args:
            examples: Input examples
            
        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
    
    def prepare_datasets(self, train_df, val_df, test_df):
        """
        Convert DataFrames to HuggingFace datasets.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Convert DataFrames to HuggingFace Dataset objects
        train_dataset = Dataset.from_pandas(train_df[['input', 'label_id']])
        val_dataset = Dataset.from_pandas(val_df[['input', 'label_id']])
        test_dataset = Dataset.from_pandas(test_df[['input', 'label_id']])
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.preprocess_for_transformer,
            batched=True,
            remove_columns=['input']
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_for_transformer,
            batched=True,
            remove_columns=['input']
        )
        
        test_dataset = test_dataset.map(
            self.preprocess_for_transformer,
            batched=True,
            remove_columns=['input']
        )
        
        # Rename 'label_id' to 'labels' for Trainer compatibility
        train_dataset = train_dataset.rename_column('label_id', 'labels')
        val_dataset = val_dataset.rename_column('label_id', 'labels')
        test_dataset = test_dataset.rename_column('label_id', 'labels')
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        print(f"Prepared datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    # def train(self, train_dataset, val_dataset, output_dir="transformer_model", batch_size=8, epochs=3):
    #     """
    #     Train the transformer model.
        
    #     Args:
    #         train_dataset: Training dataset
    #         val_dataset: Validation dataset
    #         output_dir: Directory to save the model
    #         batch_size: Batch size for training
    #         epochs: Number of training epochs
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not set up. Call setup_model() first.")
        
    #     # Create output directory if it doesn't exist
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Set up training arguments
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         evaluation_strategy="epoch",
    #         save_strategy="epoch",
    #         learning_rate=2e-5,
    #         per_device_train_batch_size=batch_size,
    #         per_device_eval_batch_size=batch_size,
    #         num_train_epochs=epochs,
    #         weight_decay=0.01,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="accuracy",
    #         push_to_hub=False,
    #         report_to="none"
    #     )
        
    #     # Set up data collator
    #     data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
    def train(self, train_dataset, val_dataset, output_dir="transformer_model", batch_size=8, epochs=3):
        """
        Train the transformer model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save the model
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            weight_decay=0.01
        )
        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     # Remove or replace problematic parameters
        #     eval_steps=100,  # Evaluate every 100 steps
        #     save_steps=100,  # Save every 100 steps
        #     logging_steps=100,  # Log every 100 steps
        #     learning_rate=2e-5,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     num_train_epochs=epochs,
        #     weight_decay=0.01,
        #     load_best_model_at_end=True,
        #     metric_for_best_model="accuracy",
        #     push_to_hub=False
        # )
        
        # Rest of the function remains the same...
        # Set up data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, 
                predictions, 
                average='macro'
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Set up trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        print("Starting transformer model training...")
        self.trainer.train()
        
        # Save the model
        self.model.save_pretrained(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        print(f"Transformer model trained and saved to {output_dir}/final_model")
            # Define compute_metrics function
        # def compute_metrics(eval_pred):
        #     predictions, labels = eval_pred
        #     predictions = np.argmax(predictions, axis=1)
            
        #     accuracy = accuracy_score(labels, predictions)
        #     precision, recall, f1, _ = precision_recall_fscore_support(
        #         labels, 
        #         predictions, 
        #         average='macro'
        #     )
            
        #     return {
        #         'accuracy': accuracy,
        #         'precision': precision,
        #         'recall': recall,
        #         'f1': f1
        #     }
        
        # # Set up trainer
        # self.trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=val_dataset,
        #     tokenizer=self.tokenizer,
        #     data_collator=data_collator,
        #     compute_metrics=compute_metrics
        # )
        
        # print("Starting transformer model training...")
        # self.trainer.train()
        
        # # Save the model
        # self.model.save_pretrained(os.path.join(output_dir, "final_model"))
        # self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        # print(f"Transformer model trained and saved to {output_dir}/final_model")
    
    def evaluate(self, test_dataset, id_to_label):
        """
        Evaluate the transformer model.
        
        Args:
            test_dataset: Test dataset
            id_to_label: Mapping from label IDs to label names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Evaluating transformer model...")
        eval_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Convert numeric predictions and labels to original class names for better readability
        pred_labels = [id_to_label[p] for p in preds]
        true_labels = [id_to_label[l] for l in labels]
        
        # Generate confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Generate classification report
        report = classification_report(
            labels, 
            preds, 
            target_names=list(id_to_label.values())
        )
        
        print("\nTransformer Model Evaluation:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClassification Report:")
        print(report)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(id_to_label.values()),
            yticklabels=list(id_to_label.values())
        )
        plt.title("Confusion Matrix - Transformer Model")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("transformer_confusion_matrix.png")
        plt.close()
        
        # Extract and standardize metrics for comparison
        # Map eval_X keys to X for compatibility with compare_models function
        standardized_results = {}
        for key in eval_results:
            if key.startswith('eval_'):
                standard_key = key[5:]  # Remove 'eval_' prefix
                standardized_results[standard_key] = eval_results[key]
        
        # If metrics are missing, calculate them from predictions
        if 'accuracy' not in standardized_results:
            standardized_results['accuracy'] = accuracy_score(labels, preds)
        if 'precision' not in standardized_results:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            standardized_results['precision'] = precision
            standardized_results['recall'] = recall
            standardized_results['f1'] = f1
        
        # Add confusion matrix and report
        standardized_results['confusion_matrix'] = cm
        standardized_results['classification_report'] = report
        
        return standardized_results

def compare_models(baseline_results, transformer_results):
    """
    Compare and visualize results from both models.
    
    Args:
        baseline_results: Results from baseline model
        transformer_results: Results from transformer model
    """
    # Extract metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Ensure all required metrics exist
    for m in metrics:
        if m not in baseline_results:
            print(f"Warning: '{m}' metric missing from baseline results")
        if m not in transformer_results:
            print(f"Warning: '{m}' metric missing from transformer results")
    
    baseline_metrics = [baseline_results.get(m, 0) for m in metrics]
    transformer_metrics = [transformer_results.get(m, 0) for m in metrics]
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_metrics, width, label='TF-IDF + LogReg')
    plt.bar(x + width/2, transformer_metrics, width, label='BioBERT')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(baseline_metrics):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(transformer_metrics):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print(f"{'Metric':<12} {'Baseline':<12} {'Transformer':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in metrics:
        baseline_val = baseline_results.get(metric, 0)
        transformer_val = transformer_results.get(metric, 0)
        improvement = transformer_val - baseline_val
        improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
        
        print(f"{metric:<12} {baseline_val:.4f}{'':<8} {transformer_val:.4f}{'':<8} {improvement:.4f} ({improvement_pct:+.1f}%)")
def main():
    """Main execution function."""
    print("Cancer Detection Pipeline - Starting...")
    
    # Set up directories
    os.makedirs("data", exist_ok=True)
    print("Please ensure your train.parquet, validation.parquet, and test.parquet files are in the 'data' directory.")
    
    # Create data loader and load data
    data_loader = CancerDataLoader()
    if not data_loader.load_data():
        print("Failed to load data. Please check that your data files exist and are correctly formatted.")
        return
    
    # Preprocess the data
    data_loader.apply_preprocessing()
    
    # Perform exploratory data analysis
    combined_data = data_loader.get_combined_data()
    eda = ExploratoryDataAnalysis(combined_data)
    eda.perform_full_eda()
    
    # Train and evaluate baseline model
    print("\n" + "="*50)
    print("BASELINE MODEL: TF-IDF + Logistic Regression")
    print("="*50)
    
    baseline = BaselineModel()
    baseline.train(
        data_loader.train_data['processed_text'],
        data_loader.train_data['label_id']
    )
    
    baseline_results = baseline.evaluate(
        data_loader.test_data['processed_text'],
        data_loader.test_data['label_id']
    )
    
    # Check if GPU is available for transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Train and evaluate transformer model
    print("\n" + "="*50)
    print("TRANSFORMER MODEL: BioBERT")
    print("="*50)
    
    transformer = TransformerModel(num_labels=len(data_loader.label_to_id))
    transformer.setup_model()
    
    # Prepare datasets for transformer model
    train_dataset, val_dataset, test_dataset = transformer.prepare_datasets(
        data_loader.train_data,
        data_loader.val_data,
        data_loader.test_data
    )
    
    # Train transformer model
    transformer.train(
        train_dataset,
        val_dataset,
        batch_size=8 if torch.cuda.is_available() else 4,  # Reduce batch size if using CPU
        epochs=3
    )
    
    # Evaluate transformer model
    transformer_results = transformer.evaluate(
        test_dataset,
        data_loader.id_to_label
    )
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    compare_models(baseline_results, transformer_results)
    
    print("\nCancer Detection Pipeline - Completed")
    print("Results and visualizations saved to the current directory.")


if __name__ == "__main__":
    main()