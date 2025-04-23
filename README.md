# Cancer Detection Pipeline

A machine learning pipeline for classifying biomedical text into three cancer types: Thyroid Cancer, Colon Cancer, and Lung Cancer.

## Project Overview

This project implements and compares two approaches for cancer type classification from biomedical text:
1. A baseline model using TF-IDF features with logistic regression
2. A state-of-the-art transformer model (BioBERT) fine-tuned for cancer classification

The pipeline includes comprehensive data preprocessing, exploratory data analysis, model training, evaluation, and comparison.

transformer_model Folder is uploaded here: [Google Drive](https://drive.google.com/drive/folders/12bBw1PWfXB8m9I4X1T-ikK0im_OQVmhG?usp=sharing)

## Results

The transformer-based approach significantly outperforms the baseline model:

![Model Performance Comparison](img1.png)

Confusion matrices show that the transformer model reduces misclassifications between cancer types:

**Baseline Model Confusion Matrix**
![Baseline Model Confusion Matrix](img3.png)

**Transformer Model Confusion Matrix**
![Transformer Model Confusion Matrix](img2.png)

A significant challenge identified is the length of biomedical texts, with most exceeding the standard BERT token limit:

![Text Length Distribution](img4.png)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
transformers
torch
tqdm
nltk
```

Install all dependencies with:
```
pip install -r requirements.txt
```

## Dataset

The pipeline expects three parquet files in a `data` directory:
- `train.parquet`
- `validation.parquet`
- `test.parquet`

Each file should contain biomedical text samples with corresponding cancer type labels.

## Usage

### Running the Pipeline

```python
python cancer-detection-pipeline.py
```

The script will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Train and evaluate the baseline model
4. Train and evaluate the transformer model
5. Compare both models
6. Generate visualizations

### Key Components

- `CancerDataLoader`: Handles loading and preprocessing of cancer dataset
- `ExploratoryDataAnalysis`: Performs EDA on the dataset
- `BaselineModel`: Implements TF-IDF + Logistic Regression model
- `TransformerModel`: Implements BioBERT model
- `compare_models`: Compares and visualizes results from both models

## Project Structure

```
cancer-detection-pipeline/
├── cancer-detection-pipeline.py    # Main Python script
├── data/                           # Data directory
│   ├── train.parquet               # Training data
│   ├── validation.parquet          # Validation data
│   └── test.parquet                # Test data
├── transformer_model/              # Transformer Model directory
│   ├── config.json                 
│   ├── special_tokens_map.json     
│   ├── tokenizer_config.json       
│   ├── tokenizer.json              
│   └── vocab.txt  
├── img1.png                        # Model comparison visualization
├── img2.png                        # Transformer confusion matrix
├── img3.png                        # Baseline confusion matrix
├── img4.png                        # Text length distribution
├── requirements.txt                # Package dependencies
└── README.md                       # This file
```

## Performance Metrics

| Metric | TF-IDF + LogReg | BioBERT | Improvement |
|--------|-----------------|---------|-------------|
| Accuracy | 92.7% | 97.7% | +5.0% |
| Precision | 92.7% | 97.7% | +5.0% |
| Recall | 92.7% | 97.7% | +5.0% |
| F1 Score | 92.7% | 97.7% | +5.0% |

## Key Findings

1. The transformer-based BioBERT model consistently outperforms the traditional TF-IDF approach across all metrics
2. Both models handle lung cancer classification extremely well
3. The main improvement of the transformer model is in reducing confusion between thyroid and colon cancer cases
4. The majority of text samples exceed the standard 512 token limit of BERT, requiring truncation strategies

## Future Work

- Explore techniques for handling longer texts, such as hierarchical models or sliding window approaches
- Incorporate domain-specific medical ontologies to enhance feature representation
- Experiment with other biomedical language models like ClinicalBERT and SciBERT
- Implement multi-label classification to identify co-occurring cancer types

## Author

Prathibha Muthukumara Prasanna  
Department of Biostatistics  
University of Michigan  
prathii@umich.edu

## Citation

If you use this code in your research, please cite:

```
@misc{prathii2025cancerdetection,
  author = {Muthukumara Prasanna, Prathibha},
  title = {BioBERT vs. TF-IDF for Multi-Class Cancer Classification from Biomedical Text},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/prathii-05/cancer-detection-pipeline}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
