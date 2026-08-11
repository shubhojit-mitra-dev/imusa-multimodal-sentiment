# IMUSA Core ML Library

This package contains the core machine learning models, dataset cleaning/processing utilities, training loops, and inference logic for the Indic Meme Understanding & Sentiment Analysis (IMUSA) project.

## Usage

```python
from imusa.config import settings
from imusa.data.cleaning import clean_dataset
from imusa.data.explorer import explore_dataset

# Run dataset cleaning
clean_df = clean_dataset()

# Run dataset EDA
stats = explore_dataset()
```
