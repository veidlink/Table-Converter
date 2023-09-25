# Table-Converter
This script uses a BERT model to find the most similar columns between tables A and B and a template table. It encodes each column as a vector and computes the cosine similarity. Then, it selects the columns with the highest similarity for each column of the template table and merges them into a new table. Finally, it converts some columns of the new table to match the template table by generating conversion functions with ChatGPT.
### Libraries 📕 
```
import sys
import torch
import openai
import pandas as pd
import numpy as np
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity
```
### How to use? 🤔
Install libraries
```pip install -r requirements.txt```
Paste this prompt to terminal
```python convert_tables.py <path-to-tableA> <path-to-tableB> <path-to-template-table> <YOUR OPENAI API KEY>```
