# Table-Converter
This script uses a BERT model to find the most similar columns between tables A and B and a template table. It encodes each column as a vector and computes the cosine similarity. Then, it selects the columns with the highest similarity for each column of the template table and merges them into a new table. Finally, it converts some columns of the new table to match the template table.
### Libraries 📕 
```
import sys
import torch
import pandas as pd
import numpy as np
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity
```
### How to use? 🤔
Paste this to terminal with specified paths
```
pip install -r requirements.txt
python convert_tables.py <path-to-tableA> <path-to-tableB> <path-to-template-table>
```
