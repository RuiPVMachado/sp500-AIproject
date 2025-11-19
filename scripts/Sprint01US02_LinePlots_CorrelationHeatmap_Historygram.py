# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:19:51 2025

@author: 2017i
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

# Add project root folder to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.load_data import load_raw_data, dataset_summary

# Load the dataset
df = load_raw_data()

# Quick check
meta, missing = dataset_summary(df)
print(meta)
print(missing)

#Line plot creation
plt.figure(figsize=(12,6))
plt.plot(df.index, df['SP500'], label='SP500')
plt.plot(df.index, df['Real Price'], label="Real Price")
plt.plot (df.index, df['PE10'], label='PE10')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('S&P 500, Real Price and PE10 over the time')
plt.legend()
plt.show()

#Correlation heatmap
import seaborn as sns
corr = df[['SP500', 'Real Price', 'PE10']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()

#Historygrams
df[['SP500', 'Real Price', 'PE10']].hist(bins=20, figsize=(12,6))
plt.show()


