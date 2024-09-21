import numpy as np
import pandas as pd
import re

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

newsgroups_train = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
print(dir(newsgroups_train))

df = pd.DataFrame()
df['text'] = newsgroups_train.data
df['text'] = df['text'].str.replace('\\n','\\\\n', regex=True)
df['target'] = [newsgroups_train.target_names[x] for x in newsgroups_train.target] 
df['filename'] = [re.search(r'[^//]+$', filename).group() for filename in newsgroups_train.filenames]

df.to_csv('20ng.csv', index=False)
