import pandas as pd
import pathlib
from os import listdir
import numpy as np

current_file_dir = pathlib.Path(__file__).parent
base_dir = current_file_dir.parent
files_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/data'))

columns =  ['act', 'rew', 'obs', 'dones']
df = pd.DataFrame(columns=columns)

for file in listdir(files_dir):
    if(file.split('.')[-1] == 'pkl'):
        file_path = '{}/{}'.format(files_dir, file)
        df2 = pd.read_pickle(file_path)
        df = pd.concat([df, df2])

print(df.info())
filename = 'forwarder_{}_steps.pkl'.format(len(df))
file_path = '{}/{}'.format(files_dir, filename)
df.to_pickle(file_path)

#df = pd.read_pickle('/Users/ilyakurinov/Documents/University/RL/data/expert_data_12_05_2023_20_42.pkl')
#df = pd.DataFrame({''})