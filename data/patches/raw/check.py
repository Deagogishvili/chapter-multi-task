import csv
import numpy as np
import pandas as pd 

path_avail=r"C:\Users\emman\OneDrive\Documents\Césure\Pays-Bas\PROT\PROT\data\ready_to_use_data.csv"
path_dataset=r"C:\Users\emman\OneDrive\Documents\Césure\Pays-Bas\PROT\PROT\data\Train_HHblits.npz"

avail=pd.read_csv(path_avail, delimiter=';')
count=0

with np.load(path_dataset) as data:
    list_name=data['pdbids']
    for i in list_name:
        if i.replace('-','').upper() in list(avail.id):
            count+=1
            print(i.replace('-','').upper())
            print(avail.loc[avail.id == i.replace('-','').upper()]['size'].item())
print("Percentage :",count/len(list_name))
print("Numbers :",count)
print("Percentage Avail :",count/len(list(avail.id)))