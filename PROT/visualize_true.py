import sys
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PROT.biolib import plot_features
import argparse

CLI=argparse.ArgumentParser()

CLI.add_argument(
    "--vizualize",
    type=bool,
    default=False
)

CLI.add_argument(
  "--protids",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

# parse the command line
args = CLI.parse_args()

AA = ['X','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
SS8 = ['G','H','I','B','E','S','T','C']
HYDR = ['A','V','L','I','P','F','C']
EXP = [f'{i}%' for i in range(0,100,10)]
SPECIES = ['HOMO SAPIENS','ESCHERICHIA COLI','SACCHAROMYCES CEREVISIAE', 'MUS MUSCULUS', 'BACILIUS SUBTILIS', 'PSEUDOMONAS AERUGINOSA', 'MYCOBACTERIUM TUBERCULOSIS', 'THERMUS THERMOPHILUS', 'ARABIDOPSIS THALIANA', 'THERMOTOGA MARITIMA']

if __name__=='__main__':
    if args.vizualize:
      path="/scistor/informatica/emi232/PROT/PROT/data/thsa_tasa_lhp/TS115_HHblits_extended.npz"
      with np.load(path) as data:
          name=data['pdbids']
          prot=data['data']
          for i in args.protids:
              print(i)
              idx = np.where(name == i)[0][0]
              mask = prot[idx, :, 50]
              x = (np.argmax(prot[idx, :, :20], axis=1) + 1)[mask == 1]
              fasta = "".join(map(lambda r: AA[int(r)], x))
              s = np.argmax(prot[idx, :, 57:64], axis=1)[mask == 1]
              SS8_prot = "".join(map(lambda r: SS8[int(r)], s))
              rsa = prot[idx, :, 55][mask == 1]
              hydr=[bool(AA[j] in HYDR) for j in np.argmax(prot[idx,:,:20],axis=-1)]
              tasa = prot[idx, 0, 68]
              thsa = prot[idx, 0, 69]
              lhp = prot[idx, 0, 71]
              hp_loc = prot[idx, :, 72][mask == 1]
              lhp_loc = prot[idx, :, 73][mask == 1]
              species = SPECIES[np.argmax(prot[idx, 0, 74:84])]
              expression = EXP[np.argmax(prot[idx, 0, 85:95])]
              disorder = 1 - prot[idx, :, 51][mask == 1]
              plot_features('ground_truth/'+name[idx],[*fasta],[*SS8_prot],[*rsa],[*hp_loc], [*lhp_loc],[*disorder], [tasa, thsa, lhp, species, expression])