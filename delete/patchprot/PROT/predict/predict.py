import sys
import torch
import numpy as np
import os
import math
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from io import StringIO
from Bio import SeqIO

from PROT.base.base_predict import BasePredict
from PROT.data_loader.augmentation import string_token
from PROT.models.metric import arctan_dihedral
from PROT.biolib import plot_features

SS3 = ['H','E','C']
SS8 = ['G','H','I','B','E','S','T','C']
EXP = [f'{i}%' for i in range(0,100,10)]
SPECIES = ['HOMO SAPIENS','ESCHERICHIA COLI','SACCHAROMYCES CEREVISIAE', 'MUS MUSCULUS', 'BACILIUS SUBTILIS', 'PSEUDOMONAS AERUGINOSA', 'MYCOBACTERIUM TUBERCULOSIS', 'THERMUS THERMOPHILUS', 'ARABIDOPSIS THALIANA', 'THERMOTOGA MARITIMA']

def list_to_chunked_list(input_list, chunk_size):
    chunked_sequences = []
    for chunk_offset in range(0, len(input_list), chunk_size):
        chunked_sequences.append(input_list[chunk_offset:chunk_offset + chunk_size])
    return chunked_sequences


class SecondaryFeatures(BasePredict):

    def __init__(self, model, model_data, vizualization):
        super(SecondaryFeatures, self).__init__(model, model_data)
        """ Predict secondary features by using raw AA sequence
        Args:
            model: instantiated model class
            model_data: path to the trained model data
        """
        self.transform = string_token()
        self.vizualize = vizualization

    def preprocessing(self, x) -> list:
        """ Loads and preprocess the a file path or string
        Args:
            x: path or string containing fasta sequences
        """
        sequences = []

        # Parse fasta file or fasta input string
        try:
            for seq_record in SeqIO.parse(x, "fasta"):
                sequences.append((seq_record.id, str(seq_record.seq)))
        except FileNotFoundError:
            print("File not found. Trying to parse argument instead...")
            fastq_io = StringIO(x)
            for seq_record in SeqIO.parse(fastq_io, "fasta"):
                sequences.append((seq_record.id, str(seq_record.seq)))
            fastq_io.close()

        # Exit if parsing not possible
        if not sequences:
            print("Parsing failed. Please input a correct fasta file")
            sys.exit()

        mask = torch.tensor([len(seq[1]) for seq in sequences])

        return sequences, mask

    def inference(self, x: list, mask: torch.tensor) -> torch.tensor:
        """ Predicts the secondary structures
        Args:
            x: list containing a tuple with name and protein sequence
        """
        x = self.transform(x)
        
        if torch.cuda.is_available():
            x = x.to('cuda:0')
        with torch.no_grad():
            x = self.model(x, mask)
        return x

    def postprocessing(self, x: torch.tensor, mask: torch.tensor):
        """ Proces the prediction results
        Args:
            x: model predictions
        """
        # convert phi and psi radians to angles
        for i in range(x[0].shape[0]):
            x[0][i] = F.softmax(x[0][i], dim=1)
            x[1][i] = F.softmax(x[1][i], dim=1)
            x[2][i] = F.softmax(x[2][i], dim=1)
            
            x[4][i, :, 0] = arctan_dihedral(x[4][i][:, 0], x[4][i][:, 1])
            x[5][i, :, 0] = arctan_dihedral(x[5][i][:, 0], x[5][i][:, 1])

            x[9][i, :, 1] = F.sigmoid(x[9][i][:, 1])
            x[10][i, :, 1] = F.sigmoid(x[10][i][:, 1])

        x[4] = x[4][:, :, 0].unsqueeze(2)
        x[5] = x[5][:, :, 0].unsqueeze(2)

        return x

    def __call__(self, x):
        """ Prediction call logic """
        fasta, mask = self.preprocessing(x)

        result_file = 'results.csv'
        identifier = []
        sequence = []
        prediction = []
        list_mask = []
        batch_size = int(250/25)
        chunk_size = 25
        sequences_chunked = list_to_chunked_list(fasta, chunk_size)
        
        import datetime
        print(f"Processing sequences in batches of {chunk_size} // super batch of {batch_size*chunk_size} ... ")
        with tqdm(total=len(fasta), desc='Generating predictions', unit='seq') as progress_bar:
            for idx_m in range(math.ceil(len(fasta)/batch_size)):
                mega_batch = sequences_chunked[idx_m*batch_size:(idx_m+1)*batch_size]
                list_global = []
                identifier = []
                sequence = []
                prediction = []
                list_mask = []
                for _,chunk in enumerate(mega_batch):
                    chunk_mask = torch.tensor([len(seq[1]) for seq in chunk])

                    x = self.inference(chunk, chunk_mask)
                    x = self.postprocessing(x, chunk_mask)

                    identifier.append([element[0] for element in chunk])
                    sequence.append([element[1] for element in chunk])
                    prediction.append([torch.Tensor.cpu(x[i]).detach().numpy() for i in range(len(x))])
                    list_mask.append([torch.Tensor.cpu(chunk_mask[i]).detach().numpy() for i in range (len(chunk_mask))])

                    del x
                    
                    progress_bar.update(len(chunk))

                list_mask = np.array(list_mask)
                # For each batch
                for i in range(len(identifier)):
                    # For each sequence
                    for j in range(len(identifier[i])):
                        # Getting each id + fasta sequence
                        id_prot, fasta = identifier[i][j], sequence[i][j]
                        SS3_prot, SS8_prot = "".join(map(lambda r: SS3[r], np.argmax(prediction[i][1][j,:list_mask[i][j],:],axis=1))), "".join(map(lambda r: SS8[r], np.argmax(prediction[i][0][j,:list_mask[i][j],:],axis=1)))
                        species, expression = SPECIES[np.argmax(np.sum(prediction[i][11][j,:list_mask[i][j]], axis=0))], EXP[np.argmax(np.sum(prediction[i][12][j,:list_mask[i][j]], axis=0))]
                        rsa, disorder, hp_loc, lhp_loc = prediction[i][3][j,:list_mask[i][j],0], prediction[i][2][j,:list_mask[i][j],1], prediction[i][9][j,:list_mask[i][j],1], prediction[i][10][j,:list_mask[i][j],1]
                        tasa, thsa, lhp = np.sum(prediction[i][6][j,:list_mask[i][j],0]), np.sum(prediction[i][7][j,:list_mask[i][j],0]), np.sum(prediction[i][8][j,:list_mask[i][j],0])
                        list_global.append([id_prot, fasta, SS3_prot, SS8_prot, tasa, thsa, thsa/tasa, lhp, species, expression])
                        if self.vizualize:
                            plot_features(identifier[i][j],[*fasta],[*SS8_prot],[*rsa],[*hp_loc],[*lhp_loc],[*disorder],[tasa, thsa, lhp, species, expression])
                        list_local = {'Fasta_sequence':[*fasta],'RSA_predicted':[*rsa],'Disorder_predicted':[*disorder],'LHP_predicted':[*lhp_loc]}
                        df_loc = pd.DataFrame(data=list_local)
                        df_loc.to_csv(f'{id_prot}.csv',index=False)

                df_new = pd.DataFrame(np.array(list_global), columns = ['ID', 'Fasta_sequence', 'SS3_predicted','SS8_predicted','TASA_predicted', 'THSA_predicted','RHSA_predicted', 'LHP_predicted', 'Species', 'Expression'])
                if os.path.isfile(result_file):
                    df_old = pd.read_csv(result_file)
                    df = pd.concat([df_old,df_new], ignore_index = True)
                    df.reset_index()
                else:
                    df = df_new
                df.to_csv(result_file, index=False)
                del mega_batch, list_global, identifier, sequence, prediction, list_mask
                torch.cuda.empty_cache()

        return "Finished", "Finished", "Finished", "Finished"
