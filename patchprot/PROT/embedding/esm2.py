import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import h5py

from PROT.data_loader.augmentation import sparse_token
from PROT.esm import ESM2, Alphabet
from PROT.esm.esm.pretrained import load_model_and_alphabet_local
import argparse
from argparse import Namespace


class ESM2Embedding(nn.Module):
    """ ESM2 embedding layer module """

    def __init__(self, embedding_args: dict = None, embedding_pretrained: str = None, finetuning: str = 'no finetuning', adapters_layer: int = 20, rank_LoRA: int = 6, gradient_checkpointing: int = None, ft_embed_tokens: bool = False, ft_transformer: bool = False, ft_contact_head: bool = False,
                 ft_layer_norm: bool = False, ft_lm_head: bool = False, max_embedding: int = 1024, offset: int = 200, trimming: int = 100):
        """ Constructor
        Args: 1024 MAX EMBEDDING
            embedding_args: arguments to embeddings model
            embedding_pretrained: path to pretrained model
            ft_embed_tokens: finetune embed tokens layer
            ft_transformer: finetune transformer layer
            ft_contact_head: finetune contact head
            ft_embed_positions: finetune embedding positions
            ft_emb_layer_norm_before: finetune embedding layer norm before
            ft_emb_layer_norm_after: finetune embedding layer norm after
            ft_lm_head: finetune lm head layer
            max_embeddings: maximum sequence length for language model
            offset: overlap offset when concatenating sequences above max embedding
        """
        super(ESM2Embedding, self).__init__()


        self.max_embedding = max_embedding
        self.offset = offset
        self.trimming = trimming
        self.finetuning = finetuning
        self.model, self.alphabet = load_model_and_alphabet_local(embedding_pretrained, finetuning=self.finetuning, adapters_layer=adapters_layer, rank_LoRA=rank_LoRA, gradient_checkpointing=gradient_checkpointing)

        # classical finetuning, freezes all layers by default
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
            ft_layer_norm, ft_lm_head]
 
        for i, child in enumerate(self.model.children()):
            if self.finetune[i] == False:
                for param in child.parameters():
                    param.requires_grad = False
        # this loop go through each transformer of the ESM model and 
        # freezes every layer for which we already have pretrained
        # weights
            if i == 1:
                # Adapters finetuning by freezing unchoosen layers
                if self.finetuning == 'adapters':
                    for j, transform in enumerate(child):
                        for k, param in enumerate(transform.parameters()):
                            if (len(list(transform.parameters()))-6<=k<=len(list(transform.parameters()))) and j >= len(list(child))-adapters_layer:
                                param.requires_grad = True
                # LoRA finetuning by freezing unchoosen layers
                if self.finetuning == 'LoRA':
                    for j, transform in enumerate(child):
                        for k, param in enumerate(transform.parameters()):
                            if k in [6,7,8,9]:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                                
    def forward(self, batch_tokens: torch.tensor, padding_length: int = None) -> torch.tensor:
        """ Convert tokens to embeddings
        Args:
            batch_tokens: tensor with sequence tokens
        """
        # length of each sequence (same for all sequences for example HHtrain.npz 1638)
        batch_residues_original = batch_tokens.shape[1]

        # remove padding ie. remove all the part of the tensor which is not in the sequence
        if padding_length:
            batch_tokens = batch_tokens[:, :padding_length]

        # new length
        batch_residues = batch_tokens.shape[1]

        # embedding for what we can (until max limit of ESM LM)
        embedding = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[33])["representations"][33]

        # if size above 1024 then generate embeddings that overlaps with the offset
        if batch_residues >= self.max_embedding:
            for i in range(1, math.floor(batch_residues / self.max_embedding) + 1):
                o1 = (self.max_embedding - self.offset) * i
                o2 = o1 + self.max_embedding
                # we implement trimming in order to get a better representation of long fasta sequences
                embedding = torch.cat([embedding[:, :o1 + self.trimming], self.model(batch_tokens[:, o1:o2], repr_layers=[33])["representations"][33][:,self.trimming:]], dim=1)

            # PyTorch 1.7 trick to do nan_to_num
            embedding[embedding != embedding] = 0.0

        # add padding
        if padding_length:
            embedding = F.pad(embedding, pad=(0, 0, 0, batch_residues_original
                            - padding_length), mode='constant', value=0)

        # cleanup
        del batch_tokens
        torch.cuda.empty_cache()
        return embedding[:, 1:embedding.shape[1]-1, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add embeddings to npz dataset. The numpy array has to indexed at name data")
    parser.add_argument("-i", "--input", help="Input path to data file")
    parser.add_argument(
        "-o", "--output", help="Output path to output of augmented file")
    parser.add_argument("-m", "--model", help="Model file path")
    args = parser.parse_args()

    EMBEDDING_SIZE = 1280

    model_path = (args.model or "../../pretrained/esm2_t33_650M_UR50D.pt")

    with h5py.File(args.output, "w") as f:
        dataset = np.load(args.input)["data"]
        sequences, residues, classes = dataset.shape

        # create dataset to augment
        augmented_dataset = f.create_dataset(
            "dataset", (sequences, residues, classes + EMBEDDING_SIZE), dtype="float64", compression="gzip", compression_opts=9)
        augmented_dataset[:sequences, :residues, :classes] = dataset

        decoded_sequences = sparse_token()(torch.from_numpy(dataset[:, :, :20]))
        print("Decoded sequences")

        with torch.no_grad():
            # Create embedding model
            model = ESM2Embedding({}, embedding_pretrained=model_path)

            # Try to move model to GPU if possible
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            try:
                model = model.cuda(device)
            except RuntimeError:
                device = 'cpu'
            # Deactivate dropout for deterministic results
            model = model.eval()
            # Generate embeddings with mini-batches in batches
            mini_batch = 2
            batch_size = 1000
            for i in range(0, sequences, batch_size):

                # empty array to store embeddings
                embedding = np.zeros([batch_size, residues, EMBEDDING_SIZE])

                # mini batches to reduce VRAM usage
                for j in range(0, batch_size, mini_batch):
                    early_break = False

                    # limit mini-batch
                    offset = 0
                    if j + mini_batch > batch_size:
                        offset = abs((j + mini_batch) - batch_size)
                    elif i + j + mini_batch > sequences:
                        offset = abs((i + j + mini_batch) - sequences)
                        early_break = True

                    # store embedding model
                    embedding_model = model(
                        decoded_sequences[i + j:i + j + mini_batch - offset].to(device)).cpu().detach().numpy()

                    # store embedding without the extra start and end token
                    embedding_residues = embedding_model.shape[1]
                    embedding[j:j + mini_batch - offset, :embedding_residues
                        - 2] = embedding_model[:, 1:embedding_residues - 1]
                    torch.cuda.empty_cache()

                    print("Batch {} out of {}".format(
                        i + j + mini_batch - offset, sequences))

                    if early_break:
                        break

                # limit the final batch
                offset = 0
                if i + batch_size > sequences:
                    offset = abs((i + batch_size) - sequences)

                # Add calculated embedding batch to augmented dataset
                embedding_sequences, embedding_residues, embedding_classes = embedding.shape
                augmented_dataset[i:(i + batch_size) - offset, :embedding_residues,
                                     classes:] = embedding[:batch_size - offset]

        print("Succesfully augmented dataset")
