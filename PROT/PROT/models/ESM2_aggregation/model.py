import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PROT.base import ModelBase
from PROT.utils import setup_logger
from PROT.embedding import ESM2Embedding


log = setup_logger(__name__)


class ESM2_aggregation(ModelBase):
    def __init__(self, init_n_channels: int, embedding_pretrained: str, **kwargs):
        """ Constructor
        Args:
            in_features: size of the embedding features
            language_model: path to the language model weights
        """
        super(ESM2_aggregation, self).__init__()

        self.embedding = ESM2Embedding(embedding_pretrained=embedding_pretrained, **kwargs)

        # Task block
        self.aggregation = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True, print: bool = False) -> list:
        """ Returns the parameters to learn """
        if print:
            log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                if print:
                    log.info("\t" + name)
                yield param
                
    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        x = self.embedding(x, max(mask))

        # hidden neurons to classes
        aggregation = self.aggregation(x)

        return [aggregation]
