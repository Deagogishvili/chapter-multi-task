import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PROT.base import ModelBase
from PROT.utils import setup_logger
from PROT.embedding import ESM2Embedding


log = setup_logger(__name__)


class ESM2_multitask(ModelBase):
    def __init__(self, init_n_channels: int, embedding_pretrained: str, **kwargs):
        """ Constructor
        Args:
            in_features: size of the embedding features
            language_model: path to the language model weights
        """
        super(ESM2_multitask, self).__init__()

        self.embedding = ESM2Embedding(embedding_pretrained=embedding_pretrained, **kwargs)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
            nn.Tanh()
        ])
        self.tasa = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=1),
        ])
        self.thsa = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=1),
        ])
        self.lhp = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=1),
        ])
        self.hp_loc = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
        ])
        self.lhp_loc = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
        ])
        self.species = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=10),
        ])
        self.expression = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=10),
        ])
        self.aggregation = nn.Sequential(*[
            nn.Linear(in_features=init_n_channels, out_features=2),
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True) -> list:
        """ Returns the parameters to learn """

        log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                log.info("\t" + name)
                yield param

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        x = self.embedding(x, max(mask))

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)
        tasa = self.tasa(x)
        thsa = self.thsa(x)
        lhp = self.lhp(x)
        hp_loc = self.hp_loc(x)
        lhp_loc = self.lhp_loc(x)
        species = self.species(x)
        expression = self.expression(x)
        aggregation = self.aggregation(x)

        return [ss8, ss3, dis, rsa, phi, psi, tasa, thsa, lhp, hp_loc, lhp_loc, species, expression, aggregation]
