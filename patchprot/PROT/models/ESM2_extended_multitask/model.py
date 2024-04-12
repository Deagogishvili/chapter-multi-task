import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PROT.base import ModelBase
from PROT.utils import setup_logger
from PROT.embedding import ESM2Embedding


log = setup_logger(__name__)


class ESM2_extended_multitask(ModelBase):
    def __init__(self, init_n_channels: int, embedding_pretrained: str, out_channels: int = 32, cnn_layers: int = 2, kernel_size: tuple = (129, 257), padding: tuple = (64, 128), n_hidden: int = 1024, dropout: float = 0.5, lstm_layers: int = 2, **kwargs):
        """ Constructor
        Args:
            in_features: size of the embedding features
            language_model: path to the language model weights
        """
        super(ESM2_extended_multitask, self).__init__()

        self.embedding = ESM2Embedding(embedding_pretrained = embedding_pretrained, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p = dropout),
                nn.Conv1d(in_channels = init_n_channels, 
                          out_channels = out_channels,
                          kernel_size = kernel_size[i], 
                          padding = padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)


        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
            nn.Tanh()
        ])
        self.tasa = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 1),
        ])
        self.thsa = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 1),
        ])
        self.lhp = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 1),
        ])
        self.hp_loc = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
        ])
        self.lhp_loc = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
        ])
        self.species = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 10),
        ])
        self.expression = nn.Sequential(*[
            nn.Linear(in_features = 2 * n_hidden, out_features = 2),
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

        # Remove end and start token from fasta sequence
        max_length = x.size(1) - 2
        # Get embedding from ESM2
        x = self.embedding(x, torch.max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim = 1)
        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first = True, enforce_sorted = False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length = max_length, batch_first = True)
        x = self.lstm_dropout_layer(x)

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

        return [ss8, ss3, dis, rsa, phi, psi, tasa, thsa, lhp, hp_loc, lhp_loc, species, expression]
