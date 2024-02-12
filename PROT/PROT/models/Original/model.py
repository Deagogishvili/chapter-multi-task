import torch
import torch.nn as nn
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PROT.base import ModelBase
from PROT.utils import setup_logger

from PROT.embedding import ESM2Embedding

log = setup_logger(__name__)

class PositionalEncoding(nn.Module):
    """ Injects some information about the relative or absolute position of the tokens in the sequence """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """ Constructor
        Args:
            d_model: size of input
            dropout: amount of hidden neurons in the bidirectional lstm
            max_len: amount of cnn layers
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNbLSTM(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int):
        """ Baseline model for CNNbLSTM
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super(CNNbLSTM, self).__init__()

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])

        log.info(f'<init>: \n{self}')

    def forward(self, x, mask) -> list:
        """ Forwarding logic """

        max_length = x.size(1)
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, dis, rsa, phi, psi]


class CNNbLSTM_ESM2_SecondaryStructure(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, language_model, **kwargs):
        """ Baseline model with ESM1b and only secondary structure predictions
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super(CNNbLSTM_ESM2_SecondaryStructure, self).__init__()

        self.embedding = ESM2Embedding(language_model, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
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

    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """

        max_length = x.size(1)

        x = self.embedding(x, max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]


class CNNbLSTM_ESM2_Complete(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, n_head: int, encoder_layers: int, embedding_args: dict = None, embedding_pretrained: str = None, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super(CNNbLSTM_ESM2_Complete, self).__init__()

        # Added parameters
        self.init_n_channels = init_n_channels
        self.out_channels = out_channels

        # ESM1b block
        self.embedding = ESM2Embedding(embedding_pretrained=embedding_pretrained, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * cnn_layers))

        # LSTM block
        self.lstm = nn.ModuleList()
        for i in range (lstm_layers):
            input_size = init_n_channels + (out_channels * cnn_layers) if i == 0 else 2*n_hidden
            self.lstm.append(nn.LSTM(input_size=input_size, hidden_size=n_hidden, batch_first=True,
                            num_layers=1, bidirectional=True, dropout=dropout))
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        """# LSTM block 2
        self.lstm2 = nn.LSTM(input_size=out_channels, hidden_size=int(n_hidden/4), batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer2 = nn.Dropout(p=dropout)

        # LSTM block 3
        self.lstm3 = nn.LSTM(input_size=out_channels, hidden_size=int(n_hidden/4), batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer3 = nn.Dropout(p=dropout)

        # LSTM block 4
        self.lstm4 = nn.LSTM(input_size=out_channels, hidden_size=int(n_hidden/4), batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer4 = nn.Dropout(p=dropout)

        # LSTM block 5
        self.lstm5 = nn.LSTM(input_size=out_channels, hidden_size=int(n_hidden/4), batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer5 = nn.Dropout(p=dropout)"""

        # Transformer block
        """self.pos_enc = PositionalEncoding(init_n_channels + (out_channels * cnn_layers))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=init_n_channels + (out_channels * cnn_layers), nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_layers)"""

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=2),
            nn.Tanh()
        ])
        self.tasa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=1),
        ])
        self.thsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=1),
        ])
        self.lhp = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=1),
        ])
        self.hp_loc = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=2),
        ])
        self.lhp_loc = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=2),
        ])
        self.species = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=10),
        ])
        self.expression = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 4 + init_n_channels + (out_channels * cnn_layers), out_features=10),
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True) -> list:
        """ Returns the parameters to learn """
        if print:
            log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                if print:
                    log.info("\t" + name)
                yield param

    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """
        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))     
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for idx, layer in enumerate(self.conv):
            r = torch.cat([r, layer(x)], dim=1)
        x = self.batch_norm(r)

        #Decompose
        """x_bas=x[:, [*range(0,self.init_n_channels)]]
        x_short=x[:, [*range(self.init_n_channels, self.init_n_channels + 1 * self.out_channels)]]
        x_mid=x[:, [*range(self.init_n_channels + 1 * self.out_channels, self.init_n_channels + 2 * self.out_channels)]]
        x_long=x[:, [*range(self.init_n_channels + 2 * self.out_channels, self.init_n_channels + 3 * self.out_channels)]]
        x_long_long=x[:, [*range(self.init_n_channels + 3 * self.out_channels, self.init_n_channels + 4 * self.out_channels)]]
        del x"""

        # LSTM block
        x = x.permute(0, 2, 1)


        r = x
        for lstm in self.lstm:
            x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
            x, _ = lstm(x)
            x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
            x = self.lstm_dropout_layer(x)
            r = torch.cat([r, x], dim=2)

        """x_bas = x_bas.permute(0, 2, 1)
        x_short = x_short.permute(0, 2, 1)
        x_mid = x_mid.permute(0, 2, 1)
        x_long = x_long.permute(0, 2, 1)
        x_long_long = x_long_long.permute(0, 2, 1)

        x_bas = pack_padded_sequence(x_bas, mask, batch_first=True, enforce_sorted=False)
        x_bas, _ = self.lstm(x_bas)
        x_bas, _ = pad_packed_sequence(x_bas, total_length=max_length, batch_first=True)
        x_bas = self.lstm_dropout_layer(x_bas)

        x_short = pack_padded_sequence(x_short, mask, batch_first=True, enforce_sorted=False)
        x_short, _ = self.lstm2(x_short)
        x_short, _ = pad_packed_sequence(x_short, total_length=max_length, batch_first=True)
        x_short = self.lstm_dropout_layer2(x_short)

        x_mid = pack_padded_sequence(x_mid, mask, batch_first=True, enforce_sorted=False)
        x_mid, _ = self.lstm3(x_mid)
        x_mid, _ = pad_packed_sequence(x_mid, total_length=max_length, batch_first=True)
        x_mid = self.lstm_dropout_layer3(x_mid)

        x_long = pack_padded_sequence(x_long, mask, batch_first=True, enforce_sorted=False)
        x_long, _ = self.lstm4(x_long)
        x_long, _ = pad_packed_sequence(x_long, total_length=max_length, batch_first=True)
        x_long = self.lstm_dropout_layer4(x_long)

        x_long_long = pack_padded_sequence(x_long_long, mask, batch_first=True, enforce_sorted=False)
        x_long_long, _ = self.lstm5(x_long_long)
        x_long_long, _ = pad_packed_sequence(x_long_long, total_length=max_length, batch_first=True)
        x_long_long = self.lstm_dropout_layer5(x_long_long)"""

        # Transformer block
        """x_long = self.pos_enc(x_long)
        x_long = self.transformer_encoder(x_long)

        x = torch.cat([x_bas, x_short, x_mid, x_long, x_long_long], dim=2)"""

        # hidden neurons to classes
        ss8 = self.ss8(r)
        ss3 = self.ss3(r)
        dis = self.disorder(r)
        rsa = self.rsa(r)
        phi = self.phi(r)
        psi = self.psi(r)
        tasa = self.tasa(r)
        thsa = self.thsa(r)
        lhp = self.lhp(r)
        hp_loc = self.hp_loc(r)
        lhp_loc = self.lhp_loc(r)
        species = self.species(r)
        expression = self.expression(r)

        return [ss8, ss3, dis, rsa, phi, psi, tasa, thsa, lhp, hp_loc, lhp_loc, species, expression]


class CNNbLSTM_Extended(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ CNNbLSTM with removed Q3 prediction which instead is predicted by remapping Q8
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super(CNNbLSTM_Extended, self).__init__()

        # ESM1b block
        self.embedding = ESM2Embedding(embedding_args, embedding_pretrained, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
        ])
        self.rsa_iso = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.rsa_cpx = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True, print: bool = True) -> list:
        """ Returns the parameters to learn """
        if print:
            log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                if print:
                    log.info("\t" + name)
                yield param

    def forward(self, x, mask) -> list:
        """ Forwarding logic """

        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        dis = self.disorder(x)
        rsa_iso = self.rsa_iso(x)
        rsa_cpx = self.rsa_iso(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, dis, rsa_iso, rsa_cpx, phi, psi]