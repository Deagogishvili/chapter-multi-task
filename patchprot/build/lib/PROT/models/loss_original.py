import torch
import torch.nn as nn

from PROT.models.metric import dihedral_to_radians, arctan_dihedral, get_mask, get_mask_glob
from PROT.utils import setup_logger

log=setup_logger(__name__)
not_passed=True

class RevisedUncertainty(nn.Module):
    """ Automatically weighted multi task loss created in Kendall paper (2018) and revised in Auxiliary task paper
    Args:
        nums: list, initial params for each loss
        loss: multi-task loss
    """
    def __init__(self,nums):
        super(RevisedUncertainty, self).__init__()
        self.params=nn.Parameter(torch.ones(nums))

    def forward(self, *x):
        loss_sum = 0
        #print(torch.stack([*x]).tolist())
        for i, loss in enumerate(x):
            if loss == 0:
                loss_sum += 0
            else:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                #print("Added_value :",0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
        return loss_sum
    
def mse(outputs: torch.tensor, labels: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """ Returns mean squared loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    loss = torch.square(outputs - labels) * mask
    loss = torch.sum(loss) / torch.sum(mask)

    return torch.nan_to_num(loss)


def cross_entropy(outputs: torch.tensor, labels: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """ Returns cross entropy loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """

    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)*mask
    loss = torch.sum(loss) / torch.sum(mask)
    
    return torch.nan_to_num(loss)

def ss8(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns SS8 loss
    Args:
        outputs: tensor with SS8 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 7:15], dim=2)
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def ss3(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns SS3 loss
    Args:
        outputs: tensor with SS3 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    # convert ss8 to ss3 class
    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)
    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def disorder(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns disorder loss
    Args:
        outputs: tensor with disorder predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1.0 - labels], dim=2), dim=2)

    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def rsa(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 5].unsqueeze(2)

    return mse(outputs, labels, mask)

def rsa_iso(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 5].unsqueeze(2)

    return mse(outputs, labels, mask)

def rsa_cpx(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 6].unsqueeze(2)

    return mse(outputs, labels, mask)

def phi(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns phi loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 15].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(
        labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def psi(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns psi loss
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 16].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(
        labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss

def tasa(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns tasa loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,18)
    mask = mask.unsqueeze(1)
    mask_loc = get_mask(labels, use_evaluation_mask=False)

    labels = labels[:, 0, 18].unsqueeze(1)
    outputs = torch.sum(outputs * mask_loc.unsqueeze(2),axis=1)
    
    return mse(outputs, labels, mask)

def thsa(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns thsa loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,19)
    mask = mask.unsqueeze(1)
    mask_loc = get_mask(labels, use_evaluation_mask=False)

    labels = labels[:, 0, 19].unsqueeze(1)
    outputs = torch.sum(outputs * mask_loc.unsqueeze(2),axis=1)

    return mse(outputs, labels, mask)

def lhp(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns largest hydrophobic patch loss
    Args:
        outputs: tensor with lhp predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,21)
    mask = mask.unsqueeze(1)
    mask_loc = get_mask(labels, use_evaluation_mask=False)

    labels = labels[:, 0, 21].unsqueeze(1)
    outputs = torch.sum(outputs * mask_loc.unsqueeze(2),axis=1)

    return mse(outputs, labels, mask)

def hp_loc(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns largest hydrophobic patch loss
    Args:
        outputs: tensor with lhp predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,22)
    mask_loc = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)[mask == 1]

    labels = labels[:, :, 22].unsqueeze(2)[mask == 1]
    labels = torch.argmax(torch.cat([1.0 - labels, labels], dim=2), dim=2)
    outputs = outputs[mask == 1]
    outputs = outputs.permute(0, 2, 1)   

    return cross_entropy(outputs, labels, mask_loc)

def lhp_loc(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns largest hydrophobic patch loss
    Args:
        outputs: tensor with lhp predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,23)
    mask_loc = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)[mask == 1]

    labels = labels[:, :, 23].unsqueeze(2)[mask == 1]
    labels = torch.argmax(torch.cat([1.0 - labels, labels], dim=2), dim=2)
    outputs = outputs[mask == 1]
    outputs = outputs.permute(0, 2, 1)   
    
    return cross_entropy(outputs, labels, mask_loc)

def species(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns species loss
    Args:
        outputs: tensor with species predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,24,10)
    mask = mask.unsqueeze(1)
    mask_loc = get_mask(labels, use_evaluation_mask=False)

    labels = torch.argmax(labels[:, 0, 24:34], dim=1)
    outputs = torch.sum(outputs * mask_loc.unsqueeze(2),axis=1)
    
    return cross_entropy(outputs, labels, mask)

def expression(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns expression loss
    Args:
        outputs: tensor with expression predictions
        labels: tensor with labels
    """
    mask = get_mask_glob(labels,35,10)
    mask = mask.unsqueeze(1)
    mask_loc = get_mask(labels, use_evaluation_mask=False)

    labels = torch.argmax(labels[:, 0, 35:45], dim=1)
    outputs = torch.sum(outputs * mask_loc.unsqueeze(2),axis=1)
    
    return cross_entropy(outputs, labels, mask)

def multi_task_loss(outputs: torch.tensor, labels: torch.tensor, method: nn.Module, not_passed: bool) -> torch.tensor:
    """ Returns a weighted multi task loss. 
        Combines ss8, ss3, disorder, rsa, phi and psi loss.
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    ss8_w = 1
    ss3_w = 5
    dis_w = 5
    rsa_w = 100
    phi_w = 5
    psi_w = 5
    tasa_w = 0.0000001
    thsa_w = 0.000002
    lhp_w = 0.000005
    hp_loc_w = 5
    lhp_loc_w = 5
    #Old 0.004
    species_w = 0.04
    #Old 0.006
    expression_w = 0.06

    if not_passed:
        not_passed=False
        print("Multi Task Loss")
        print(f"SS8 : {ss8_w} // SS3: {ss3_w} // DIS: {dis_w} // RSA: {rsa_w} // PHI: {phi_w} // PSI: {psi_w} // TASA: {tasa_w} // THSA: {thsa_w} // LHP: {lhp_w} // HP LOC: {hp_loc_w} // LHP LOC: {lhp_loc_w} // SPECIES: {species_w} // EXPRESSION: {expression_w}")
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * ss8_w
    _ss3 = ss3(outputs[1], labels) * ss3_w
    _dis = disorder(outputs[2], labels) * dis_w
    _rsa = rsa(outputs[3], labels) * rsa_w
    _phi = phi(outputs[4], labels) * phi_w
    _psi = psi(outputs[5], labels) * psi_w 
    _tasa = tasa(outputs[6], labels) * tasa_w
    _thsa = thsa(outputs[7], labels) * thsa_w
    _lhp = lhp(outputs[8], labels) * lhp_w
    _hp_loc = hp_loc(outputs[9], labels) * hp_loc_w
    _lhp_loc = hp_loc(outputs[10], labels) * lhp_loc_w
    _species = species(outputs[11], labels) * species_w
    _expression = expression(outputs[12], labels) * expression_w

    loss = method(_ss8, _ss3, _dis, _rsa, _phi, _psi, _tasa, _thsa, _lhp, _hp_loc, _lhp_loc, _species, _expression)
    return loss, not_passed

def multi_task_loss_basic(outputs: torch.tensor, labels: torch.tensor, method: nn.Module, not_passed: bool) -> torch.tensor:
    """ Returns a weighted multi task loss. 
        Combines ss8, ss3, disorder, rsa, phi and psi loss.
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    ss8_w = 1
    ss3_w = 5
    dis_w = 5
    rsa_w = 100
    phi_w = 5
    psi_w = 5
    tasa_w = 0.00000003
    thsa_w = 0.00000045
    lhp_w = 0.000001
    hp_loc_w = 5
    lhp_loc_w = 5
    species_w = 5
    expression_w = 0.05

    if not_passed:
        not_passed=False
        print("Multi Task Loss")
        print(f"SS8 : {ss8_w} // SS3: {ss3_w} // DIS: {dis_w} // RSA: {rsa_w} // PHI: {phi_w} // PSI: {psi_w} // TASA: {tasa_w} // THSA: {thsa_w} // LHP: {lhp_w} // HP LOC: {hp_loc_w} // LHP LOC: {lhp_loc_w} // SPECIES: {species_w} // EXPRESSION: {expression_w}")
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * ss8_w
    _ss3 = ss3(outputs[1], labels) * ss3_w
    _dis = disorder(outputs[2], labels) * dis_w
    _rsa = rsa(outputs[3], labels) * rsa_w
    _phi = phi(outputs[4], labels) * phi_w
    _psi = psi(outputs[5], labels) * psi_w 
    _tasa = tasa(outputs[6], labels) * tasa_w
    _thsa = thsa(outputs[7], labels) * thsa_w
    _lhp = lhp(outputs[8], labels) * lhp_w
    _hp_loc = hp_loc(outputs[9], labels) * hp_loc_w
    _lhp_loc = hp_loc(outputs[10], labels) * lhp_loc_w
    _species = hp_loc(outputs[11], labels) * species_w
    _expression = expression(outputs[12], labels) * expression_w

    loss = torch.stack([_ss8, _ss3, _dis, _rsa, _phi, _psi, _tasa, _thsa, _lhp, _hp_loc, _lhp_loc, _species, _expression])
    return loss.sum(), not_passed


def multi_task_extended(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted multi task loss. 
        Combines ss8, ss3, disorder, rsa_iso, rsa_cpx, phi and psi loss.
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _dis = disorder(outputs[1], labels) * 5
    _rsa_iso = rsa_iso(outputs[2], labels) * 100
    _rsa_cpx = rsa_cpx(outputs[3], labels) * 100
    _phi = phi(outputs[4], labels) * 5
    _psi = psi(outputs[5], labels) * 5

    loss = torch.stack([_ss8, _dis, _rsa_iso, _rsa_cpx, _phi, _psi])

    return loss.sum()


def secondary_structure_loss(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted double task loss for secondary structure. 
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _ss3 = ss3(outputs[1], labels) * 5

    loss = torch.stack([_ss8, _ss3])

    return loss.sum()
