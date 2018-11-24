# Sebastian Raschka 2018
# mytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch


def continuous_jaccard(x, y):
    """
    Implementation of the continuous version of the
    Jaccard distance:

    1 - [sum_i min(x_i, y_i)] / [sum_i max(x_i, y_i)]
    """
    c = torch.cat((x.view(-1).unsqueeze(1), y.view(-1).unsqueeze(1)), dim=1)

    numerator = torch.sum(torch.min(c, dim=1)[0])
    denominator = torch.sum(torch.max(c, dim=1)[0])

    return 1. - numerator/denominator
