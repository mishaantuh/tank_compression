from torch import nn


class Criterion:
    def __init__(self):
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.BCE = nn.BCEWithLogitsLoss()


class Logger:
    """
    model: g - generator, d - discriminator
    """
    def __init__(self, model="g"):
        self.rec_loss = []
        self.adv_loss = []
        self.model = model

    def reset(self):
        self.rec_loss = []
        self.adv_loss = []

    def add(self, adv, rec=None):
        self.adv_loss.append(adv)
        if rec is not None:
            self.rec_loss.append(rec)

    def mean_last_steps(self, n_steps):
        mean_adv = sum(self.adv_loss[-n_steps:]) / n_steps
        if self.model == "g":
            mean_rec = sum(self.rec_loss[-n_steps:]) / n_steps
            return mean_adv, mean_rec
        return mean_adv


def get_log_train(logger_g, logger_d, n_steps):
    adv_d = logger_d.mean_last_steps(n_steps)
    adv_g, rec_g = logger_g.mean_last_steps(n_steps)

    return {
        "train adv loss discriminator": adv_d,

        "train adv loss generator": adv_g,
        "train rec loss generator": rec_g,
    }


def get_log_test(logger_g):
    adv_g, rec_g = logger_g.mean_last_steps(1)

    return {
        "test adv loss generator": adv_g,
        "test rec loss generator": rec_g,
    }
