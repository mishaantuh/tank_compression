import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader
from model import Generator, Discriminator
from loader import DatasetLoader
from sklearn.model_selection import train_test_split
from .helper import Logger, Criterion, get_log_train, get_log_test


def fit(cfg):
    gen = Generator(n_layers=5).to(cfg["device"])
    dis = Discriminator().to(cfg["device"])

    opt_gen = torch.optim.Adam(gen.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_dis = torch.optim.Adam(dis.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    criterion = Criterion()

    images = np.loadtxt(cfg["attr_path"], skiprows=1, usecols=[0], dtype=np.str, delimiter=',')
    indexes_train, indexes_test = train_test_split(np.arange(0, len(images) - 1), random_state=42, test_size=0.1)
    train_img = images[indexes_train]
    test_img = images[indexes_test]

    loader_train = DataLoader(dataset=DatasetLoader(cfg["image_path"], train_img),
                              batch_size=cfg["batch_size"], shuffle=True, num_workers=5)

    loader_test = DataLoader(dataset=DatasetLoader(cfg["image_path"], test_img),
                             batch_size=cfg["batch_size_test"], shuffle=True, num_workers=1)

    print("Generator: {}, Discriminator: {}".format(
        sum(p.numel() for p in gen.parameters()), sum(p.numel() for p in dis.parameters())))

    logger_g = Logger("g")
    logger_d = Logger("d")
    logger_g_test = Logger("g")

    for i in range(cfg["num_epoch"]):
        if cfg["wandb_enable"]:
            wandb.log({"epoch": i + 1})

        logger_d.reset()
        logger_d.reset()

        if (i + 1) % 5 == 0:
            torch.save(gen.state_dict(), "checkpoints/{}/epoch_{}".format(cfg["name_run"], i+1))

        for j, img in enumerate(loader_train):
            img = img.to(cfg["device"])

            # <--- discriminator fit ---> #
            if (j + 1) % 5 != 0:
                fake_img = gen(img, mode="encode-decode")

                adv_real = dis(img)
                adv_fake = dis(fake_img)

                d_loss_adv = criterion.MSE(adv_real, torch.ones_like(adv_real)) + criterion.MSE(adv_fake,
                                                                                                torch.zeros_like(
                                                                                                    adv_fake))

                d_loss = cfg["lambda_3"] * d_loss_adv

                opt_dis.zero_grad()
                d_loss.backward()
                opt_dis.step()

                logger_d.add(d_loss_adv.item())

            # <--- generator fit ---> #
            else:
                opt_gen.zero_grad()
                h = gen(img, mode="encode")
                fake_img = gen(h, mode="decode")
                real_img = gen(h, mode="decode")

                adv_fake = dis(fake_img)

                g_loss_adv = criterion.MSE(adv_fake, torch.ones_like(adv_fake))
                g_loss_rec = criterion.MSE(real_img, img)

                g_loss = cfg["lambda_1"] * g_loss_rec + cfg["lambda_3"] * g_loss_adv

                g_loss.backward()
                opt_gen.step()

                logger_g.add(g_loss_adv.item(), g_loss_rec.item())

            # TODO: дополнить тестирование
            if (j + 1) % (len(loader_train) // cfg["logs_per_epoch"] - 1) == 0:
                wandb.log(get_log_train(logger_g, logger_d, n_steps=10))
