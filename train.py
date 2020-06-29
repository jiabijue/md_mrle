import os
import sys
import pickle
import logging
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import config as cfg
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':

    # ---------- Arguments and configurations ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'gru', 'cnn-gru'],
                        help='setting of the experiment to use')
    parser.add_argument('--fold', type=int, help='which fold is this time')
    parser.add_argument('--round', type=int, help='which round is this time')
    args = parser.parse_args()

    settings = utils.load_json('settings.json')
    setting = settings[args.model]

    result_file = 'save/results-{}.csv'.format(args.model)
    epoch_stop_at = 0

    exp = 'fold{}_round{}'.format(args.fold, args.round)
    save_dir = 'save/{}/{}'.format(args.model, exp)
    os.makedirs(save_dir, exist_ok=True)

    log_file = '{}/log.txt'.format(save_dir)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, mode='w'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/{}/{}'.format(args.model, exp))

    # ---------- Prepare data ----------
    logger.info(">> Prepare data ...")
    train_names, val_names, test_names = [], None, None
    test_split = args.fold
    val_split = (args.fold + 1) % 10
    for i in range(10):
        names = utils.get_split_list(i)
        if i == test_split:
            test_names = names
        elif i == val_split:
            val_names = names
        else:
            train_names += names

    if not os.path.exists(cfg.MEAN_STD_PATH):
        os.makedirs(cfg.MEAN_STD_PATH)
    mean_file = os.path.join(cfg.MEAN_STD_PATH, 'mean_fold{}.npy'.format(args.fold))
    std_file = os.path.join(cfg.MEAN_STD_PATH, 'std_fold{}.npy'.format(args.fold))

    if not os.path.isfile(mean_file) or not os.path.isfile(std_file):
        logger.info("creating mean.npy and std.npy ...")
        utils.save_mean_std(train_names, mean_file, std_file)

    trainset = utils.OpenBMAT(train_names, mean_file, std_file)
    valset = utils.OpenBMAT(val_names, mean_file, std_file)
    testset = utils.OpenBMAT(test_names, mean_file, std_file)

    train_loader = DataLoader(trainset, setting["batch_size"], shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(valset, setting["batch_size"])
    test_loader = DataLoader(testset, 1)

    # ---------- Build model ----------
    model = utils.build_model(setting["model"])
    model = model.to(device)

    # ---------- Train ----------
    logger.info(">> Training starts ...")
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), setting["lr"])

    try:
        for epoch in range(setting["n_epochs"]):
            epoch_stop_at = epoch
            model.train()

            total_loss_train = 0.
            for batch, (x, y1, y2) in enumerate(train_loader):
                x = x.transpose(0, 1).to(device)   # [T, B, S]
                y = y2.transpose(0, 1).to(device)

                y_ = model(x)
                loss = criterion(y_, y)
                total_loss_train += loss

                model.zero_grad()
                loss.backward()
                opt.step()

                if (batch + 1) % 5 == 0:
                    logger.info("Epoch %d Batch %d | loss: %.4f" % (epoch, batch, loss.item()))

            avg_loss_train = total_loss_train / (batch + 1)
            writer.add_scalar('loss/train', avg_loss_train, epoch)

            # ---------- Validate ----------
            best_val_loss = None
            with torch.no_grad():
                model.eval()

                total_loss_val = 0.
                for i, (x, y1, y2) in enumerate(val_loader):
                    x = x.transpose(0, 1).to(device)
                    y = y2.transpose(0, 1).to(device)

                    y_ = model(x)
                    loss_val = criterion(y_, y)
                    total_loss_val += loss_val
                avg_loss_val = total_loss_val / (i + 1)
                logger.info("validation loss: %.4f\n" % (avg_loss_val.item()))
                writer.add_scalar('loss/val', avg_loss_val, epoch)

                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or avg_loss_val < best_val_loss:
                    torch.save(model, os.path.join(save_dir, 'model.pt'))

    except KeyboardInterrupt:
        logger.info("\nTraining stops due to keyboard interrupt.")
        sys.exit()

    # ---------- Test ----------
    finally:
        logger.info(">> Test result:")
        model = torch.load(os.path.join(save_dir, 'model.pt'))
        model = model.to(device)

        with torch.no_grad():
            model.eval()

            total_loss_test = 0.
            predictions1, ground_truths1, predictions2, ground_truths2 = [], [], [], []
            for i, (x, y1, y2) in enumerate(test_loader):
                x = x.transpose(0, 1).to(device)
                y2 = y2.transpose(0, 1).to(device)

                y2_ = model(x)
                loss_test = criterion(y2_, y2)
                total_loss_test += loss_test

                gt2 = y2.long().squeeze().T.cpu().numpy()
                ground_truths2.append(gt2)
                gt1 = utils.mrle_label_to_mud_label(gt2)
                ground_truths1.append(gt1)

                p2 = y2_.squeeze().T.cpu().numpy()
                p2 = utils.apply_threshold2(p2)
                predictions2.append(p2)
                p1 = utils.mrle_label_to_mud_label(p2)
                predictions1.append(p1)

            mud_p_events, mud_gt_events, mrle_p_events, mrle_gt_events = [], [], [], []
            for p1, gt1, p2, gt2 in zip(predictions1, ground_truths1, predictions2, ground_truths2):
                mud_p_events.append(utils.label_to_annotation(p1, 'mud'))
                mud_gt_events.append(utils.label_to_annotation(gt1, 'mud'))
                mrle_p_events.append(utils.label_to_annotation(p2, 'mrle'))
                mrle_gt_events.append(utils.label_to_annotation(gt2, 'mrle'))

            # save events
            events_file = open(os.path.join(save_dir, 'events.pkl'), 'wb')
            pickle.dump((mud_p_events, mud_gt_events, mrle_p_events, mrle_gt_events), events_file)
            events_file.close()

            mud_s_metrics, mud_e_metrics = \
                utils.eval(mud_gt_events, mud_p_events, segment_length=0.01, event_tolerance=0.5)
            mrle_s_metrics, mrle_e_metrics = \
                utils.eval(mrle_gt_events, mrle_p_events, segment_length=0.01, event_tolerance=0.5)

            avg_loss_test = total_loss_test / (len(testset))
            logger.info("test loss: %.4f\n" % (avg_loss_test.item()))

            utils.log(result_file, exp, setting, epoch_stop_at, avg_loss_test.item(),
                      mud_s_metrics, mud_e_metrics, mrle_s_metrics, mrle_e_metrics)

            logger.info("Test done.")

    writer.close()
