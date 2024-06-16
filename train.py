import os
import torch
from torch.utils.data import DataLoader
import time
import numpy as np

import config
from model import REAUNet
from loss import BCEDiceLoss
from dataset import ParcelDataset
from metric import calculate_accuracy


def run():
    args = config.get_args()

    model = REAUNet(args.in_channels, args.num_class)
    model = model.to(args.device)

    loss_func = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma,
                                                           patience=args.lr_step, verbose=True)

    train_set = ParcelDataset(args.txt_path_train, args.image_root, args.label_root, True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_set = ParcelDataset(args.txt_path_val, args.image_root, args.label_root, False)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    print('=' * 20, 'Data', '=' * 20)
    print('Train set:', len(train_set), '  Valid set:', len(valid_set), '  Batch size:', args.batch_size)

    # checkpoint save folder
    if not os.path.exists(args.checkpoint_folder):
        os.mkdir(args.checkpoint_folder)
    current_time = time.localtime()
    time_str = time.strftime("%Y%m%d%H%M", current_time)
    model_save_folder = os.path.join(args.checkpoint_folder, time_str)
    if not os.path.exists(model_save_folder):
        os.mkdir(model_save_folder)
    print('model checkpoint folder:', model_save_folder)

    print('=' * 20, 'Train', '=' * 20)
    early_stop_count = 0
    best_loss = float('inf')

    for epoch in range(args.max_epochs):
        # ----- train -----
        model.train()
        train_loss = []
        start_time = time.perf_counter()

        for idx, data in enumerate(train_loader):
            image, label = data
            image, label = image.to(args.device), label.to(args.device)

            optimizer.zero_grad()
            results = model(image)
            if isinstance(results, list):
                loss = torch.zeros(1).to(args.device)
                for k in range(len(results) - 1):
                    loss += (0.1 * k + 0.2) * loss_func(results[k], label)
                loss += loss_func(results[-1], label)
            else:
                loss = loss_func(results, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        training_loss = np.mean(train_loss)
        print("\r" + f"[ Train | <{epoch + 1:d}/{args.max_epochs:d}> time:{time.perf_counter() - start_time:.2f}s | "
                     f"train_loss={training_loss:.6f} ]", end='  ')

        # ----- valid -----
        model.eval()
        valid_loss, valid_OA, valid_F1, valid_IOU = [], [], [], []
        start_time = time.perf_counter()

        with torch.no_grad():
            for data in valid_loader:
                image, label = data
                image, label = image.to(args.device), label.to(args.device)

                results = model(image)
                if isinstance(results, list):
                    loss = torch.zeros(1).to(args.device)
                    for k in range(len(results) - 1):
                        loss += (0.1 * k + 0.2) * loss_func(results[k], label)
                    loss += loss_func(results[-1], label)
                else:
                    loss = loss_func(results, label)
                valid_loss.append(loss.item())

                # final output
                if isinstance(results, list):
                    results = results[-1]

                # device and numpy
                if results.device != 'cpu':
                    result_fuse = results.cpu()
                else:
                    result_fuse = results
                result_fuse = result_fuse.squeeze(1).numpy()
                result_fuse = np.where(result_fuse > args.threshold, 1, 0)
                if label.device != 'cpu':
                    label_output = label.cpu()
                else:
                    label_output = label
                label_output = label_output.squeeze(1).numpy()

                OA, F1, IOU = calculate_accuracy(result_fuse, label_output)
                valid_OA.append(OA)
                valid_F1.append(F1)
                valid_IOU.append(IOU)

        validation_loss = np.mean(valid_loss)
        print(f"[ Valid | time:{time.perf_counter() - start_time:.2f}s | val_loss = {validation_loss:.6f} "
              f"({np.mean(valid_OA):.4f}, {np.mean(valid_F1):.4f}, {np.mean(valid_IOU):.4f}) ]")
        scheduler.step(validation_loss)  # learning rate scheduler

        # ----- save model -----
        if best_loss > validation_loss:
            best_loss = validation_loss
            model_save_best = os.path.join(model_save_folder, 'model_best.pth')
            torch.save(model.state_dict(), model_save_best)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if (epoch + 1) % args.save_epoch == 0:
            model_save_epoch = os.path.join(model_save_folder, 'model_epoch_%d.pth' % (epoch + 1))
            torch.save(model.state_dict(), model_save_epoch)

        if early_stop_count >= args.early_stop:
            print(f"Early stop at Epoch <{epoch + 1:03d}>, early stop count <{early_stop_count:02d}>.")
            break

    model_save_last = os.path.join(model_save_folder, 'model_last.pth')
    torch.save(model.state_dict(), model_save_last)


if __name__ == '__main__':
    run()