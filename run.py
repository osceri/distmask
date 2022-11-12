import os
import torch
from utils.utils import Logger
import argparse
import glob
import regex

from models import LSTM
from dataset import get_dataset_test, get_dataset_train, get_dataset_train_student
from engine import evaluate, train_one_epoch, train_one_epoch_student

def get_args_parser():
    parser = argparse.ArgumentParser('Openmic18 classifier', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    parser.add_argument('--device', default='cpu', type=str)

    parser.add_argument('--data_path_train', default='/home/oscar/openmic18/data/mp3/openmic_train.csv_mp3.hdf', type=str)
    parser.add_argument('--data_length_train', default=-1, type=int)
    parser.add_argument('--data_path_test', default='/home/oscar/openmic18/data/mp3/openmic_test.csv_mp3.hdf', type=str)
    parser.add_argument('--data_length_test', default=-1, type=int)
    parser.add_argument('--data_path_train_student', default='/home/oscar/openmic18/data/mp3/openmic_train_student.csv_mp3.hdf', type=str)
    parser.add_argument('--data_length_train_student', default=-1, type=int)

    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--log_dir', default='./log_dir', type=str)
    parser.add_argument('--resume_dir', default='./resume_dir', type=str)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--student', action='store_true')

    return parser
    
def main(args):
    logger = Logger()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger(f'Created {args.output_dir}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        logger(f'Created {args.log_dir}')

    if args.resume_dir and not args.resume:
        path_index = []

        for path in glob.glob(f'{args.resume_dir}/*.pth'):
            index = -1
            for index in regex.findall(r'-([\d]+).pth', path):
                break
            index = int(index)
            if -1 < index:
                path_index.append((path, index))

        if path_index:
            args.resume = list(map(lambda x: x[0], sorted(path_index, key=lambda x: x[1])))[-1]
            logger(f"Found checkpoint at {args.resume}")


    logger("Loading datasets")
    if args.student:
        dataset_train_student = get_dataset_train_student(args.data_path_train_student, length=args.data_length_train_student)
    else:
        dataset_train = get_dataset_train(args.data_path_train, length=args.data_length_train)
    dataset_test = get_dataset_test(args.data_path_test, length=args.data_length_test)

    logger("Creating data samplers")
    if args.student:
        sampler_train_student = torch.utils.data.SequentialSampler(dataset_train_student)
    else:
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    logger("Creating data loaders")
    if args.student:
        data_loader_train_student = torch.utils.data.DataLoader(
            dataset_train_student, sampler=sampler_train_student,
            batch_size=args.batch_size, drop_last=True
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, drop_last=True
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, drop_last=True
    )

    logger(f"Initializing device {args.device}")
    device = torch.device(args.device)
    torch.set_num_threads(8)

    logger("Initializing model")

    model = LSTM(
            input_dim = 320000,
            embedding_dim = 1000,
            hidden_dim = 10,
            output_dim = 20,
            n_layers = 10,
            bidirectional = True,
            dropout = 0.5,
            batch_size = args.batch_size,
            device = device)

    logger("Model initialized")

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)

    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.state_dict = checkpoint
        logger(f"Model loaded with checkpoint {args.resume}")
    
    if args.eval:
        stats = evaluate(model, data_loader_test, optimizer, device, 0, logger=logger)

    else:
        for epoch in range(args.epochs):
            if args.student:
                train_loss = train_one_epoch_student(model, data_loader_train_student, optimizer, device, epoch, logger=logger)
            else:
                train_loss = train_one_epoch(model, data_loader_train, optimizer, device, epoch, logger=logger)
            test_loss, mAP = evaluate(model, data_loader_train, optimizer, device, epoch, logger=logger)
            logger(f'Epoch {epoch} : train loss {train_loss}, test loss {test_loss}, mAP {mAP}')

            if (epoch % 5) == 0:
                logger("Saving model")
                torch.save(model.state_dict, f'{args.output_dir}/lstm-{epoch}.pth')

    logger("Done")


if "__main__" == __name__:
    args  = get_args_parser()
    args = args.parse_args()
    main(args)
