import os
import argparse
from os import path

from experiment import Experiment

def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=24)
    parser.add_argument('--encode_init_state', type=str, default='NL', choices=[False, 'raw', 'NL'])
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--use_state_loss', default=False, action="store_true")
    parser.add_argument('--rap_prob', default=0.25, type=float)
    parser.add_argument('--base_model_dir', type=str, default='models')
    args = parser.parse_args()

    model_dir_str = "epochs_" + str(args.epochs)
    model_dir_str += "_patience_" + str(args.patience)

    if args.use_state_loss:
        model_dir_str += "_state"
        model_dir_str += f"_{args.rap_prob}"

    args.model_dir = path.join(args.base_model_dir, model_dir_str)
    args.best_model_dir = path.join(args.model_dir, "best")

    if not path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not path.exists(args.best_model_dir):
        os.makedirs(args.best_model_dir)

    args.model_path = path.join(args.model_dir, "model.pt")
    args.best_model_path = path.join(args.best_model_dir, "model.pt")

    Experiment(args)


if __name__ == '__main__':
    main()