import os
import argparse
import wandb
from os import path

from experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=24)
    parser.add_argument('--encode_init_state', type=str, default='NL', choices=[False, 'raw', 'NL'])
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_train', type=int, default=None)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--base_model_dir', type=str, default='models')
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--use_wandb', default=False, action="store_true")

    parser.add_argument('--rap_prob', default=0.0, type=float)
    parser.add_argument('--add_state', choices=['all', 'targeted', 'random'], type=str, default='targeted')
    parser.add_argument('--state_repr', default="text", choices=["text", "raw"], type=str)

    args = parser.parse_args()

    model_dir_str = "gen_state_"
    model_dir_str += "size_" + str(args.model_size)
    model_dir_str += "_epochs_" + str(args.epochs)
    model_dir_str += "_patience_" + str(args.patience)

    if args.rap_prob:
        model_dir_str += "_state"
        model_dir_str += f"_{args.rap_prob}"
        model_dir_str += f"_{args.add_state}"
        model_dir_str += f"_{args.state_repr}"

    if args.num_train is not None:
        model_dir_str += f"_num_train_{args.num_train}"

    model_dir_str += f"_seed_{args.seed}"

    args.model_dir = path.join(args.base_model_dir, model_dir_str)
    args.best_model_dir = path.join(args.model_dir, "best")

    # Set log dir
    wandb_dir = path.join(args.model_dir, "wandb")
    os.environ['WANDB_DIR'] = wandb_dir

    if not path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not path.exists(args.best_model_dir):
        os.makedirs(args.best_model_dir)

    if not path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    args.model_path = path.join(args.model_dir, "model.pt")
    args.best_model_path = path.join(args.best_model_dir, "model.pt")

    if args.use_wandb:
        wandb.init(
            id=model_dir_str, project="state-probing", resume=True,
            notes="State probing", tags="november", config={},
        )
        wandb.config.update(args)

    Experiment(args)


if __name__ == '__main__':
    main()