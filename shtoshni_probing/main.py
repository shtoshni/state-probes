import os
import argparse
from os import path

from collections import itertools
# from experiment import Experiment


def main():
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--eval_batchsize', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--arch', type=str, default='bart', choices=['t5', 'bart', 'bert'])

    parser.add_argument('--encode_tgt_state', type=str, default='NL.bart',
                        choices=[False, 'raw.mlp', 'NL.bart', 'NL.t5'],
                        help="how to encode the state before probing")
    parser.add_argument('--tgt_agg_method', type=str,
                        choices=['sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default='avg',
                        help="how to aggregate across tokens of target, if `encode_tgt_state` is set True")
    # parser.add_argument('--probe_type', type=str, choices=['linear', 'mlp', 'lstm', 'decoder'], default='linear')
    parser.add_argument('--encode_init_state', type=str, default='NL', choices=[False, 'raw', 'NL'])
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lm_save_path', type=str, default=None, help="load existing LM checkpoint (if any)")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--probe_save_path', type=str, default=None,
                        help="load existing state model checkpoint (if any)")
    parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
    parser.add_argument('--probe_target', type=str,
                        choices=['text'] + [f'{target}.{text_type}' for target, text_type in itertools.product([
                            'state', 'init_state', 'single_beaker_init', 'single_beaker_final',
                        ], ['NL', 'raw'])], default="state.NL", help="what to probe for")
    parser.add_argument('--probe_agg_method', type=str,
                        choices=[None, 'sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'],
                        default='avg', help="how to aggregate across tokens")
    parser.add_argument('--probe_attn_dim', type=int, default=None,
                        help="what dimensions to compress sequence tokens to")
    single_beaker_opts = ['single_beaker_init', 'single_beaker_init_color', 'single_beaker_final',
                          'single_beaker_all'] + [
                             f'single_beaker_init_{att}'
                             for att in ['amount', 'full', 'verb', 'article', 'end_punct', 'pos.R0', 'pos.R1', 'pos.R2',
                                         'beaker.R0', 'beaker.R1', 'beaker.R2']
                         ]
    parser.add_argument('--localizer_type', type=str, default='all',
                        choices=['all', 'init_state'] + single_beaker_opts + [f'{opt}.offset{i}' for opt in
                                                                              single_beaker_opts for i in range(7)] + [
                                    f'single_beaker_{occurrence}{token_offset}{offset}' for occurrence in
                                    ["init", "init_full"] for offset in [""] + [f".offset{i}" for i in range(7)] for
                                    token_offset in [""] + [f".R{j}" for j in range(-7, 8)]
                                ],
                        help="which encoded tokens of the input to probe; `offset` gives how much ahead the encoded representation should be; `R` gives the offset in tokens, relative to 'beaker'")
    parser.add_argument('--probe_max_tokens', type=int, default=None,
                        help="how many tokens (max) to feed into probe, set None to use all tokens")
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()

    # make save path

    if args.use_state_loss:
        args.model_dir += "_state"
        args.model_dir += f"_{args.rap_prob}"

    if not path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    args.best_model_dir = path.join(args.model_dir, "best")
    if not path.exists(args.best_model_dir):
        os.makedirs(args.best_model_dir)

    args.model_path = path.join(args.model_dir, "model.pt")
    args.best_model_path = path.join(args.best_model_dir, "model.pt")

    # Experiment(args)


if __name__ == '__main__':
    main()