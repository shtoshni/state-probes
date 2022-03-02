import torch
import argparse
import json
import itertools
import os
from os import path
import logging
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizerFast

from data.alchemy.utils import int_to_word, colors
# from data_transformer import convert_to_transformer_batches
from data.alchemy.parseScone import loadData
from shtoshni_probing.config import PROBE_START, PROBE_END
import wandb


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def get_model_name(model_path):
	model_path = model_path.rstrip("/")
	model_path = model_path.rstrip("/best/doc_encoder")
	model_name = path.split(model_path)[1]

	return model_name


def initialize_model(model_path: str):
	model = BartForConditionalGeneration.from_pretrained(model_path)
	tokenizer = BartTokenizerFast.from_pretrained(model_path)

	return model, tokenizer


# def get_all_states(tokenizer, device):
# 	# state_suffixes = get_all_beaker_state_suffixes()
# 	all_seqs = []
# 	for idx in range(len(int_to_word)):
# 		beaker_str = int_to_word[idx]
# 		prefix = f"the {beaker_str} beaker "
# 		state_seqs = [(PROBE_START + prefix + state_suffix + PROBE_END) for state_suffix in state_suffixes]
#
# 		state_seq_ids = tokenizer.batch_encode_plus(
# 			state_seqs, padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
#
# 		all_seqs.append(state_seq_ids)
#
# 	return all_seqs


# def convert_to_transformer_batches(
#     dataset, tokenizer, batchsize, random=None,
#     domain="alchemy",
#     state_targets_type="state.NL", device='cuda', num_instances=20,
# ):
# 	state_targets_type_split = state_targets_type.split('.')
# 	batches = list(getBatchesWithInit(dataset, batchsize, get_subsequent_state=True))
#
# 	for batch in batches:
# 		inputs, lang_targets, prev_state_targets, subsequent_state_targets, init_states = zip(*batch)
# 		for



def loadClozeData(data_file):
	data = []
	with open(data_file) as f:
		for line in f:
			data.append(line.strip().split('\t'))

	return data


@torch.no_grad()
def probing_exp(model_path: str, base_dir: str):
	model_name = get_model_name(model_path)

	device = torch.device('cpu')
	if torch.cuda.is_available():
		device = torch.device('cuda')

	dev_dataset = loadClozeData(path.join(base_dir, "cloze_data/alchemy.txt"))
	model, tokenizer = initialize_model(model_path)
	if 'state_' in model_name:
		probing_tokens_mask = [0.0] * (len(tokenizer) - 2) + [1.0, 1.0]
		logit_mask = torch.tensor(
			probing_tokens_mask, dtype=torch.float32, device=device)

	if torch.cuda.is_available():
		model = model.cuda()

	# Tokenize all next steps
	cloze_steps = list(list(zip(*dev_dataset))[2])
	# print(cloze_steps)
	# cloze_seq_ids = tokenizer.batch_encode_plus(
	# 	cloze_steps, padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
	cloze_seq_ids = tokenizer.batch_encode_plus(
		cloze_steps, padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)

	num_states = cloze_seq_ids.shape[0]

	loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")

	total = 0
	corr = 0

	output = {}

	mrr = 0
	for idx, (init_state, prev_actions, next_action) in enumerate(dev_dataset):
		input_string = init_state + '. ' + prev_actions
		inputs = tokenizer(input_string, return_tensors='pt', padding=True, truncation=False, return_offsets_mapping=True).to(
			device)

		return_dict = model(input_ids=inputs['input_ids'].repeat(num_states, 1).to(device),
		                    attention_mask=inputs['attention_mask'].repeat(num_states, 1).to(device),
          		          labels=cloze_seq_ids, return_dict=True)

		lm_logits = return_dict.logits
		lm_logits = lm_logits * (1 - logit_mask) + logit_mask * (-1e10)

		lang_loss = loss_fct(lm_logits.view(-1, len(tokenizer)), cloze_seq_ids.view(-1))
		lang_loss = torch.sum(lang_loss.reshape_as(cloze_seq_ids), dim=1)

		rank_idx = list(torch.sort(lang_loss)[1]).index(idx)
		mrr += 1/(rank_idx + 1)

		# print(lang_loss.shape)
		# break

	mrr /= len(dev_dataset)
	print(f"{mrr: .3f}")
	# wandb.log({"dev/probing_acc": corr*100/total})
	wandb.log({"dev/cloze_mrr": mrr, "steps": 0})

	output_file = path.join(path.dirname(path.dirname(model_path.rstrip("/"))), "cloze_mrr.txt")
	with open(output_file) as f:
		f.write(f"{mrr: .3f}")


	# output_file = path.join(path.dirname(path.dirname(model_path.rstrip("/"))), "dev.jsonl")
	# logging.info(f'Output_file: {path.abspath(output_file)}')
	# json.dump(output, open(output_file, 'w'), indent=4)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path', type=str, help="Model path")
	parser.add_argument('base_dir', type=str, default=None)

	args = parser.parse_args()

	assert (path.exists(args.model_path))

	model_name = get_model_name(args.model_path)
	wandb.init(
		id=model_name, project="state-probing", resume=True,
		notes="State probing", tags="november", config={},
	)
	probing_exp(args.model_path, args.base_dir)


if __name__=='__main__':
	main()