import torch
import argparse
import json
import itertools
from os import path
import logging
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizerFast

from data.alchemy.utils import int_to_word, colors
from data_transformer import convert_to_transformer_batches
from data.alchemy.parseScone import loadData


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def initialize_model(model_path: str):
	model = BartForConditionalGeneration.from_pretrained(model_path)
	tokenizer = BartTokenizerFast.from_pretrained(model_path)

	return model, tokenizer


def get_all_beaker_state_suffixes():
	all_beaker_states = set()
	d_colors = {color[0]: color for color in colors}
	for beaker_amount in range(5):
		if beaker_amount == 0:
			all_beaker_states.add("_")
		else:
			all_beaker_states = all_beaker_states.union(set(itertools.product(d_colors, repeat=beaker_amount)))

	outputs = set()
	for beaker_state in all_beaker_states:
		if '_' in beaker_state:
			string = f"is empty"
		else:
			colors_to_amount = {}
			for item in beaker_state:
				if d_colors[item] not in colors_to_amount:
					colors_to_amount[d_colors[item]] = 0
				colors_to_amount[d_colors[item]] += 1

			string = []
			for color in sorted(colors_to_amount.keys()):
				string.append(f"{colors_to_amount[color]} {color}")
			if len(string) > 1:
				string = " and ".join(string)
			else:
				string = string[0]
			string = f"has {string}"

		outputs.add(string)
	return list(outputs)


def get_all_states(tokenizer, device):
	state_suffixes = get_all_beaker_state_suffixes()
	all_seqs = []
	for idx in range(len(int_to_word)):
		beaker_str = int_to_word[idx]
		prefix = f"the {beaker_str} beaker "
		state_seqs = [('</s>[PROBE_START]' + prefix + state_suffix + '[PROBE_END]') for state_suffix in state_suffixes]

		state_seq_ids = tokenizer.batch_encode_plus(
			state_seqs, padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)

		all_seqs.append(state_seq_ids)

	return all_seqs


@torch.no_grad()
def probing_exp(model_path: str, base_dir: str):
	device = torch.device('cpu')
	if torch.cuda.is_available():
		device = torch.device('cuda')

	dev_dataset, _, _ = loadData(split="dev", kind="alchemy", synthetic=False, base_dir=base_dir)

	model, tokenizer = initialize_model(model_path)
	if torch.cuda.is_available():
		model = model.cuda()

	loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
	all_seqs = get_all_states(tokenizer, device)
	num_states = all_seqs[0].shape[0]

	total = 0
	corr = 0

	output = {}
	for j, (inputs, lang_tgts, state_tgts, raw_state_targets, init_states) in enumerate(
					convert_to_transformer_batches(dev_dataset, tokenizer, 1, domain="alchemy", device=device,
					                               state_targets_type='state.NL')):

		# print(raw_state_targets, tokenizer.batch_decode(inputs['input_ids']))
		state_target_str = raw_state_targets['full_state'][0].split(", ")
		# print(raw_state_targets)
		output[j] = {'state': raw_state_targets['full_state'][0], 'output': []}
		for seq_idx, all_seq in enumerate(all_seqs):
			return_dict = model(input_ids=inputs['input_ids'].repeat(num_states, 1).to(device),
			                    attention_mask=inputs['attention_mask'].repeat(num_states, 1).to(device),
			                    labels=all_seq, return_dict=True)

			lm_logits = return_dict.logits
			lang_loss = loss_fct(lm_logits.view(-1, len(tokenizer)), all_seq.view(-1))
			lang_loss = torch.sum(lang_loss.reshape_as(all_seq), dim=1)

			argmin = torch.argmin(lang_loss, dim=0).item()
			pred_state = tokenizer.decode(all_seq[argmin], skip_special_tokens=True).strip()
			gt_state = state_target_str[seq_idx].strip()

			output[j]['output'].append({'pred': pred_state, 'gt': gt_state, 'corr': pred_state == gt_state})
			logger.info(f"{pred_state}, {gt_state}")
			if pred_state == gt_state:
				corr += 1
			total += 1

			# break
		logger.info(f"Total: {total}, Correct: {corr}")
		# if j == 10:
		# 	break

	output_file = path.join(model_path, "dev.jsonl")
	logging.info(f'Output_file: {path.abspath(output_file)}')
	json.dump(output, open(output_file, 'w'), indent=4)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path', type=str, help="Model path")
	parser.add_argument('base_dir', type=str, default=None)

	args = parser.parse_args()

	assert (path.exists(args.model_path))
	probing_exp(args.model_path, args.base_dir)


if __name__=='__main__':
	main()