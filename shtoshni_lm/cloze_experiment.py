from turtle import circle
import torch
import argparse
from os import path
import logging
from transformers import BartForConditionalGeneration, BartTokenizerFast

from shtoshni_lm.config import PROBE_START, PROBE_END
from shtoshni_lm.probing_experiment import get_all_states
from shtoshni_lm.data_transformer import represent_add_state_str
from data.alchemy.utils import int_to_word, translate_nl_to_states
import wandb
import json


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


def get_model_name(model_path):
    model_path = model_path.rstrip("/")
    model_path = model_path.rstrip("/best/doc_encoder")
    model_name = path.split(model_path)[1]

    return model_name


def load_cloze_data(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            data.append(line.strip().split("\t"))

    return data


class ClozeExperiment:
    def __init__(self, model_path, data_file, num_prev_steps=0):
        self.model_path = model_path
        self.model, self.tokenizer = self.initialize_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.dev_dataset = load_cloze_data(data_file)
        self.num_prev_steps = num_prev_steps

        self.model_name = get_model_name(model_path)
        self.all_seqs = get_all_states(self.tokenizer, self.model.device)
        self.num_states = self.all_seqs[0].shape[0]

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")

    @property
    def device(self) -> torch.device:
        return self.model.device

    def initialize_model(self):
        model = BartForConditionalGeneration.from_pretrained(self.model_path)
        tokenizer = BartTokenizerFast.from_pretrained(self.model_path)

        return model, tokenizer

    def predict_state(self, input_ids, attention_mask, gt_state_str=None):
        pred_state_seq = []
        gt_state_list = None
        if gt_state_str:
            gt_state_list = gt_state_str.split(", ")
        corr_state = 0

        for seq_idx, all_seq in enumerate(self.all_seqs):
            return_dict = self.model(
                input_ids=input_ids.repeat(self.num_states, 1).to(self.device),
                attention_mask=attention_mask.repeat(self.num_states, 1).to(self.device),
                labels=all_seq,
                return_dict=True,
            )

            lm_logits = return_dict.logits
            lang_loss = self.loss_fct(lm_logits.view(-1, len(self.tokenizer)), all_seq.view(-1))
            lang_loss = torch.sum(lang_loss.reshape_as(all_seq), dim=1)

            argmin = torch.argmin(lang_loss, dim=0).item()
            pred_state = self.tokenizer.decode(all_seq[argmin], skip_special_tokens=True).strip()
            pred_state_seq.append(pred_state)
            if gt_state_list:
                gt_state = gt_state_list[seq_idx].strip()
                if pred_state == gt_state:
                    corr_state += 1

        pred_state_str = ", ".join(pred_state_seq)
        pred_state_indices = self.tokenizer(
            represent_add_state_str(pred_state_str), return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        pred_state_indices = pred_state_indices.to(self.device)

        return pred_state_indices, pred_state_str, corr_state

    @torch.no_grad()
    def perform_cloze_exp(self):
        model_name = get_model_name(self.model_path)

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        if "state_" in model_name:
            probing_tokens_mask = [0.0] * (len(self.tokenizer) - 2) + [1.0, 1.0]
            logit_mask = torch.tensor(probing_tokens_mask, dtype=torch.float32, device=device)

        # Tokenize all next steps
        cloze_steps = list(list(zip(*self.dev_dataset))[3])
        cloze_seq_ids = self.tokenizer.batch_encode_plus(
            cloze_steps, padding=True, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)

        num_options = cloze_seq_ids.shape[0]

        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")

        mrr = 0
        corr_state = 0
        total_instances = len(cloze_steps)
        output_file = path.join(
            path.dirname(path.dirname(self.model_path.rstrip("/"))), f"cloze_mrr_fine_{self.num_prev_steps}.jsonl"
        )
        logger.info(f"Output_file: {path.abspath(output_file)}")

        with open(output_file, "w") as f:
            for idx, (init_state, prev_actions, cur_state_str, _) in enumerate(self.dev_dataset):
                input_string = init_state + ". " + prev_actions
                inputs = self.tokenizer(
                    input_string,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                ).to(device)

                instance_dict = {}
                pred_state_indices, pred_state_str, instance_corr_state = self.predict_state(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"], gt_state_str=cur_state_str
                )
                logger.info(f"Correct states: {instance_corr_state}")
                corr_state += instance_corr_state

                pred_state_indices = pred_state_indices.repeat(num_options, 1)
                labels = torch.cat([pred_state_indices, cloze_seq_ids], dim=1)

                return_dict = self.model(
                    input_ids=inputs["input_ids"].repeat(num_options, 1).to(device),
                    attention_mask=inputs["attention_mask"].repeat(num_options, 1).to(device),
                    labels=labels,
                    return_dict=True,
                )

                # print(self.tokenizer.batch_decode(labels.tolist()[:10]))

                num_state_tokens = pred_state_indices.shape[1]
                lm_logits = torch.squeeze(return_dict.logits, dim=0)
                lm_logits = lm_logits[
                    :,
                    num_state_tokens:,
                ]

                if "state_" in model_name:
                    lm_logits = lm_logits * (1 - logit_mask) + logit_mask * (-1e10)

                lang_loss = self.loss_fct(lm_logits.view(-1, len(self.tokenizer)), cloze_seq_ids.view(-1))
                lang_loss = torch.sum(lang_loss.reshape_as(cloze_seq_ids), dim=1)
                # lang_loss = torch.mean(lang_loss.reshape_as(cloze_seq_ids), dim=1)

                sorted_list = [val.item() for val in list(torch.sort(lang_loss)[1])]

                rank_idx = sorted_list.index(idx)
                cur_mrr = 1 / (rank_idx + 1)

                instance_dict["input_string"] = input_string
                instance_dict["gt_state"] = translate_nl_to_states(cur_state_str, domain="alchemy")
                instance_dict["pred_state"] = translate_nl_to_states(pred_state_str, domain="alchemy")

                instance_dict["corr_state"] = str(instance_corr_state)

                instance_dict["gt_action"] = cloze_steps[idx]
                instance_dict["pred_action"] = cloze_steps[sorted_list[0]]

                instance_dict["rank"] = str(rank_idx)
                instance_dict["sorted_list"] = sorted_list
                # print(instance_dict)

                f.write(json.dumps(instance_dict) + "\n")
                f.flush()
                mrr += cur_mrr
                # break

        mrr /= len(self.dev_dataset)
        state_tracking_acc = (corr_state * 100.0) / (total_instances * len(int_to_word))
        logger.info(f"State tracking accuracy: {state_tracking_acc}")
        print(f"{mrr: .3f}")
        # wandb.log({"dev/cloze_mrr": mrr})

        output_file = path.join(
            path.dirname(path.dirname(self.model_path.rstrip("/"))), f"cloze_mrr_{self.num_prev_steps}.txt"
        )
        with open(output_file, "w") as f:
            f.write(f"{mrr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Model path")
    parser.add_argument("base_dir", type=str, default=None)

    args = parser.parse_args()

    assert path.exists(args.model_path)

    for num_prev_steps in range(4):
        cloze_file = path.join(args.base_dir, f"cloze_data/alchemy_{num_prev_steps}.txt")
        cloze_exp = ClozeExperiment(model_path=args.model_path, data_file=cloze_file, num_prev_steps=num_prev_steps)
        model_name = get_model_name(args.model_path)
        cloze_exp.perform_cloze_exp()

        # wandb.init(
        #     id=model_name,
        #     project="state-probing",
        #     resume=True,
        #     notes="State probing",
        #     tags="november",
        #     config={},
        # )


if __name__ == "__main__":
    main()
