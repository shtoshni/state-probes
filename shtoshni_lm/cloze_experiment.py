import torch
import argparse
from os import path
import logging
from transformers import BartForConditionalGeneration, BartTokenizerFast

from shtoshni_lm.config import PROBE_START, PROBE_END
from shtoshni_lm.probing_experiment import get_all_states
from shtoshni_lm.data_transformer import represent_add_state_str
import wandb


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
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.dev_dataset = load_cloze_data(data_file)

        self.model_name = get_model_name(model_path)

        self.model, self.tokenizer = self.initialize_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.all_seqs = get_all_states(self.tokenizer, self.model.device)
        self.num_states = self.all_seqs[0].shape[0]

        self.state_loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="none"
        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    def initialize_model(self):
        model = BartForConditionalGeneration.from_pretrained(self.model_path)
        tokenizer = BartTokenizerFast.from_pretrained(self.model_path)

        return model, tokenizer

    def predict_state(self, input_ids, attention_mask, gt_state_str):
        pred_state_seq = []
        gt_state_list = gt_state_str.split(", ")
        corr_state = 0

        for seq_idx, all_seq in enumerate(self.all_seqs):
            return_dict = self.model(
                input_ids=input_ids.repeat(self.num_states, 1).to(self.device),
                attention_mask=attention_mask.repeat(self.num_states, 1).to(
                    self.device
                ),
                labels=all_seq,
                return_dict=True,
            )

            lm_logits = return_dict.logits
            lang_loss = self.state_loss_fct(
                lm_logits.view(-1, len(self.tokenizer)), all_seq.view(-1)
            )
            lang_loss = torch.sum(lang_loss.reshape_as(all_seq), dim=1)

            argmin = torch.argmin(lang_loss, dim=0).item()
            pred_state = self.tokenizer.decode(
                all_seq[argmin], skip_special_tokens=True
            ).strip()
            pred_state_seq.append(pred_state)

            gt_state = gt_state_list[seq_idx].strip()
            if pred_state == gt_state:
                corr_state += 1

        pred_state_str = represent_add_state_str(", ".join(pred_state_seq))
        pred_state_indices = self.tokenizer.tokenize(pred_state_str)

        return pred_state_indices, pred_state_str, corr_state

    @torch.no_grad()
    def perform_cloze_exp(self):
        model_name = get_model_name(self.model_path)

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        if "state_" in model_name:
            probing_tokens_mask = [0.0] * (len(self.tokenizer) - 2) + [1.0, 1.0]
            logit_mask = torch.tensor(
                probing_tokens_mask, dtype=torch.float32, device=device
            )

        # Tokenize all next steps
        cloze_steps = list(list(zip(*self.dev_dataset))[3])
        cloze_seq_ids = self.tokenizer.batch_encode_plus(
            cloze_steps, padding=True, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)

        num_states = cloze_seq_ids.shape[0]

        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="none"
        )

        mrr = 0
        output_file = path.join(
            path.dirname(path.dirname(self.model_path.rstrip("/"))), "cloze_mrr_fine.txt"
        )
        logging.info(f"Output_file: {path.abspath(output_file)}")

        with open(output_file, "w") as f:
            for idx, (init_state, prev_actions, cur_state, next_action) in enumerate(
                self.dev_dataset
            ):
                input_string = init_state + ". " + prev_actions
                inputs = self.tokenizer(
                    input_string,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_offsets_mapping=True,
                ).to(device)

                return_dict = self.model(
                    input_ids=inputs["input_ids"].repeat(num_states, 1).to(device),
                    attention_mask=inputs["attention_mask"]
                    .repeat(num_states, 1)
                    .to(device),
                    labels=cloze_seq_ids,
                    return_dict=True,
                )

                lm_logits = return_dict.logits
                if "state_" in model_name:
                    lm_logits = lm_logits * (1 - logit_mask) + logit_mask * (-1e10)

                lang_loss = loss_fct(
                    lm_logits.view(-1, len(self.tokenizer)), cloze_seq_ids.view(-1)
                )
                lang_loss = torch.sum(lang_loss.reshape_as(cloze_seq_ids), dim=1)

                rank_idx = list(torch.sort(lang_loss)[1]).index(idx)
                cur_mrr = 1 / (rank_idx + 1)
                f.write(str(cur_mrr) + "\n")
                mrr += cur_mrr

        mrr /= len(self.dev_dataset)
        print(f"{mrr: .3f}")
        # wandb.log({"dev/probing_acc": corr*100/total})
        wandb.log({"dev/cloze_mrr": mrr})

        output_file = path.join(
            path.dirname(path.dirname(self.model_path.rstrip("/"))), "cloze_mrr.txt"
        )
        with open(output_file, "w") as f:
            f.write(f"{mrr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Model path")
    parser.add_argument("base_dir", type=str, default=None)

    args = parser.parse_args()

    assert path.exists(args.model_path)
    cloze_file = path.join(args.base_dir, "cloze_data/alchemy.txt")

    cloze_exp = ClozeExperiment(model_path=args.model_path, data_file=cloze_file)
    cloze_exp.perform_cloze_exp()

    model_name = get_model_name(args.model_path)
    wandb.init(
        id=model_name,
        project="state-probing",
        resume=True,
        notes="State probing",
        tags="november",
        config={},
    )
    probing_exp(args.model_path, args.base_dir)


if __name__ == "__main__":
    main()
