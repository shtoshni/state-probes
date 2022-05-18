import sys
import os
import time
import random
import wandb
import json
import torch
import numpy as np

from os import path

from transformers import BartTokenizerFast, BartForConditionalGeneration, get_linear_schedule_with_warmup

from data.alchemy.utils import int_to_word
from data.alchemy.parseScone import loadData
from shtoshni_lm.config import PROBE_START, PROBE_END
from shtoshni_lm.data_transformer import (
    represent_add_state_str,
    convert_to_transformer_batches,
    get_tokenized_seq,
    get_all_states,
)
from shtoshni_lm.base_logger import logger


class Experiment(object):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Whether to train or not
        self.eval_model: bool = self.args.eval

        # Step 1 - Build model
        self._build_model()

        # Step 2 - Load Data - Data processing choices such as tokenizer will depend on the model
        self._load_data()

        if self.args.add_state:
            self.all_seqs = get_all_states(self.tokenizer, self.device)
            self.num_states = self.all_seqs[0].shape[0]

        # Step 3 - Load model and resume training if required
        # Firstly initialize dictionary to track key training variables
        self.train_info = {
            "best_val_loss": 10**10,
            "global_steps": 0,
            "num_epochs": 0,
            "num_stuck_evals": 0,
        }
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.state_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")

        if self.eval_model:
            # Load the best checkpoint
            self._load_previous_checkpoint(last_checkpoint=False)
        else:
            # Resume training
            self._build_optimizers()
            # Loading the checkpoint also restores the training metadata
            self._load_previous_checkpoint(last_checkpoint=True)

            # All set to resume training
            # But first check if training is remaining
            if self._is_training_remaining():
                self.train()

        # Step 4 - Perform final evaluation
        if path.exists(self.best_model_path):
            self._load_model(self.best_model_path, last_checkpoint=False)
            self.final_eval()
        elif path.exists(self.model_path):
            logger.info("Couldn't find the best model! Using the last checkpoint!")
            self._load_model(self.model_path, last_checkpoint=False)
            self.final_eval()
        else:
            logger.info("No model accessible!")
            sys.exit(1)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _build_model(self):
        """Constructs the model with given config."""
        model_fp = f"facebook/bart-{self.args.model_size}"
        self.tokenizer = BartTokenizerFast.from_pretrained(model_fp)
        self.model = BartForConditionalGeneration.from_pretrained(model_fp)

        if torch.cuda.is_available():
            self.model.cuda()

        if self.args.add_state:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [PROBE_START, PROBE_END]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Mask to mask out these additional tokens
            self.probing_tokens_mask = [0.0] * (len(self.tokenizer) - 2) + [1.0, 1.0]

    def _load_data(self):
        # loading data
        self.dataset, _, _ = loadData(split="train", kind="alchemy", synthetic=False, base_dir=self.args.base_data_dir)
        if self.args.num_train is not None:
            self.dataset = self.dataset[: self.args.num_train]
        self.dev_dataset, _, _ = loadData(
            split="dev", kind="alchemy", synthetic=False, base_dir=self.args.base_data_dir
        )
        if self.args.num_dev is not None:
            self.dev_dataset = self.dev_dataset[: self.args.num_dev]

    def _load_previous_checkpoint(self, last_checkpoint=True):
        """Loads the last checkpoint or best checkpoint.

        Args:
                last_checkpoint: If true, load the last checkpoint to resume training.
                        Otherwise, load the best model for evaluation.
                        If the above two models don't exist, set the random seeds for training initialization.
        """
        self.model_path = self.args.model_path
        self.best_model_path = self.args.best_model_path

        if last_checkpoint:
            # Resume training
            if path.exists(self.model_path):
                self._load_model(self.model_path, last_checkpoint=last_checkpoint)
            else:
                # Starting training
                torch.random.manual_seed(self.args.seed)
                np.random.seed(self.args.seed)
                random.seed(self.args.seed)
        else:
            # Load best model
            if path.exists(self.best_model_path):
                self._load_model(self.best_model_path, last_checkpoint=last_checkpoint)
            else:
                raise IOError(f"Best model path at {self.best_model_path} not found")

            logger.info("\nModel loaded\n")
            sys.stdout.flush()

    def _is_training_remaining(self):
        """Check if training is done or remaining.

        There are two cases where we don't resume training:
        (a) The dev performance has not improved for the allowed patience parameter.
        (b) Number of gradient updates is already >= Total training steps.

        Returns:
                bool: If true, we resume training. Otherwise do final evaluation.
        """
        if self.train_info["num_stuck_evals"] >= self.args.patience:
            return False
        if self.train_info["num_epochs"] >= self.args.epochs:
            return False

        return True

    def _build_optimizers(self):
        """Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""
        # Optimizer for clustering params
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        num_training_steps = ((610 * 24) // self.args.batchsize) * self.args.epochs
        logger.info(f"Number of training steps: {num_training_steps}")
        self.optim_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.1 * num_training_steps,
            num_training_steps=num_training_steps,
        )

    def train(self) -> None:
        """Method for training the model.

        This method implements the training loop.
        Within the training loop, the model is periodically evaluated on the dev set(s).
        """
        model, optimizer, optim_scheduler = (
            self.model,
            self.optimizer,
            self.optim_scheduler,
        )

        while True:
            logger.info("Steps done %d" % (self.train_info["global_steps"]))
            lang_train_losses = []
            model.train()
            start_time = time.time()

            for j, (inputs, lang_tgts, state_tgts, _, _) in enumerate(
                convert_to_transformer_batches(
                    self.dataset,
                    self.tokenizer,
                    self.args.batchsize,
                    # Training information
                    training=True,
                    # State variable
                    add_state=self.args.add_state,
                    state_repr=self.args.state_repr,
                    # Device info
                    device=self.device,
                )
            ):

                def handle_example():
                    if self.args.add_state == "explanation":
                        target = state_tgts["input_ids"]

                    elif self.args.add_state == "ras":
                        if random.random() < self.args.rap_prob:
                            target = state_tgts["input_ids"]
                        else:
                            target = lang_tgts["input_ids"]
                    else:
                        target = lang_tgts["input_ids"]

                    if random.random() < 0.01:
                        logger.info(f"\nEncoder sequence: {self.tokenizer.decode(inputs['input_ids'][0])}")
                        if self.args.add_state:
                            output_seq = torch.clone(state_tgts["input_ids"][0])
                            output_seq.masked_fill_(output_seq == -100, self.tokenizer.pad_token_id)
                            logger.info(f"Probing Decoder sequence: {self.tokenizer.decode(output_seq)}\n")

                        output_seq = torch.clone(lang_tgts["input_ids"][0])
                        output_seq.masked_fill_(output_seq == -100, self.tokenizer.pad_token_id)
                        logger.info(f"Simple Decoder sequence: {self.tokenizer.decode(output_seq)}\n")

                    return_dict = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=target,
                        return_dict=True,
                    )

                    loss = return_dict.loss
                    return update_model(loss)

                def handle_example_multitask():
                    return_dict_lang = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=lang_tgts["input_ids"],
                        return_dict=True,
                    )

                    return_dict_state = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=state_tgts["input_ids"],
                        return_dict=True,
                    )

                    if random.random() < 0.01:
                        logger.info(f"\nEncoder sequence: {self.tokenizer.decode(inputs['input_ids'][0])}")
                        output_seq = torch.clone(state_tgts["input_ids"][0])
                        output_seq.masked_fill_(output_seq == -100, self.tokenizer.pad_token_id)
                        logger.info(f"Probing Decoder sequence: {self.tokenizer.decode(output_seq)}\n")

                        output_seq = torch.clone(lang_tgts["input_ids"][0])
                        output_seq.masked_fill_(output_seq == -100, self.tokenizer.pad_token_id)
                        logger.info(f"Simple Decoder sequence: {self.tokenizer.decode(output_seq)}\n")

                    # Multitasking loss weighed by RAP probability
                    loss = (
                        return_dict_lang.loss * (1 - self.args.rap_prob) + return_dict_state.loss * self.args.rap_prob
                    )
                    return update_model(loss)

                def update_model(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optim_scheduler.step()

                    loss_val = loss.item()
                    lang_train_losses.append(loss_val)
                    self.train_info["global_steps"] += 1

                    return loss_val

                if self.args.add_state == "multitask":
                    lang_loss = handle_example_multitask()
                else:
                    lang_loss = handle_example()

                if self.args.use_wandb:
                    wandb.log(
                        {
                            "train/loss": lang_loss,
                        }
                    )

                if j % 100 == 0:
                    output_str = f"epoch {self.train_info['num_epochs']}, batch {j}, lang score: {lang_loss: .3f}"
                    logger.info(output_str)

            self.train_info["num_epochs"] += 1
            logger.info(
                f"epoch {self.train_info['num_epochs']}, avg lang loss {(sum(lang_train_losses) / len(lang_train_losses)):.3f}"
            )
            dev_loss = self.periodic_model_eval()
            if self.args.use_wandb:
                wandb.log({"dev/loss": dev_loss, "batch": self.train_info["global_steps"]})
                wandb.log({"dev/best_loss": self.train_info["best_val_loss"]})

            # Get elapsed time
            elapsed_time = time.time() - start_time

            logger.info(
                "Steps: %d, Log-loss: %.3f, Best log-loss: %.3f, Time: %.2f\n\n"
                % (
                    self.train_info["global_steps"],
                    dev_loss,
                    self.train_info["best_val_loss"],
                    elapsed_time,
                )
            )

            # Check stopping criteria
            if not self._is_training_remaining():
                break

    def predict_state(self, model, input_ids, attention_mask, gt_state_str):
        pred_state_seq = []
        gt_state_list = gt_state_str.split(", ")
        corr_state = 0

        for seq_idx, all_seq in enumerate(self.all_seqs):
            return_dict = model(
                input_ids=input_ids.repeat(self.num_states, 1).to(self.device),
                attention_mask=attention_mask.repeat(self.num_states, 1).to(self.device),
                labels=all_seq,
                return_dict=True,
            )

            lm_logits = return_dict.logits
            lang_loss = self.state_loss_fct(lm_logits.view(-1, len(self.tokenizer)), all_seq.view(-1))
            lang_loss = torch.sum(lang_loss.reshape_as(all_seq), dim=1)

            argmin = torch.argmin(lang_loss, dim=0).item()
            pred_state = self.tokenizer.decode(all_seq[argmin], skip_special_tokens=True).strip()
            pred_state_seq.append(pred_state)

            gt_state = gt_state_list[seq_idx].strip()
            if pred_state == gt_state:
                corr_state += 1

        pred_state_str = represent_add_state_str(", ".join(pred_state_seq))
        pred_state_indices = self.tokenizer.tokenize(pred_state_str)

        return pred_state_indices, pred_state_str, corr_state

    @torch.no_grad()
    def get_dataset_loss(self, model, dataset) -> float:
        total_tokens = 0
        n_val = 0
        tot_val_loss = 0
        # Track state-level prediction stats
        total_state_corr, total_state_pred = 0, 0
        # Track entity-level prediction stats
        total_entity_corr, total_entity_pred = 0, 0

        if self.args.add_state:
            output_file = path.join(self.args.model_dir, "state_tracking.jsonl")
            state_writer = open(output_file, "w")

        for (inputs, lang_tgts, state_tgts, _, _) in convert_to_transformer_batches(
            dataset,
            self.tokenizer,
            batchsize=1,  # We set dev batchsize to 1
            device=self.device,
            add_state=self.args.add_state,
            training=False,
            state_repr=self.args.state_repr,
        ):

            if self.args.add_state:
                # Step 1: Predict state
                gt_state_str = state_tgts["state_str"][0]  # Indexing with 0 because the batchsize is 1
                pred_state_indices, pred_state_str, corr_state = self.predict_state(
                    model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    gt_state_str=gt_state_str,
                )
                # Update state prediction stats
                total_state_corr += int(corr_state == len(int_to_word))
                total_state_pred += 1

                # Update entity prediction stats
                total_entity_corr += corr_state
                total_entity_pred += len(int_to_word)

                output_dict = {"gt": gt_state_str, "pred": pred_state_str, "corr": corr_state}
                state_writer.write(json.dumps(output_dict, indent=4) + "\n")

                lang_target = lang_tgts["original_text"][0]  # Indexing with 0 because the batchsize is 1

                if self.args.add_state == "explanation":
                    # Predicted state is part of decoder input
                    labels = pred_state_str + lang_target
                else:
                    labels = lang_target
                label_ids = get_tokenized_seq(self.tokenizer, [labels])["input_ids"].to(self.device)
            else:
                label_ids = lang_tgts["input_ids"]

            return_dict = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=label_ids,
                return_dict=True,
            )

            lm_logits = return_dict.logits
            lm_logits = lm_logits.view(-1, len(self.tokenizer))

            if self.args.add_state:
                logit_mask = torch.tensor(
                    self.probing_tokens_mask,
                    dtype=torch.float32,
                    device=inputs["input_ids"].device,
                )
                lm_logits = lm_logits * (1 - logit_mask) + logit_mask * (-1e10)

                if self.args.add_state == "explanation":
                    # Remove predicted state tokens from loss calculation
                    num_state_tokens = len(pred_state_indices)
                    lm_logits = lm_logits[num_state_tokens:]

            lang_loss = self.loss_fct(lm_logits, lang_tgts["input_ids"].view(-1))
            tot_val_loss += lang_loss.item()
            # Num instances
            n_val += len(inputs["input_ids"])
            # Num tokens
            num_tokens = torch.sum((lang_tgts["input_ids"] != -100).to(torch.float)).item()
            total_tokens += num_tokens

        logger.info(f"Total instances: {n_val}, Num tokens: {total_tokens}")
        if self.args.add_state:
            # State tracking accuracy
            state_tracking_acc = (100.0 * total_state_corr) / total_state_pred
            logger.info(f"State tracking accuracy: {state_tracking_acc:.2f}")
            if self.args.use_wandb:
                wandb.log({"State Tracking Acc": state_tracking_acc})

            # Entity tracking accuracy
            entity_tracking_acc = (100.0 * total_entity_corr) / total_entity_pred
            logger.info(f"Entity tracking accuracy: {entity_tracking_acc:.2f}")
            if self.args.use_wandb:
                wandb.log({"Entity Tracking Acc": entity_tracking_acc})

            state_writer.close()

        avg_val_loss = tot_val_loss / total_tokens
        return avg_val_loss

    @torch.no_grad()
    def periodic_model_eval(self):
        model = self.model
        model.eval()

        avg_val_loss = self.get_dataset_loss(model, self.dev_dataset)
        logger.info(f"epoch {self.train_info['num_epochs']}, avg val loss: {avg_val_loss}")
        if avg_val_loss <= self.train_info["best_val_loss"]:
            logger.info("NEW BEST MODEL")
            self.train_info["best_val_loss"] = avg_val_loss
            self.train_info["num_stuck_evals"] = 0
            self.save_model(self.args.best_model_path, last_checkpoint=False)
        else:
            self.train_info["num_stuck_evals"] += 1

        self.save_model(self.args.model_path, last_checkpoint=True)
        return avg_val_loss

    @torch.no_grad()
    def final_eval(self):
        model = self.model
        model.eval()
        avg_val_loss = self.get_dataset_loss(model, self.dev_dataset)
        logger.info(f"epoch {self.train_info['num_epochs']}, avg val loss: {avg_val_loss}")
        return avg_val_loss

    def _load_model(self, location: str, last_checkpoint=True) -> None:
        """Load model from given location.

        Args:
                location: str
                        Location of checkpoint
                last_checkpoint: bool
                        Whether the checkpoint is the last one saved or not.
                        If false, don't load optimizers, schedulers, and other training variables.
        """

        checkpoint = torch.load(location, map_location="cpu")
        doc_encoder_dir = path.join(path.dirname(location), "doc_encoder")
        logger.info("Loading document encoder from %s" % path.abspath(doc_encoder_dir))

        # Load the encoder
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=doc_encoder_dir).to(
            self.device
        )
        self.tokenizer = BartTokenizerFast.from_pretrained(pretrained_model_name_or_path=doc_encoder_dir)

        if last_checkpoint:
            # If resuming training, restore the optimizer state as well
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.optim_scheduler.load_state_dict(checkpoint["scheduler"])

            self.train_info = checkpoint["train_info"]
            torch.set_rng_state(checkpoint["rng_state"])
            np.random.set_state(checkpoint["np_rng_state"])

    def save_model(self, location: os.PathLike, last_checkpoint=True) -> None:
        """Save model.

        Args:
                location: Location of checkpoint
                last_checkpoint:
                        Whether the checkpoint is the last one saved or not.
                        If false, don't save optimizers and schedulers which take up a lot of space.
        """

        doc_encoder_dir = path.join(path.dirname(location), "doc_encoder")
        if not path.exists(doc_encoder_dir):
            os.makedirs(doc_encoder_dir)

        logger.info(f"Encoder saved at {path.abspath(doc_encoder_dir)}")
        # Save the encoder
        self.model.save_pretrained(save_directory=doc_encoder_dir, save_config=True)
        # Save the tokenizer
        self.tokenizer.save_pretrained(doc_encoder_dir)

        save_dict = {
            "train_info": self.train_info,
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "args": self.args,
        }

        if last_checkpoint:
            # For last checkpoint save the optimizer and scheduler states as well
            save_dict["optimizer"] = self.optimizer.state_dict()
            save_dict["scheduler"] = self.optim_scheduler.state_dict()

        torch.save(save_dict, location)
        logger.info(f"Model saved at: {path.abspath(location)}")
