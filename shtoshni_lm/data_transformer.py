import itertools
import random
from data.alchemy.utils import int_to_word, colors, decide_translate, translate_states_to_nl
from data.alchemy.parseScone import getBatchesWithInit
from transformers import BartTokenizerFast

from shtoshni_lm.config import PROBE_START, PROBE_END


def identify_beaker_idx(state_targets, subsequent_state_targets):
    output = []
    for state_target, subsequent_state_target in zip(state_targets, subsequent_state_targets):
        for idx, (before_content, after_content) in enumerate(zip(state_target, subsequent_state_target)):
            if before_content.strip() != after_content.strip():
                output.append(idx)

        # print(state_target, subsequent_state_target, output[-1])

    return output


def represent_add_state_str(state_str):
    return PROBE_START + state_str + PROBE_END


def get_tokenized_decoder_seq(tokenizer, seq_list):
    return tokenizer(seq_list, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)


def get_tokenized_encoder_seq(tokenizer, seq_list):
    return tokenizer(seq_list, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)


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
        if "_" in beaker_state:
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

        # state_seqs = [represent_add_state_str(prefix + state_suffix) for state_suffix in state_suffixes]
        # Not adding PROBE_END because we're truncating the state sequence to just the current state
        state_seqs = [(PROBE_START + prefix + state_suffix + PROBE_END) for state_suffix in state_suffixes]

        state_seq_ids = tokenizer.batch_encode_plus(
            state_seqs, padding=True, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)

        all_seqs.append(state_seq_ids)

    return all_seqs


def convert_to_transformer_batches(
    dataset,
    tokenizer,
    batchsize,
    domain="alchemy",
    device="cuda",
    training=True,
    add_state="explanation",
    state_repr="all",
):
    """
    state_targets_type (str): what to return for `state_tgt_enc` and `state_targets`
        {state|init_state|interm_states|single_beaker_{...}|utt_beaker_{...}{.offset{1|2|...|6}}}.{NL|raw}|text
    """
    batches = list(getBatchesWithInit(dataset, batchsize, get_subsequent_state=True))

    # print(len(batches))
    if training:
        random.shuffle(batches)

    for batch in batches:
        (
            inputs,
            lang_targets,
            prev_state_targets,
            subsequent_state_targets,
            init_states,
        ) = zip(*batch)
        init_states = [" ".join(init_state) for init_state in init_states]

        state_targets = [
            decide_translate(
                " ".join(tgt),
                "state.NL",
                domain,
                isinstance(tokenizer, BartTokenizerFast),
            )
            for tgt in prev_state_targets
        ]

        # make inputs
        inps = []
        init_states_str = []
        steps = []
        for i, inp in enumerate(inputs):
            string = " ".join(inp).replace(" \n ", ". ")
            steps.append(string)
            init_state = translate_states_to_nl(init_states[i], domain, isinstance(tokenizer, BartTokenizerFast))
            init_states_str.append(init_state)
            string = init_state + ". " + string
            inps.append(string)

        lang_targets = [" ".join(tgt) + "." for tgt in lang_targets]

        # Encode inputs
        inp_enc = get_tokenized_encoder_seq(tokenizer, inps).to(device)

        # Encode outputs
        lang_tgt_enc = get_tokenized_decoder_seq(tokenizer, lang_targets).to(device)

        inp_enc["original_text"] = inps
        inp_enc["init_state"] = init_states_str
        inp_enc["steps"] = steps

        lang_tgt_enc["original_text"] = lang_targets
        lang_tgt_enc["input_ids"].masked_fill_(lang_tgt_enc["input_ids"] == tokenizer.pad_token_id, -100)
        lang_tgt_enc["input_ids"].to(device)

        beaker_targets = identify_beaker_idx(prev_state_targets, subsequent_state_targets)
        state_tgt_enc = None
        if add_state:
            target_list = []
            state_slice_list = []
            for (state_target, lang_target, beaker_target) in zip(state_targets, lang_targets, beaker_targets):
                if training:
                    beaker_states = state_target.split(", ")
                    random.shuffle(beaker_states)
                    state_slice = ", ".join(beaker_states)

                    if add_state == "multitask":
                        target_list.append(represent_add_state_str(state_slice))
                    else:
                        target_list.append(represent_add_state_str(state_slice) + lang_target)

                else:
                    state_slice = state_target

                state_slice_list.append(state_slice)

            if training:
                state_tgt_enc = get_tokenized_decoder_seq(tokenizer, target_list).to(device)
                state_tgt_enc["input_ids"].masked_fill_(state_tgt_enc["input_ids"] == tokenizer.pad_token_id, -100)
                state_tgt_enc["input_ids"].to(device)
            else:
                state_tgt_enc = {}
            state_tgt_enc["state_str"] = state_slice_list

        yield inp_enc, lang_tgt_enc, state_tgt_enc, state_targets, init_states
