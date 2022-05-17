from data.alchemy.utils import int_to_word, decide_translate, translate_states_to_nl
from data.alchemy.parseScone import getBatchesWithInit
from transformers import BartTokenizerFast, T5TokenizerFast

from data.alchemy.utils import int_to_word
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


def get_tokenized_seq(tokenizer, seq_list):
    return tokenizer(seq_list, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)


def convert_to_transformer_batches(
    dataset,
    tokenizer,
    batchsize,
    random=None,
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
    if random:
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
        inp_enc = get_tokenized_seq(tokenizer, inps).to(device)

        # Encode outputs
        lang_tgt_enc = get_tokenized_seq(tokenizer, lang_targets).to(device)

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
                state_tgt_enc = get_tokenized_seq(tokenizer, target_list).to(device)
                state_tgt_enc["input_ids"].masked_fill_(state_tgt_enc["input_ids"] == tokenizer.pad_token_id, -100)
                state_tgt_enc["input_ids"].to(device)
            else:
                state_tgt_enc = {}
            state_tgt_enc["state_str"] = state_slice_list

        yield inp_enc, lang_tgt_enc, state_tgt_enc, state_targets, init_states
