
from data.alchemy.utils import int_to_word, decide_translate, translate_states_to_nl
from data.alchemy.parseScone import getBatchesWithInit
from transformers import BartTokenizerFast, T5TokenizerFast

from data.alchemy.utils import int_to_word

from shtoshni_probing.config import PROBE_START, PROBE_END


def identify_beaker_idx(state_targets, subsequent_state_targets):
    output = []
    for state_target, subsequent_state_target in zip(state_targets, subsequent_state_targets):
        for idx, (before_content, after_content) in enumerate(
                zip(state_target, subsequent_state_target)):
            if before_content.strip() != after_content.strip():
                output.append(idx)

        # print(state_target, subsequent_state_target, output[-1])

    return output


def convert_to_transformer_batches(
        dataset, tokenizer, batchsize, random=None, domain="alchemy",
        state_targets_type="state.NL", device='cuda', add_state='random',
):
    """
    state_targets_type (str): what to return for `state_tgt_enc` and `state_targets`
        {state|init_state|interm_states|single_beaker_{...}|utt_beaker_{...}{.offset{1|2|...|6}}}.{NL|raw}|text
    """
    state_targets_type_split = state_targets_type.split('.')
    batches = list(getBatchesWithInit(dataset, batchsize, get_subsequent_state=True))

    if random:
        random.shuffle(batches)

    for batch in batches:
        inputs, lang_targets, prev_state_targets, subsequent_state_targets, init_states = zip(*batch)
        init_states = [' '.join(init_state) for init_state in init_states]

        # The target in this case is the next/subsequent state
        state_targets = [
            decide_translate(' '.join(tgt), state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
            for tgt in subsequent_state_targets]
        state_targets = {'full_state': state_targets}

        # make inputs
        inps = []
        init_states_str = []
        steps = []
        for i, inp in enumerate(inputs):
            string = ' '.join(inp).replace(' \n ', '. ')
            steps.append(string)
            init_state = translate_states_to_nl(
                init_states[i], domain, isinstance(tokenizer, BartTokenizerFast))
            init_states_str.append(init_state)
            string = init_state + '. ' + string
            inps.append(string)

        if state_targets_type_split[0] == 'text':
            state_targets = {'original_text': inps}

        # make lang targets
        lang_targets_new = []
        for tgt in lang_targets:
            tgt = ' '.join(tgt)
            lang_targets_new.append(tgt)  # + tokenizer.eos_token)
        lang_targets = lang_targets_new

        inp_enc = tokenizer(
            inps, return_tensors='pt', padding=True, truncation=False,
            return_offsets_mapping=True).to(device)
        lang_tgt_enc = tokenizer(
            lang_targets, return_tensors='pt', padding=True, truncation=True,
            add_special_tokens=True).to(device)

        inp_enc['original_text'] = inps
        inp_enc['init_state'] = init_states_str
        inp_enc['steps'] = steps

        lang_tgt_enc['original_text'] = lang_targets
        lang_tgt_enc['input_ids'].masked_fill_(lang_tgt_enc['input_ids'] == tokenizer.pad_token_id, -100)
        lang_tgt_enc['input_ids'].to(device)

        beaker_targets = identify_beaker_idx(prev_state_targets, subsequent_state_targets)
        for state_key in state_targets:
            state_tgt_enc = None
            if random:
                target_list = []
                for (state_target, lang_target, beaker_target) in zip(
                        state_targets[state_key], lang_targets, beaker_targets):
                    if add_state == 'all':
                        # Randomize the state slice
                        beaker_states = state_target.split(", ")
                        random.shuffle(beaker_states)
                        state_slice = ", ".join(beaker_states)
                    elif add_state == 'targeted':
                        # beaker_idx = identify_beaker_idx(lang_target)
                        if beaker_target == -1:
                            state_slice = random.choice(state_target.split(","))
                        else:
                            state_slice = state_target.split(",")[beaker_target]
                    elif add_state == 'random':
                        state_slice = random.choice(state_target.split(","))
                    else:
                        raise ValueError(add_state)
                    target_list.append(
                        lang_target + PROBE_START + state_slice + PROBE_END)

                state_tgt_enc = tokenizer(
                    target_list, return_tensors='pt', padding=True, truncation=False,
                    add_special_tokens=True).to(device)
                state_tgt_enc['input_ids'].masked_fill_(state_tgt_enc['input_ids'] == tokenizer.pad_token_id, -100)
                state_tgt_enc['input_ids'].to(device)
                state_tgt_enc['key'] = state_key

            inp_enc['state_key'] = state_key
            yield inp_enc, lang_tgt_enc, state_tgt_enc, state_targets, init_states


def convert_to_transformer_batches_probing(
        dataset, tokenizer, batchsize, domain="alchemy",
        state_targets_type="state.NL", device='cuda',
):
    """
    state_targets_type (str): what to return for `state_tgt_enc` and `state_targets`
        {state|init_state|interm_states|single_beaker_{...}|utt_beaker_{...}{.offset{1|2|...|6}}}.{NL|raw}|text
    """
    batches = list(getBatchesWithInit(dataset, batchsize, get_subsequent_state=True))

    for batch in batches:
        inputs, lang_targets, prev_state_targets, subsequent_state_targets, init_states = zip(*batch)
        init_states = [' '.join(init_state) for init_state in init_states]

        # The target in this case is the next/subsequent state
        state_targets = [
            decide_translate(' '.join(tgt), state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
            for tgt in prev_state_targets]
        state_targets = {'full_state': state_targets}

        # make inputs
        inps = []
        cur_actions = []
        for i, inp in enumerate(inputs):
            action_string = ' '.join(inp).replace(' \n ', '. ')
            all_actions = action_string.split('. ')

            prev_actions = all_actions[:-1]
            prev_action_string = '. '.join(prev_actions)
            cur_action = all_actions[-1]
            cur_actions.append(cur_action)

            steps.append(action_string)
            init_state = translate_states_to_nl(
                init_states[i], domain, isinstance(tokenizer, BartTokenizerFast))
            init_states_str.append(init_state)
            prev_action_string = init_state + '. ' + prev_action_string
            inps.append(prev_action_string)

        inp_enc = tokenizer(
            inps, return_tensors='pt', padding=True, truncation=False,
            return_offsets_mapping=True).to(device)

        inp_enc['original_text'] = inps
        inp_enc['init_state'] = init_states_str
        inp_enc['steps'] = steps

        lang_tgt_enc['original_text'] = lang_targets
        lang_tgt_enc['input_ids'].masked_fill_(lang_tgt_enc['input_ids'] == tokenizer.pad_token_id, -100)
        lang_tgt_enc['input_ids'].to(device)

        yield inp_enc, cur_actions, state_targets
