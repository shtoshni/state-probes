
import torch
from data.alchemy.utils import int_to_word, decide_translate, translate_states_to_nl
from data.alchemy.parseScone import getBatchesWithInit
from transformers import BartTokenizerFast, T5TokenizerFast

from data.alchemy.utils import int_to_word

PROBE_START = '[PROBE_START]'
PROBE_END = '[PROBE_END]'


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
    dataset, tokenizer, batchsize, random=None,
    domain="alchemy",
    state_targets_type="state.NL", device='cuda', add_state='random', eval=False,
):
    """
    state_targets_type (str): what to return for `state_tgt_enc` and `state_targets`
        {state|init_state|interm_states|single_beaker_{...}|utt_beaker_{...}{.offset{1|2|...|6}}}.{NL|raw}|text
    """
    state_targets_type_split = state_targets_type.split('.')
    batches = list(getBatchesWithInit(dataset, batchsize, get_subsequent_state=True))

    # print(len(batches))
    if random:
        random.shuffle(batches)

    for batch in batches:
        inputs, lang_targets, prev_state_targets, subsequent_state_targets, init_states = zip(*batch)
        init_states = [' '.join(init_state) for init_state in init_states]

        # make state targets
        if state_targets_type_split[0] == 'state':
            state_targets = [
                decide_translate(' '.join(tgt), state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
                for tgt in prev_state_targets]
            state_targets = {'full_state': state_targets}
            # print(state_targets, len(state_targets['full_state']))
        elif state_targets_type_split[0] == 'init_state':
            state_targets = [
                decide_translate(init_state, state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
                for init_state in init_states]
            state_targets = {'full_state': state_targets}
        elif state_targets_type_split[0].startswith('single_beaker'):
            assert domain == 'alchemy'
            if state_targets_type_split[0].endswith("init"):
                states = init_states
            elif state_targets_type_split[0].endswith("final"):
                states = [' '.join(tgt) for tgt in prev_state_targets]
            else:
                raise NotImplementedError()
            # {bn -> [`bn`-th beaker's state in example i (x # examples for beaker bn)]}
            state_targets = {int(beaker.split(':')[0]) - 1: [] for beaker in states[0].split(' ')}

            for state in states:
                state_descr = decide_translate(state, state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
                if isinstance(tokenizer, BartTokenizerFast):
                    beaker_states = state_descr.split(',')
                else:
                    beaker_states = state_descr.split(', ')
                for bn, beaker_state in enumerate(beaker_states):
                    state_targets[bn].append(beaker_state)
        else:
            assert state_targets_type_split[0] == 'text'

        # make inputs
        inps = []
        for i, inp in enumerate(inputs):
            # string = ' '.join(inp).replace(' \n ', '.\n')
            string = ' '.join(inp).replace(' \n ', '. ')
            string = translate_states_to_nl(
                init_states[i], domain, isinstance(tokenizer, BartTokenizerFast)) + '. ' + string
            inps.append(string)

        if state_targets_type_split[0] == 'text':
            state_targets = {'original_text': inps}

        # make lang targets
        lang_targets_new = []
        for tgt in lang_targets:
            tgt = ' '.join(tgt)  # + '.'
            # if isinstance(tokenizer, T5TokenizerFast) and '  ' in tgt:
            #     tgt = tgt.replace('  ', ' first ')
            lang_targets_new.append(tgt + tokenizer.eos_token)
        lang_targets = lang_targets_new

        # if control_input:
        #     # the first beaker has 1 red, the second beaker has 1 red
        #     inps = [", ".join([f"the {int_to_word[num]} beaker is empty" for num in range(7)]) + "." for _ in inps]

        inp_enc = tokenizer(inps, return_tensors='pt', padding=True, truncation=False, return_offsets_mapping=True).to(device)
        lang_tgt_enc = tokenizer(lang_targets, return_tensors='pt', padding=True, truncation=True,
                                 add_special_tokens=False).to(device)
        inp_enc['original_text'] = inps
        lang_tgt_enc['original_text'] = lang_targets
        lang_tgt_enc['input_ids'].masked_fill_(lang_tgt_enc['input_ids'] == tokenizer.pad_token_id, -100)
        lang_tgt_enc['input_ids'].to(device)
        # print(lang_tgt_enc['input_ids'])
        beaker_targets = identify_beaker_idx(prev_state_targets, subsequent_state_targets)

        for state_key in state_targets:
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
                    PROBE_START + " " + state_slice + " " + PROBE_END + " " + lang_target)

            state_tgt_enc = tokenizer(
                target_list, return_tensors='pt', padding=True, truncation=False, add_special_tokens=False).to(device)
            state_tgt_enc['input_ids'].masked_fill_(state_tgt_enc['input_ids'] == tokenizer.pad_token_id, -100)
            state_tgt_enc['input_ids'].to(device)

            if eval:
                probe_end_token = tokenizer.convert_tokens_to_ids(PROBE_END)
                probe_end_token_idx = (state_tgt_enc['input_ids'] == probe_end_token).nonzero(as_tuple=True)[0].unsqueeze(1)
                batch_size = state_tgt_enc['input_ids'].size()[0]
                max_len = state_tgt_enc['input_ids'].size()[1]
                tmp = torch.arange(max_len, device=state_tgt_enc['input_ids'].device).expand(batch_size, max_len)

                # Mask out input ids before the probing sequence
                # print(tmp <= probe_end_token_idx)
                state_tgt_enc['input_ids'][tmp <= probe_end_token_idx] = -100
                print(state_tgt_enc['input_ids'][0], probe_end_token_idx[0])
                print(probe_end_token_idx)

            state_tgt_enc['key'] = state_key

            inp_enc['state_key'] = state_key
            yield inp_enc, lang_tgt_enc, state_tgt_enc, state_targets, init_states
