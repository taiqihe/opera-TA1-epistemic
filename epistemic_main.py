#

# main decoding

import argparse
import logging
import json
import os
import re
import tqdm
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEP = '</s></s>'

def parse_args():
    parser = argparse.ArgumentParser('Main Decoding')
    # input & output
    parser.add_argument('--mode', type=str, default='csr', choices=['gold', 'csr', 'demo'])
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--model_kwargs', type=str, default='None')  # for example: '{"qa_label_pthr":1.}'
    # more
    parser.add_argument('--device', type=int, default=0)  # gpuid, <0 means cpu
    parser.add_argument('--batch_size', type=int, default=16)
    # specific ones for csr mode!
    parser.add_argument('--csr_query_topic', type=int, default=1)  # query all subtopics under topic?
    parser.add_argument('--csr_ctx_size', type=int, default=8)  # number of sent per chunk?
    parser.add_argument('--csr_ctx_stride', type=int, default=4)  # number of sent between chunks?
    parser.add_argument('--csr_prob_thresh', type=float, default=0.5)  # >this to be valid!
    parser.add_argument('--csr_cf_sratio', type=float, default=0.5)  # max number (ratio*sent_num) per doc
    parser.add_argument('--csr_cf_per_sent', type=int, default=1)  # max number of cf per sent
    parser.add_argument('--csr_cf_context', type=int, default=5, help='Max number of sentences to count back for author queries')
    # --
    args = parser.parse_args()
    logging.info(f"Start decoding with: {args}")
    return args

def batched_forward(args, tokenizer, model, pairs):
    # --
    # sort by length
    sorted_insts = [(ii, zz) for ii, zz in enumerate(pairs)]
    sorted_insts.sort(key=lambda x: len(x[1][0])+len(x[1][1]))
    # --
    bs = args.batch_size
    tmp_logits = []
    for ii in range(0, len(sorted_insts), bs):
        batch = [f'{x} {SEP} {y}' for _,(x,y) in sorted_insts[ii:ii+bs]]
        encoded_inputs = tokenizer(batch, return_tensors='pt', padding = True).to(args.device)
        res = model(**encoded_inputs)
        res_t = res['logits']

        res_logits = list(res_t.detach().cpu().numpy())  # List[len, ??]
        tmp_logits.extend(res_logits)
    # --
    # re-sort back
    sorted_logits = [(xx[0], zz) for xx, zz in zip(sorted_insts, tmp_logits)]
    sorted_logits.sort(key=lambda x: x[0])  # resort back
    all_logits = [z[-1] for z in sorted_logits]
    all_ids = np.argmax(all_logits, axis=-1)
    all_labels = [model.config.id2label[idx] for idx in all_ids]
    return all_labels

def decode_gold(args, tokenizer, model):
    # load dataset
    pass

def decode_demo(args, tokenizer, model):
    # --
    # read from inputs
    while True:
        str_question = input("Input a statement: >> ").strip()
        str_context = input("Input another statement: >> ").strip()
        if str_question == '' and str_context == '':
            break
        ipt = [[str_question, str_context]]
        ans = batched_forward(args, tokenizer, model, ipt)[0]
        logging.info(f"=> {ans}")
    # --
    logging.info("Finished!")

# =====
# csr related

def read_pct(file: str):
    ret = OrderedDict()
    with open(file) as fd:
        fd.readline()  # skip first line
        for line in fd:
            fields = line.rstrip().split("\t")
            parent_uid, child_uid, child_asset_type, topic = [fields[z] for z in [2,3,5,6]]
            if child_asset_type != ".ltf.xml":
                continue
            if child_uid == "n/a":
                continue
            # assert child_uid not in ret
            ret[child_uid] = {
                'parent_uid': parent_uid, 'child_uid': child_uid, 'child_asset_type': child_asset_type, 'topic': topic,
            }
    return ret

def decode_csr(args, model):
    assert model.args.qa_head_type == 'label', "Should use labeling-head in this mode!"

    logging.info(f"Decode csr: input={args.input_path}, output={args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    csr_files = sorted([z for z in os.listdir(args.input_path) if z.endswith('.csr.json')])
    # for fn in tqdm.tqdm(csr_files):
    for fii, fn in enumerate(csr_files):
        try:
            input_path = os.path.join(args.input_path, fn)
            doc = CsrDoc(input_path)
        except Exception:
            logging.warning(f'Error trying to open {input_path}, skipping document')
            continue
        # --
        # process it
        cc = decode_one_csr(doc, args, model)
        logging.info(f"Process {doc.doc_id}[{fii}/{len(csr_files)}]: {cc}")
        # --
        doc.write_output(os.path.join(args.output_path, f"{doc.doc_id}.csr.json"))
    # --
    logging.info(f"Finished decoding.")
    # --

def candidate_sort_key(cand):
    # s0 = 0 if 'event' in cand['@type'] else 1  # prefer event over entity [nope!]
    s1 = - np.average(cand['qa_scores']).item()
    return s1  # only rank by score!

def decode_one_csr(doc, args, model):
    cc = defaultdict(int)
    _limit_q, _limit_full = GR.args_max_query_length, GR.args_max_seq_length

    qas = []
    for cf in doc.cf_frames:
        x_question = re.sub(r'[\w/]*[-]?[xX]', cf['x_text'], cf['subtopic']['template'])
        x_question = 'Who said that ' + x_question[0].lower() + x_question[1:] + '?'
        x_question = TextPiece(x_question)

        x_ent = doc.id2frame[cf['x']]
        x_sent_id = int(re.findall(r'\d+$', x_ent['provenance']['parent_scope'])[0])

        _sid = x_sent_id
        _remaining_budget = _limit_full - min(_limit_q, len(x_question.subtoken_ids)) - 3
        while _remaining_budget > 0 and _sid >= max(0, x_sent_id-args.csr_cf_context):
            _remaining_budget -= len(doc.sents[_sid].subtoken_ids)
            _sid-=1

        x_context = TextPiece.merge_pieces(doc.sents[_sid+1: x_sent_id+1], sent_range=(_sid+1, x_sent_id+1))
        qas.append(QaInstance(x_context, x_question, ''))


    # do forwarding and get all logits
    all_probs = batched_forward(args, model, qas, apply_labeling_prob=True)  # List[(slen, )]
    # --
    # repack and decide outputs (for each question)
    for _cf, _inst, _probs in zip(doc.cf_frames, qas, all_probs):
        # assign scores
        s_start, s_end = _inst.context.info['sent_range']  # sentence range
        _cur_soff = _inst.context_offset  # subtoken offset
        token_scores = dict()
        for _sent in doc.sents[s_start:s_end]:
            _t_probs = np.zeros(len(_sent.tokens))  # current probs
            for _tid in _sent.sub2tid:
                if _cur_soff < len(_probs):  # otherwise, things are truncated and just ignore those!
                    _t_probs[_tid] = max(_t_probs[_tid], _probs[_cur_soff])  # note: maximum for subtok->tok
                _cur_soff += 1
            token_scores[_sent.info['id']] = _t_probs
        
        if _cur_soff < len(_inst.input_ids) and _inst.input_ids[_cur_soff] != GR.sub_tokenizer.sep_token_id:
            logging.warning("Probably internal error!!")

        # rank the events/entities (first for each sents)
        doc_cands = []
        for _sent in doc.sents[s_start:s_end]:
            _sid = _sent.info['id']
            _scores = token_scores[_sid]
            # first get cands
            _cands = []
            for _cols in [doc.cand_entities]:
                for _item in _cols[_sid]:  # check each item
                    _widx, _wlen = _item['tok_posi']
                    if any(z>args.csr_prob_thresh for z in _scores[_widx:_widx+_wlen]):
                        # any token larger than thresh will be fine!
                        _item['qa_scores'] = _scores[_widx:_widx+_wlen]
                        _cands.append(_item)
            # then prune by sent
            cc['cand_init'] += len(_cands)
            _cands_sorted = sorted(_cands, key=candidate_sort_key)
            # --
            sent_cands = []
            for _one_cand in _cands_sorted:  # go through to check no-overlap!
                if len(sent_cands) >= args.csr_cf_per_sent:
                    break
                _overlap = False
                _start1, _length1 = doc.get_provenance_span(_one_cand, False, False)  # get full span!
                for _cand2 in sent_cands:
                    _start2, _length2 = doc.get_provenance_span(_cand2, False, False)  # get full span!
                    if (_start2>=_start1 and _start2<(_start1+_length1)) or (_start1>=_start2 and _start1<(_start2+_length2)):
                        _overlap = True
                        break
                if not _overlap:
                    sent_cands.append(_one_cand)
            # --
            doc_cands.extend(sent_cands)
        cc['cand_sent'] += len(doc_cands)
        final_cands = sorted(doc_cands, key=candidate_sort_key)[:int(np.ceil(args.csr_cf_sratio * len(doc.sents)))]
        cc['cand_final'] += len(final_cands)
        # --
        # put final results
        # first set default value
        _cf['claimer'] = _cf['claimer_score'] = _cf['claimer_text'] = None
        for f_cand in final_cands:
            ff_start, ff_length = doc.get_provenance_span(f_cand)
            # try to find its claiming frame: preferring the smallest ranged one that contains the cand
            best_ce, best_length = None, 100000
            for ce in doc.claim_events[f_cand['provenance']['parent_scope']]:
                ce_start, ce_length = doc.get_provenance_span(ce, False, False)
                if ff_start>=ce_start and (ff_start+ff_length) <= (ce_start+ce_length) and ce_length < best_length:
                    best_ce = ce
                    best_length = ce_length
            # find it?
            cc['cand_finalCE'] += int(best_ce is not None)
            # put it!
            _cf['claimer'] = f_cand['@id']
            _cf['claimer_score'] = np.average(f_cand['qa_scores']).item()
            _cf['claimer_text'] = doc.id2frame[f_cand['@id']]['provenance']['text']
    # --
    cc.update({'sent': len(doc.sents), 'questions': len(qas)})
    return dict(cc)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    args = parse_args()
    # load model
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model.eval()
    device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    model.to(device)
    args.device = device

    logging.info(f"#--\nStart decoding with {args}:\n")
    with torch.no_grad():
        if args.mode == 'csr':
            decode_csr(args, tokenizer, model)
        elif args.mode == 'gold':
            decode_gold(args, tokenizer, model)
        elif args.mode == 'demo':
            decode_demo(args, tokenizer, model)
        else:
            raise NotImplementedError()
    # --

if __name__ == '__main__':
    main()

# --
# decode csr
# python3 qa_main.py --model zmodel.best --input_path csr_in --output_path csr_out
# python3 qa_main.sh csr_in csr_out
# python3 qa_main.py --model zmodel.best --mode demo --input_path '' --output_path ''
