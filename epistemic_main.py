#

# main decoding

import argparse
import logging
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from qa_data import CsrDoc

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
    parser.add_argument('--input_topic', type=str)  # topic json input
    # specific ones for csr mode!
    # parser.add_argument('--csr_cf_context', type=int, default=5, help='Max number of sentences to count back for author queries')
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
        encoded_inputs = tokenizer(batch, return_tensors='pt', padding = True, truncation=True).to(args.device)
        res = model(**encoded_inputs)
        res_t = torch.nn.functional.softmax(res['logits'], dim=-1)

        res_logits = list(res_t.detach().cpu().numpy())  # List[len, ??]
        tmp_logits.extend(res_logits)
    # --
    # re-sort back
    sorted_logits = [(xx[0], zz) for xx, zz in zip(sorted_insts, tmp_logits)]
    sorted_logits.sort(key=lambda x: x[0])  # resort back
    all_logits = [z[-1] for z in sorted_logits]
    all_ids = np.argmax(all_logits, axis=-1)
    all_scores = np.max(all_logits, axis=-1)
    all_labels = [model.config.id2label[idx] for idx in all_ids]
    return all_labels, all_scores


def categorical_f1(y, pred):
    cats = set(pred)
    scores = dict()
    for cat in cats:
        correct = np.sum(np.logical_and(np.char.equal(y, [cat]), np.char.equal(y, pred)))
        scores[cat] = [correct/np.sum(np.char.equal(pred, [cat])), correct/np.sum(np.char.equal(y, [cat]))]

    scores = {c:2*s[0]*s[1]/(s[0]+s[1]+1e-7) for c,s in scores.items()}
    return scores


def decode_gold(args, tokenizer, model):
    # load dataset
    input_path = Path(args.input_path)
    annotated_frames = input_path.glob('**/claim_frames.tab')
    
    with open(args.input_topic) as fd:
        d_topic = json.load(fd)
    subtopics = d_topic['subtopics']
    subtopic_lookup = {s['subtopic']:k for k,s in subtopics.items()}

    for file in annotated_frames:
        with open(file) as fin:
            header = fin.readline().strip().split('\t')
            try:
                subtopic_col = header.index('subtopic')
                text_col = header.index('description')
                epistemic_col = header.index('epistemic_status')
            except ValueError:
                continue
            frames = []
            for line in fin:
                row = line.strip().split('\t')
                if row[subtopic_col] in subtopic_lookup and row[epistemic_col]!='unknown':
                    frames.append([row[text_col], subtopic_lookup[row[subtopic_col]], row[epistemic_col]])
        if len(frames) > 0:
            frames_dict = {i:[f[0], f[1]] for i,f in enumerate(frames)}
            labels = decode_frames(args, tokenizer, model, frames_dict, subtopics, False)
            labels_tf, labels_certainty = zip(*[l[0].split('-') for l in labels.values()])
            f_tf, f_certainty = zip(*[f[2].split('-') for f in frames])
            logging.info(f'Processed annotation file {file}: found {len(frames)} frames, tf categorical f1 {categorical_f1(f_tf, labels_tf)}, certainty categorical f1 {categorical_f1(f_certainty, labels_certainty)}')



def decode_demo(args, tokenizer, model):
    # --
    # read from inputs
    while True:
        str_question = input("Input a statement: >> ").strip()
        str_context = input("Input another statement: >> ").strip()
        if str_question == '' and str_context == '':
            break
        ipt = [[str_question, str_context]]
        ans, score = batched_forward(args, tokenizer, model, ipt)
        logging.info(f"=> {ans[0]}, score => {score[0]}")
    # --
    logging.info("Finished!")

# =====
# csr related


def decode_csr(args, tokenizer, model):
    logging.info(f"Decode csr: input={args.input_path}, output={args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    csr_files = sorted([z for z in os.listdir(args.input_path) if z.endswith('.csr.json')])

    for fii, fn in enumerate(csr_files):
        try:
            input_path = os.path.join(args.input_path, fn)
            doc = CsrDoc(input_path)
        except Exception:
            logging.warning(f'Error trying to open {input_path}, skipping document')
            continue
        # input_path = os.path.join(args.input_path, fn)
        # doc = CsrDoc(input_path)

        # process it
        cc = decode_one_csr(doc, args, tokenizer, model)
        logging.info(f"Process {doc.doc_id}[{fii}/{len(csr_files)}]: {cc}")
        # --
        doc.write_output(os.path.join(args.output_path, f"{doc.doc_id}.csr.json"))
    # --
    logging.info(f"Finished decoding.")
    # --


def decode_frames(args, tokenizer, model, frames, subtopics, replace_x=True):
    nli2tf = {'ENTAILMENT': 'true', 'NEUTRAL': 'true',  'CONTRADICTION': 'false'}
    nli2certain = {'ENTAILMENT': 'certain', 'NEUTRAL': 'certain',  'CONTRADICTION': 'uncertain'}
    tf2template = {'true': 'pos', 'false': 'neg'}
    if replace_x:
        tf_pairs = [(f[0], subtopics[f[1]]['seqs']['template_pos'].replace('X', f[2])) for f in frames.values()]
    else:
        tf_pairs = [(f[0], subtopics[f[1]]['seqs']['template_pos']) for f in frames.values()]
    tf_labels, tf_scores = batched_forward(args, tokenizer, model, tf_pairs)
    tf_labels = [nli2tf[l] for l in tf_labels]

    if replace_x:
        certainty_pairs = [(f[0], subtopics[f[1]]['seqs'][f'template_{tf2template[label]}_certain'].replace('X', f[2])) for f, label in zip(frames.values(), tf_labels)]
    else:
        certainty_pairs = [(f[0], subtopics[f[1]]['seqs'][f'template_{tf2template[label]}_certain']) for f, label in zip(frames.values(), tf_labels)]
    certainty_labels, certainty_scores = batched_forward(args, tokenizer, model, certainty_pairs)
    certainty_labels = [nli2certain[l] for l in certainty_labels]

    ret = {idx:[f'{tf}-{certain}', s1*s2] for idx, tf, certain, s1, s2 in zip(frames.keys(), tf_labels, certainty_labels, tf_scores, certainty_scores)}
    return ret


def decode_one_csr(doc, args, tokenizer, model):
    cc = defaultdict(int)

    with open(args.input_topic) as fd:
        d_topic = json.load(fd)
    subtopics = d_topic['subtopics']

    frames = dict()
    for cf in doc.cf_frames:
        cf['epistemic_status'] = None
        if cf['subtopic']['id'] not in subtopics:
            logging.warning('Claim frame subtobic not found in list, skipping.')
            continue
        x_ent = doc.id2frame[cf['x']]
        sent = doc.id2sent[x_ent['provenance']['parent_scope']]
        frames[cf['@id']] = [sent.orig_text, cf['subtopic']['id'], cf['x_text']]

    if len(frames) > 0:
        frames_epistemic = decode_frames(args, tokenizer, model, frames, subtopics)
        for idx in frames_epistemic:
            doc.id2frame[idx]['epistemic_status'] = frames_epistemic[idx][0]
            doc.id2frame[idx]['epistemic_score'] = float(frames_epistemic[idx][1])

    cc.update({'sent': len(doc.sents), 'frames': len(frames)})
    return dict(cc)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    args = parse_args()
    # load model
    model = AutoModelForSequenceClassification.from_pretrained("./models/roberta-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("./models/roberta-large-mnli")
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
