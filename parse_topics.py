#

# parse the "topic_list.txt" file

from typing import List
import sys
import string
import logging
import json
import stanza
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

class StanzaParser:
    def __init__(self, stanza_dir: str):
        common_kwargs = {"lang": 'en', "use_gpu": False, 'dir': stanza_dir}
        self.parser = stanza.Pipeline(
            processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, **common_kwargs)

    def parse(self, tokens: List[str]):
        res = self.parser([tokens])
        words = res.sentences[0].words
        if len(words) != len(tokens):
            logging.warning(f"Sent length mismatch: {len(words)} vs {len(tokens)}")
        ret = {
            'text': [w.text for w in words],
            'lemma': [w.lemma for w in words], 'upos': [w.upos for w in words],
            'head': [w.head for w in words], 'deprel': [w.deprel for w in words],
        }
        return ret

def read_tab_file(file: str):
    with open(file) as fd:
        headline = fd.readline()
        head_fields = [z.lower() for z in headline.rstrip().split("\t")]
        ret = []
        for line in fd:
            fields = line.rstrip().split("\t")
            # assert len(fields) == len(head_fields)
            ret.append({k: v for k,v in zip(head_fields, fields)})
        return ret

class TemplateParser:
    def __init__(self, stanza_dir, cerntain_probe='definitely'):
        self.word_toker = TreebankWordTokenizer()
        self.parser = StanzaParser(stanza_dir)
        self.cerntain_probe = cerntain_probe

    def parse_template(self, template: str, quiet=False):
        # step 1: normalize text
        raw_tokens = self.word_toker.tokenize(template)
        x_widx = None
        x_mod = None
        normed_tokens = []
        for tok in raw_tokens:
            if tok.lower() == 'x' or tok.lower().endswith("-x"):
                # if tok.lower() == 'x':
                #     _toks = ['X']
                # else:
                #     _toks = [tok[:-2].split("/")[0], 'X']
                _toks = ['X']

                if x_widx is not None:
                    logging.warning(f"Hit multiple Xs, ignore the later one: {raw_tokens}")
                else:
                    x_widx = len(normed_tokens) + len(_toks) - 1
                    if tok.lower().endswith("-x"):
                        x_mod = _toks[0]
            elif tok == '/':  # by itself
                _toks = ['or']
            else:
                _toks = [tok]
            normed_tokens.extend(_toks)
        # step 2: parse it!
        if normed_tokens[-1] != '.':  # add last dot if there are not
            normed_tokens = normed_tokens + ["."]
        sent = self.parser.parse(normed_tokens)  # parse this one!

        # step 3: more templates with simple negation and cerntainty probe
        more_templates = self.create_more_templates(sent)
        # --
        if not quiet:
            logging.info(f"#-- Parse template: {template} ||| {hint}\n"
                 f"=>raw={raw_tokens}\n=>norm={normed_tokens}")
        # if debug:
        #     breakpoint()
        return sent, more_templates

    def simple_negation(self, tokens):
        # todo: really bad replacements here ...
        REPL_MAP0 = {
            "can": "cannot", "may": "may not", "might": "might not",
            "is": "is n't", "was": "was n't", "are": "are n't", "were": "were n't", "will": "wo n't",
            "did": "did n't", "does": "does n't", "do": "do n't",
        }
        if any(t in REPL_MAP0 for t in tokens):  # simply negate the aux verb is fine
            ret = []
            hit_flag = False
            for t in tokens:
                if not hit_flag and t in REPL_MAP0:
                    ret.extend(REPL_MAP0[t].split())
                else:
                    ret.append(t)
        else:  # need to parse
            # find root verb
            parse_res = self.parser.parse(tokens)
            root_widx = [widx for widx, (head, deprel) in enumerate(zip(parse_res['head'], parse_res['deprel']))
                         if head==0 and deprel=='root']
            if len(root_widx)==0 or parse_res['upos'][root_widx[0]] != 'VERB':
                all_verb_widxes = [widx for widx, upos in enumerate(parse_res['upos']) if upos=='VERB']
                if len(all_verb_widxes) > 0:
                    center_verb_widx = all_verb_widxes[0]
                    logging.warning(f"Cannot find center verb, fall back to VERBs: {parse_res}")
                else:  # even cannot find verb??
                    center_verb_widx = None
                    logging.warning(f"Cannot find any verbs, simply add 'No': {parse_res}")
                    return ["No"] + parse_res['text']
            else:
                center_verb_widx = root_widx[0]
            # negate for center-verb
            center_verb_conjs = [widx for widx, (upos, head, deprel) in enumerate(zip(parse_res['upos'], parse_res['head'], parse_res['deprel'])) if head==(1+center_verb_widx) and deprel=='conj' and upos=='VERB']
            ret = []
            for ii, t in enumerate(parse_res['text']):
                if ii == center_verb_widx:  # add neg & put lemma
                    cur_lemma = parse_res['lemma'][ii]
                    if t == cur_lemma:
                        _extra = ["do", "n't"]
                    elif t == cur_lemma + 's' or t == cur_lemma + 'es' or t in ['has']:  # special ones!
                        _extra = ["does", "n't"]
                    else:
                        _extra = ["did", "n't"]
                    ret.extend(_extra)
                    ret.append(parse_res['lemma'][ii])
                elif ii in center_verb_conjs:  # put lemma
                    ret.append(parse_res['lemma'][ii])
                else:  # otherwise simply append
                    ret.append(t)
        # --
        return ret

    def find_root(self, parse):
        acceptable_roots = {'VERB', 'AUX', 'PART'}
        root = parse['head'].index(0)
        if parse['upos'][root] not in acceptable_roots:
            new_root = -1
            for i in range(len(parse['text'])):
                if parse['head'][i] == root+1 and parse['upos'][i] in acceptable_roots and parse['deprel'][i] not in {'csubj'}:
                    new_root = i
                    break
            if new_root == -1:
                logging.warning(f'No acceptable roots found, using default root position 1: {parse}')
                new_root = 1
            root = new_root
        for i in range(root, -1, -1):
            if parse['upos'][i] in acceptable_roots:
                root = i
            else:
                break
        return root


    def add_cerntainty(self, sent):
        parse = self.parser.parse(sent)
        root = self.find_root(parse)
        insert_pos = root
        if parse['lemma'][root] == 'be' and parse['upos'][root+1] != 'PART':
            insert_pos = root+1
        sent = sent.copy()
        sent.insert(insert_pos, self.cerntain_probe)
        return sent

    def create_more_templates(self, sent):
        ret = {
            "template_pos": sent['text'],
            "template_neg": self.simple_negation(sent['text']),
            "template_pos_certain": self.add_cerntainty(sent['text']),
            "template_neg_certain": self.add_cerntainty(self.simple_negation(sent['text'])),
        }
        return ret

    def get_chs_lists(self, cur_heads):
        chs = [[] for _ in range(len(cur_heads) + 1)]
        for m, h in enumerate(cur_heads):  # note: already sorted in left-to-right
            chs[h].append(m)  # note: key is hidx, value is midx
        return chs

    def get_ranges(self, cur_heads):
        ranges = [[z, z] for z in range(len(cur_heads))]
        for m in range(len(cur_heads)):
            cur_idx = m
            while cur_idx >= 0:
                ranges[cur_idx][0] = min(m, ranges[cur_idx][0])
                ranges[cur_idx][1] = max(m, ranges[cur_idx][1])
                cur_idx = cur_heads[cur_idx] - 1  # offset -1
        return ranges



# note: special post-processing!
def postprocess_tokens(toks):
    ts = []
    for t in toks:
        if t.lower() in [z.lower() for z in ["SARS-CoV-2", "COVID-19", "virus"]]:
            ts.extend(f"{t} or coronavirus".split())
        else:
            ts.append(t)
    return ts

def main(input_file='', output_file='', stanza_dir='./stanza_resources'):
    # stanza.download('en', model_dir=stanza_dir)
    # --
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # --
    parser = TemplateParser(stanza_dir)
    final_res = {"topics": {}, "subtopics": {}}
    tabs = read_tab_file(input_file)
    detoker = TreebankWordDetokenizer()
    for v in tabs:
        sent_parse, seqs = parser.parse_template(v['template'], quiet=True)
        
        # postprocessing
        seqs = {k: postprocess_tokens(v) for k,v in seqs.items()}
        # --
        v['seqs'] = {k: detoker.detokenize(v) for k,v in seqs.items()}
        # v['parse'] = sent_parse
        # --
        final_res['subtopics'][v['id']] = v
        if v['topic'] not in final_res['topics']:
            final_res['topics'][v['topic']] = []
        final_res['topics'][v['topic']].append(v['id'])
    if output_file:
        with open(output_file, 'w') as fd:
            json.dump(final_res, fd, indent=4)
    # --

# python3 parse_topics.py
if __name__ == '__main__':
    main(*sys.argv[1:])