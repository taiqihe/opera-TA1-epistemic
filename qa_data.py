#

import os
from typing import List
from collections import defaultdict, Counter
import string
import logging
import torch
import json
import copy
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer
import re

# --
# todo(note): likely to be nltk's bug
class ModifiedTreebankWordTokenizer(TreebankWordTokenizer):
    def span_tokenize(self, sentence):
        raw_tokens = self.tokenize(sentence)
        # convert
        if ('"' in sentence) or ("''" in sentence):
            # Find double quotes and converted quotes
            matched = [m.group() for m in re.finditer(r"``|'{2}|\"", sentence)]
            # Replace converted quotes back to double quotes
            tokens = [matched.pop(0) if tok in ['"', "``", "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        # align_tokens
        point = 0
        offsets = []
        for token in tokens:
            try:
                start = sentence.index(token, point)
            except ValueError as e:
                # raise ValueError(f'substring "{token}" not found in "{sentence}"') from e
                logging.warning(f"Tokenizer skip unfound token: {token} ||| {sentence[point:]}")
                continue  # note: simply skip this one!!
            point = start + len(token)
            offsets.append((start, point))
        return offsets
# --

# --
# word tokenizer
class NTokenizer:
    def __init__(self):
        self.word_toker = ModifiedTreebankWordTokenizer()
        self.sent_toker = PunktSentenceTokenizer()
        # --

    def tokenize(self, text: str):
        # first split sent
        sent_spans = list(self.sent_toker.span_tokenize(text))
        sents = [text[a:b] for a,b in sent_spans]
        # then split tokens
        char2posi = [None] * len(text)  # [L_char]
        all_tokens = []  # [L_tok]
        all_token_spans = []  # [L_tok]
        mark_eos = []  # [L_tok]
        for sid, sent in enumerate(sents):
            if len(mark_eos) > 0:
                mark_eos[-1] = True
            # --
            tok_spans = list(self.word_toker.span_tokenize(sent))
            tokens = [sent[a:b] for a,b in tok_spans]
            for ii, (a, b) in enumerate(tok_spans):
                _offset = sent_spans[sid][0]
                _s0, _s1 = _offset+a, _offset+b
                char2posi[_s0:_s1] = [len(all_tokens)] * (b - a)
                all_tokens.append(tokens[ii])
                all_token_spans.append((_s0, _s1))
                mark_eos.append(False)
        if len(mark_eos) > 0:
            mark_eos[-1] = True
        # return all_tokens, all_token_spans, char2posi, mark_eos
        return all_tokens, all_token_spans


class TextPiece:
    def __init__(self, text: str, **info_kwargs):
        self.orig_text = text

    def __repr__(self):
        return f"Text({self.orig_text[:100]}" + ('...' if len(self.orig_text)>100 else '') + ')'


# --
# csr related

class CsrDoc:
    def __init__(self, csr_path: str):
        # read csr
        with open(csr_path) as fd:
            json_doc = json.load(fd)
        self.orig_doc = copy.deepcopy(json_doc)  # keep a deep-copy for output
        # --
        # parse it!
        self.doc_id = os.path.basename(csr_path)
        self.frame_id_infix = ""
        _csr_suffix = ".csr.json"
        if self.doc_id.endswith(_csr_suffix):
            self.doc_id = self.doc_id[:-len(_csr_suffix)]
        # --
        # record all frames
        _id2frame = {}  # @id -> one
        for ff in json_doc['frames']:
            if ff['@id'] in _id2frame:
                logging.warning(f"Ignoring repeated @id frame: {ff}")  # this should not happen!
                continue
            if ff['@type'] == 'document' and ff['@id'] != f"data:{self.doc_id}":  # check docid
                logging.warning(f"Mismatched doc-id: {self.doc_id} vs {ff}")
            if ff['@type'] == 'sentence' and not self.frame_id_infix:  # find an infix
                self.frame_id_infix = '-'.join(ff['@id'].split('-')[1:-1])  # "{doc_id}-text-cmu-{time_stamp}"
            _id2frame[ff['@id']] = ff
        # --
        # parse sentences
        _sents = []  # List[TextPiece]
        _id2sent = {}  # @id -> TextPiece
        for ff in json_doc['frames']:
            if ff['@type'] == 'sentence':
                s = TextPiece(text=ff['provenance']['text'])
                _sents.append(s)
                _id2sent[ff['@id']] = s
        # --
        # prepare useful entities and events
        self.claim_events = defaultdict(list)  # sid -> claim events
        self.cand_events = defaultdict(list)  # sid -> other candidate events
        self.cand_entities = defaultdict(list)  # sid -> candidate entities
        # self.cand_claimers = defaultdict(list)  # sid -> candidate claimers
        self.cf_frames = []  # cf-frames
        failed_spans = Counter()
        for ff in json_doc['frames']:
            if ff['@type'] in ['entity_evidence', 'event_evidence']:
                _provenance = ff['provenance']

                if ff['@type'] == 'entity_evidence':
                    self.cand_entities[_provenance['parent_scope']].append(ff)
                else:
                    if ff['interp'].get('info', {}).get('sip', False):
                        self.claim_events[_provenance['parent_scope']].append(ff)
                    else:
                        self.cand_events[_provenance['parent_scope']].append(ff)
                # --
            elif ff['@type'] == 'claim_frame_evidence':
                self.cf_frames.append(ff)
        if len(failed_spans) > 0:
            logging.warning(f"Cannot find head tok_posi for {self.doc_id}: {failed_spans}")
        # --
        # remember these
        self.id2frame = _id2frame
        self.sents = _sents
        self.id2sent = _id2sent
        # --

    def get_provenance_span(self, ff, try_base=True, try_head=True):
        _provenance = ff['provenance']
        if try_base and 'base_provenance' in _provenance:
            _provenance = _provenance['base_provenance']
        if try_head:
            if 'head_span_start' in _provenance and 'head_span_length' in _provenance:  # use head if possible
                return _provenance['head_span_start'], _provenance['head_span_length']
            else:
                return None
        else:
            return _provenance['start'], _provenance['length']

    def add_cf(self, subtopic: dict, x: str, x_score: float, claim_evt: str):
        # here simply add id
        _id = f"data:cf-{self.frame_id_infix}-{len(self.cf_frames)}"
        # some checking
        # assert self.id2frame[x]['@type'] in ['entity_evidence', 'event_evidence']
        # assert claim_evt is None or self.id2frame[claim_evt]['@type']=='event_evidence' \
        #        and self.id2frame[claim_evt]['interp']['info']['sip']
        # --
        # also include text for easier checking
        x_text = self.id2frame[x]['provenance']['text']
        claim_evt_text = None if claim_evt is None else self.id2frame[claim_evt]['provenance']['text']
        _subtopic = {'@type': 'subtopic'}
        _subtopic.update({k:v for k,v in subtopic.items() if k not in ['parse', 'seqs']})  # don't be too verbose!
        res = {
            '@id': _id, '@type': 'claim_frame_evidence', 'component': 'opera.cf.qa',
            'subtopic': _subtopic, 'x': x, 'x_score': x_score, 'x_text': x_text,
            'claim_evt': claim_evt, 'claim_evt_text': claim_evt_text,
        }
        self.cf_frames.append(res)
        # --

    def write_output(self, output_path: str):
        with open(output_path, 'w') as fd:
            res = copy.deepcopy(self.orig_doc)
            frames = [ff for ff in res['frames'] if ff['@type'] != 'claim_frame_evidence']
            frames.extend(self.cf_frames)
            res['frames'] = frames
            json.dump(res, fd, indent=2, ensure_ascii=False)
        # --
