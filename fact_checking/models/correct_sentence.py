import re
import spacy
import pyinflect
from .claucy import add_to_pipe

nlp = spacy.load("en_core_web_sm")
add_to_pipe(nlp)

tag2tense = {
    "VBD": "past",
    "VBN": "past",
    "VBZ": "present",
    "VBP": "present",
    "VB": "present",
}
affirmatives = ["Yes", "Yeah", "Yep"]


def get_clauses(sent):
    doc = nlp(sent)
    new_sents = []
    for clause in doc._.clauses:
        new_sent = ""
        for el in str(clause).split():
            el = el.strip(",")
            if el not in ["None"] and el.isalnum():
                new_sent += f"{el} "
        new_sents.append(new_sent)

    return new_sents


def find_negation_triplets(sent):
    clause_sents = get_clauses(sent)
    if len(set(clause_sents)) == 1:
        clause_sents = [sent]
    subj_obj_list, subj_obj_list_str = [], []
    for sent in set(clause_sents):
        doc = nlp(sent)
        for tok in doc:
            obj = ""
            prep = []
            if tok.dep_ == "neg" and tok.head.pos_ in ["VERB", "AUX"]:
                negative_verb = tok.head
                prep = [
                    c for c in negative_verb.children if c.dep_ in ["prep", "agent"]
                ]
                subj = [
                    c
                    for c in negative_verb.children
                    if c.dep_ in ["nsubj", "nsubjpass"]
                ]
                if tok.head.pos_ == "AUX":
                    obj = [c for c in negative_verb.children if c.dep_ in ["attr"]]
                if prep and not obj:
                    obj = [c for c in prep[0].children if c.dep_ in ["pobj"]]
                elif not obj:
                    obj = [
                        c
                        for c in negative_verb.children
                        if c.dep_ in ["dobj", "iobj", "acomp"]
                    ]
                subj_obj_list.append((subj, obj))

    if not subj_obj_list:
        return []

    for subj, obj in subj_obj_list:
        if subj:
            subj = subj[0]
            subj_compounds = [(c.text, c.idx) for c in subj.subtree]
            subj_compounds_sorted = sorted(subj_compounds, key=lambda x: x[1])
            subj_final = " ".join([text for text, pos in subj_compounds_sorted])
        else:
            subj_final = ""
        if obj:
            obj = obj[0]
            obj_len = len(obj) + 1
            compounds = [
                (child.text, child.idx)
                for child in obj.subtree
                if child.dep_ in ["pobj", "dobj", "compound", "nummod", "iobj"]
            ]
            compounds += [(obj.text, obj.idx)]
            compounds_sorted = sorted(compounds, key=lambda x: x[1])
            obj_final = " ".join([text for text, pos in compounds_sorted])
        else:
            obj_final = ""
        subj_obj_list_str.append([subj_final, obj_final])
    return subj_obj_list_str


def check_negative_status(sub, rel, obj, sent):
    negation_triplets = find_negation_triplets(sent)
    negative_status = False
    for triplet_list in negation_triplets:
        neg_sub, neg_obj = triplet_list
        if set(f"{sub} {rel} {obj}".lower().split()).intersection(
            set(f"{neg_sub} {neg_obj}".lower().split())
        ):
            negative_status = True
    return negative_status


def correct_contractions(sent):
    mapping = {"s": "is", "ve": "have", "d": "had", "re": "are"}
    return re.sub(
        r"\b([a-zA-Z]+)\'(s|ve|re|d)",
        lambda m: f"{m.group(1)} {mapping[m.group(2)]}",
        sent,
    )


def correct_existence_sent(sent):
    if any([sent.startswith(wh) for wh in ["What", "When", "Who", "Why", "Where"]]):
        return sent

    sent = re.sub(r"([Yy]e[ps]|[Yy]eah)", "No", sent)
    sent = correct_contractions(sent)
    try:
        last_clause = get_clauses(sent)[-1]
    except:
        last_clause = sent
    doc = nlp(last_clause)

    root_verb = [
        tok for tok in doc if tok.dep_ == "ROOT" and tok.pos_ in ["VERB", "AUX"]
    ]
    if not root_verb:
        root_verb = [tok for tok in doc if tok.pos_ in ["VERB", "AUX"]]
    if not root_verb:
        return sent

    root_verb = root_verb[0]
    new_root = old_root = root_verb.text
    if " not " in sent.lower():
        aux = [
            child
            for child in root_verb.children
            if child.pos_ == "AUX" and child != root_verb
        ]
        if aux:
            tense = tag2tense[aux[0].tag_]
            remove = aux[0].text + " not "
            if tense == "past":
                new_root = root_verb._.inflect("VBD")
            return sent.replace(remove, "").replace(old_root, new_root)

        else:
            new_root = root_verb._.inflect("VBZ")
            return sent.replace(" not ", " ")

    if root_verb.pos_ == "AUX" or any(
        [exis in sent.lower() for exis in ["there was", "there is"]]
    ):
        return sent.replace(f" {old_root} ", f" {old_root} not ")
    root_tense = tag2tense[root_verb.tag_]
    new_root = root_verb._.inflect("VBP")
    subj = [child for child in root_verb.children if child.dep_ == "nsubj"]
    if root_tense == "past":
        add = f"did not {new_root}"
    elif (
        subj
        and subj[0].tag_ in ["NNPS", "NNS"]
        or subj[0].text.lower() in ["i", "you", "we", "they"]
    ):
        add = f"do not {new_root}"
    else:
        add = f"does not {new_root}"
    return sent.replace(old_root, add)
