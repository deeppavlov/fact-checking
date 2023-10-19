# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import string
import decimal
import sqlite3
import datetime
import dateutil.parser
from logging import getLogger
from typing import List, Dict, Tuple, Union, Any

import spacy
from rdflib import Graph
from rdflib_hdt import HDTStore
from hdt import HDTDocument
from rapidfuzz import fuzz

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

from .correct_sentence import correct_existence_sent, check_negative_status

log = getLogger(__name__)


@register("fact_checker_kb")
class FactCheckerKB(Component):
    """
    Class for checking if the extracted triplets are factually correct by matching them to DBpedia
    """

    def __init__(
        self,
        hdt_kb_path: str,
        entity_sql_path: str,
        long_rels_path: str,
        prefixes: List[str] = [],
        **kwargs,
    ) -> None:

        self.nlp = spacy.load("en_core_web_sm")
        self.prefixes = prefixes
        self.hdt_kb_path = os.path.expanduser(hdt_kb_path)
        self.entity_sql_path = os.path.expanduser(entity_sql_path)
        self.long_rels_path = os.path.expanduser(long_rels_path)
        self.regex_num = re.compile(r"^[\d]+([\.-:][\d]+)*$")
        self.regex_num_float = re.compile(r"([\d]+(\.?[\d]+)*)")

        self.default_date = datetime.datetime(3050, 1, 1, 0, 0)
        self.rels_for_literals_en = [
            "leaderName",
            "leaderTitle",
            "governmentType",
            "abbreviation",
            "nationality",
        ]
        self.rels_only_integer = ["selection"]
        self.rels_similar = {
            "placeOfBirth": "birthPlace",
            "placeOfDeath": "deathPlace",
            "deathDate": "dateOfDeath",
            "birthDate": "dateOfBirth",
        }
        self.load()

    def load(self):
        self.graph = Graph(store=HDTStore(self.hdt_kb_path))
        self.kb = HDTDocument(self.hdt_kb_path)
        conn = sqlite3.connect(expand_path(self.entity_sql_path), check_same_thread=False)
        self.cur = conn.cursor()

        with open(expand_path(self.long_rels_path), "r") as f:
            self.long_rels = {
                json.loads(line).split("/")[-1]: json.loads(line) for line in f
            }

        with open("/src/models/prepositions.txt", "r") as f:
            self.prepositions = [line.strip() for line in f.readlines()]

    def __call__(self, triplets_pred_batch, sentences_batch):
        pred_labels_batch, sents_new_batch = [], []
        for (triplets_pred_top_n, sent) in zip(triplets_pred_batch, sentences_batch):
            try:
                final_label = False
                final_sent_new = sent_new = ""
                for triplets_pred in triplets_pred_top_n:
                    pred_list = []
                    if all([triplet[0] != "unk" for triplet in triplets_pred]):
                        for triplet in triplets_pred:
                            pred, sent_new = self.check_factoid_fuzzy(
                                triplet[0], triplet[1], triplet[2], sent
                            )
                            pred_list.append(pred)
                        pred_label = bool(set(pred_list) == set([True]))
                    elif len(triplets_pred) > 1 and any(
                        [
                            (triplet[0] != "unk" and triplet[2] == "unk")
                            for triplet in triplets_pred
                        ]
                    ):
                        pred_label, sent_new = self.iterative_sparql(
                            triplets_pred, sent
                        )
                    else:
                        sparql, literals = self.make_sparql(triplets_pred, sent)
                        sent_new = sent
                        if literals:
                            pred_label = self.execute_sparql_select(sparql, literals)
                        else:
                            pred_label = self.execute_sparql_ask(sparql)
                        if re.search(r"(not|cannot|\w+n't)", sent.lower()):
                            pred_label = not pred_label
                    final_label = final_label or pred_label
                    if sent_new != sent:
                        final_sent_new = sent_new
                    if len(triplets_pred) == 1 and triplets_pred[0][2] == "unk":
                        sent_new = correct_existence_sent(sent)
                if bool(re.search(r"\b(i wish|if only|i imagined)\b", sent.lower())):
                    final_label = not final_label
            except Exception as e:
                final_label = False
                final_sent_new = sent
                log.warning(f"{e} - sent {sent}; triplets {triplets_pred_top_n}")
            pred_labels_batch.append(final_label)
            sents_new_batch.append(final_sent_new)
        return pred_labels_batch, sents_new_batch

    def link_entity(self, entity):
        query = "SELECT label FROM inverted_index WHERE title MATCH ?;"
        entities = []
        ent_copy = entity.strip(" .,'\"")
        for p in string.punctuation:
            ent_copy = ent_copy.replace(p, "")

        try:
            res = self.cur.execute(query, (ent_copy,))
            entities = res.fetchall()
        except Exception as e:
            log.warning(f"error {e} in linking the entity: {ent_copy}")
        if entities:
            entities_cleaned = [
                ent[0].split("/")[-1].replace("_", " ") for ent in entities
            ]
            ents_with_scores = [
                (ent_cand, fuzz.ratio(ent_cand, entity))
                for ent_cand in entities_cleaned
            ]
            ents_with_scores_top = sorted(
                ents_with_scores, key=lambda x: x[1], reverse=True
            )[:3]
            if ents_with_scores_top[0][1] >= 80.0:
                top_ent = ents_with_scores_top[0][0].replace(" ", "_")
                return f"http://dbpedia.org/resource/{top_ent}"
        entity = "_".join(
            [
                word.capitalize()
                if (word.islower() and not word in self.prepositions)
                else word
                for word in entity.split()
            ]
        )
        if re.search(r"\.[A-Z]$", entity):
            entity += "."
        return f"http://dbpedia.org/resource/{entity}"

    def make_sparql(self, triplets_pred, sent):
        sparql_skeleton = """
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX dbtype: <http://dbpedia.org/datatype/>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
            ASK {  
        """
        filters = []
        minutes = "minute" in sent.lower()
        for idx, triplet in enumerate(triplets_pred):
            sub, rel, obj = triplet
            if sub.lower() == "unk":
                sub_uri = "?unk"
            else:
                sub = sub.replace(" ", "_")
                sub_uri = f"<http://dbpedia.org/resource/{sub}>"

            if obj.lower() == "unk":
                obj_uri = "?unk"
            elif FactCheckerKB.is_date(obj) and not rel in self.rels_only_integer:
                date = str(
                    dateutil.parser.parse(obj, default=self.default_date)
                ).split()[0]
                obj_uri = f'"{date}"^^xsd:date'
            elif self.regex_num.search(obj):
                if minutes:
                    obj_uri = f"'{obj}'^^dbtype:minute"
                elif "." in obj:
                    obj_uri = f"?number{idx}"
                    filters += [
                        f"FILTER( ?number{idx} in ('{obj}'^^xsd:double, '{obj}'^^xsd:float))"
                    ]
                else:
                    obj_uri = obj
            elif FactCheckerKB.check_literal(obj) or rel in self.rels_for_literals_en:
                obj_uri = f"'{obj}'"
            else:
                obj_uri = self.link_entity(obj)
                obj_uri = f"<{obj_uri}>"

            rel = self.rels_similar.get(rel, rel)
            rel_uri = f"dbp:{rel}"
            if rel in self.long_rels:
                rel_uri += f"|<{self.long_rels[rel]}>"
            sparql_skeleton += f"{sub_uri} {rel_uri} {obj_uri} .\n"
        sparql_skeleton = sparql_skeleton + "\n".join(filters) + "}"
        return sparql_skeleton, []

    def search_triplet_obj(self, sub, rel, obj):
        rel = self.rels_similar.get(rel, rel)
        sub = sub.replace(" ", "_")
        sub_uri = f"http://dbpedia.org/resource/{sub}"
        rel_uri = f"http://dbpedia.org/property/{rel}"
        res, _ = self.kb.search_triples(sub_uri, rel_uri, "")
        res_long = []
        if rel in self.long_rels:
            res_long, _ = self.kb.search_triples(sub_uri, self.long_rels[rel], "")
        total_triplets = list(res) + list(res_long)
        return total_triplets

    def iterative_sparql(self, triplets_pred, sent):
        start_triplets = [triplet for triplet in triplets_pred if triplet[2] == "unk"]
        full_triplets = [
            triplet
            for triplet in triplets_pred
            if (triplet[0] != "unk" and triplet[2] != "unk")
        ]
        full_label = [True]
        for triplet in full_triplets:
            sub, rel, obj = triplet
            lbl, _ = self.check_factoid_fuzzy(sub, rel, obj, sent)
            full_label.append(lbl)

        sub, rel, obj = start_triplets[0]
        negative_status = check_negative_status(sub, rel, obj, sent)
        res_triplets = self.search_triplet_obj(sub, rel, obj)
        unk_objs = set()
        for triplet in res_triplets:
            rel_found = triplet[1].split("/")[-1]
            if rel_found == rel:
                obj_found = triplet[2].replace("_", " ")
                for prefix in self.prefixes:
                    obj_found = obj_found.replace(prefix, "")
                obj_found = re.sub(r'(\'|")(@en)?$', "", obj_found)
                obj_found = obj_found.strip()
                unk_objs.add(obj_found)

        pred_labels_total = []
        for triplet in triplets_pred:
            if triplet[0] == "unk":
                pred_labels_per_obj = []
                for unk_obj in unk_objs:
                    sub, rel, obj = triplet
                    sub = unk_obj
                    pred_label, _ = self.check_factoid_fuzzy(sub, rel, obj, sent)
                    pred_labels_per_obj.append(pred_label)
                if len(set(pred_labels_per_obj)) > 1:
                    pred_labels_total.append(True)
                elif pred_labels_per_obj:
                    pred_labels_total.append(pred_labels_per_obj[0])
        pred_label_final = bool(set(pred_labels_total) == set([True])) and bool(
            set(full_label) == set([True])
        )
        if negative_status:
            return not pred_label_final, sent
        return pred_label_final, sent

    def execute_sparql_select(self, sparql, literals_dict):
        try:
            res = self.graph.query(sparql)
        except Exception as e:
            log.warning(f"error {e} in executing sparql {sparql}")
            return False
        else:
            for lit_uri in literals_dict:
                check = False
                literals_dict[lit_uri]["gold"] = []
                pred = literals_dict[lit_uri]["pred"]
                if FactCheckerKB.is_date(pred):
                    pred = str(dateutil.parser.parse(pred, default=default_date)).split(
                        " "
                    )[0]
                for cand in res:
                    cand = str(cand.asdict()[lit_uri.strip("?")].toPython())
                    if FactCheckerKB.is_date(cand) and FactCheckerKB.is_date(pred):
                        cand = str(
                            dateutil.parser.parse(cand, default=default_date)
                        ).split(" ")[0]
                        if cand == pred:
                            check = True
                            break
                    else:
                        cand = self.regex_num_float.findall(cand)
                        pred = self.regex_num_float.findall(pred)[0][0]
                        if cand:
                            cand = cand[0][0].replace(".", "")
                            pred = pred.replace(".", "")
                            if FactCheckerKB.check_nums(float(cand), float(pred)):
                                check = True
                                break
                if not check:
                    return False
        return check

    def execute_sparql_ask(self, sparql_query):
        try:
            qres = self.graph.query(sparql_query)
        except Exception as e:
            log.warning(f"error {e} in executing sparql {sparql_query}")
            return False
        return list(qres)[0]

    def check_factoid_fuzzy(self, sub, rel, obj, sent):
        total_triplets = self.search_triplet_obj(sub, rel, obj)
        negative_status = check_negative_status(sub, rel, obj, sent)
        for triplet in total_triplets:
            rel_found = triplet[1].split("/")[-1]
            obj_found = triplet[2].replace("_", " ")
            for prefix in self.prefixes:
                obj_found = obj_found.replace(prefix, "")
            obj_found = re.sub(r"\(.*?\)$", "", obj_found)
            obj_found = obj_found.strip("\"'@en ")
            if rel_found == rel:
                match_status = (
                    fuzz.ratio(obj.lower(), obj_found.lower()) >= 80.0 or obj == "unk"
                )
                if "XMLSchema" in obj_found or "datatype/" in obj_found:
                    obj_found = obj_found.split("^^")[0].strip("\"'")
                    if "E" in obj_found:
                        obj_found = re.sub(
                            r"E(\d+)$", lambda m: int(m.group(1)) * "0", obj_found
                        )
                    if FactCheckerKB.is_date(obj) and FactCheckerKB.is_date(obj_found):
                        obj_found_date = str(
                            dateutil.parser.parse(obj_found, default=self.default_date)
                        ).split(" ")[0]
                        obj_date = str(
                            dateutil.parser.parse(obj, default=self.default_date)
                        ).split(" ")[0]
                        match_status = obj_found_date == obj_date
                    else:
                        obj_found_num = self.regex_num_float.findall(obj_found)
                        obj_num = self.regex_num_float.findall(obj)
                        if obj_num and obj_found_num:
                            obj_found_num = obj_found_num[0][0].replace(".", "")
                            obj_num = obj_num[0][0].replace(".", "")
                            match_status = FactCheckerKB.check_nums(
                                float(obj_found_num), float(obj_num)
                            )
                sent = sent.replace(obj, obj_found)
                if match_status:
                    return True ^ negative_status, sent
        return False ^ negative_status, sent

    @staticmethod
    def check_literal(entity):
        return (
            len(entity) < 10
            and sum([l.isalpha() for l in entity]) < 4
            and sum([l.isdigit() for l in entity]) > 2
        )

    @staticmethod
    def float_to_str(float_num):
        """
        Convert the given float to a string,
        without resorting to scientific notation
        """
        ctx = decimal.Context()
        ctx.prec = 20
        decimal_ = ctx.create_decimal(repr(float_num))
        return format(decimal_, "f")

    @staticmethod
    def is_date(entity):
        if (entity.isdigit() and not 1800 < int(entity) < 2030) or re.search(
            r"^\d+\.\d+$", entity
        ):
            return False
        try:
            dateutil.parser.parse(entity)
            return True
        except:
            return False

    @staticmethod
    def check_nums(num_gold, num_pred):
        num1_len = len(re.sub(r"\.\d+", "", FactCheckerKB.float_to_str(num_gold)))
        num2_len = len(re.sub(r"\.\d+", "", FactCheckerKB.float_to_str(num_pred)))
        if num1_len > num2_len:
            num_pred = num_pred * 10 ** (num1_len - num2_len)
        if num2_len > num1_len:
            num_gold = num_gold * 10 ** (num2_len - num1_len)
        diff = abs(num_gold - num_pred)
        return diff < num_gold * 0.05
