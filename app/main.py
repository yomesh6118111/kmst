from matplotlib import pyplot as plt
import spacy
import networkx as nx
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch

import wikipedia

# scraping of web articles
from newspaper import Article, ArticleException

# google news scraping
from GoogleNews import GoogleNews

# graph visualization
from pyvis.network import Network

# to show HTML in notebook
import IPython

# Load the spaCy English language model
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

# from https://huggingface.co/Babelscape/rebel-large
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

class KB():
    def __init__(self):
        self.entities = set()
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def add_entity(self, entity):
        self.entities.add(entity)

    def print(self):

        list_entitits =[]
        list_relations = []
        for e in self.entities:
            list_entitits.append(e)

        for r in self.relations:
            list_relations.append(r)
        return list_entitits, list_relations
            


def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Tokenize text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                            return_tensors='pt')
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate predictions
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Create KB
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)
            kb.add_entity(r["head"])
            kb.add_entity(r["tail"])
    
    list_entitits, list_relations = kb.print()
    return list_entitits, list_relations
    # return kb


text = "Napoleon Bonaparte (born Napoleone di Buonaparte; 15 August 1769 – 5 " \
"May 1821), and later known by his regnal name Napoleon I, was a French military " \
"and political leader who rose to prominence during the French Revolution and led " \
"several successful campaigns during the Revolutionary Wars. He was the de facto " \
"leader of the French Republic as First Consul from 1799 to 1804. As Napoleon I, " \
"he was Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's " \
"political and cultural legacy has endured, and he has been one of the most " \
"celebrated and controversial leaders in world history."

# kb = from_small_text_to_kb(text, verbose=True)
# kb.print()

# # Visualize the knowledge graph
# def save_network_html(kb, filename="network.html"):
#     # create network
#     net = Network(directed=True, width="auto", height="700px", bgcolor="#eeeeee")

#     # nodes
#     color_entity = "#00FF00"
#     for e in kb.entities:
#         net.add_node(e, shape="circle", color=color_entity)

#     # edges
#     for r in kb.relations:
#         net.add_edge(r["head"], r["tail"],
#                     title=r["type"], label=r["type"])
        
#     # save network
#     net.repulsion(
#         node_distance=200,
#         central_gravity=0.2,
#         spring_length=200,
#         spring_strength=0.05,
#         damping=0.09
#     )
#     net.set_edge_smooth('dynamic')
#     net.show(filename)

# # save_network_html(kb)

