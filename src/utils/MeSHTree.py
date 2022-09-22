from typing import Optional, List, Dict, Tuple
import xml.etree.ElementTree as ET

import torch
from torch import nn
from tqdm import tqdm

Value = str

top_term_number_to_name: Dict[str, str] = {
    'A': 'Anatomy',
    'B': 'Organisms',
    'C': 'Diseases',
    'D': 'Chemicals and Drugs',
    'E': 'Analytical, Diagnostic and Therapeutic Techniques, and Equipment',
    'F': 'Psychiatry and Psychology',
    'G': 'Phenomena and Processes',
    'H': 'Disciplines and Occupations',
    'I': 'Anthropology, Education, Sociology, and Social Phenomena',
    'J': 'Technology, Industry, and Agriculture',
    'K': 'Humanities',
    'L': 'Information Science',
    'M': 'Named Groups',
    'N': 'Health Care',
    'V': 'Publication Characteristics',
    'Z': 'Geographicals'
}


def get_text(node):
    try:
        return node.text
    except AttributeError:
        return None


class MeSHTree:
    def __init__(self, value: Optional[str] = None, descriptor_name: Optional[str] = None,
                 tree_number: Optional[str] = None):
        self.children_nodes: List[MeSHTree] = []
        self.value: Optional[Value] = value
        self.descriptor_name: Optional[str] = descriptor_name
        self.tree_number: Optional[str] = tree_number
        self.tree_numbers_list: List[List[str]] = []
        self.clf_out = None
        self.clf = None

    def get_or_create_child(self, child_value: Value, descriptor_name: Optional[str] = None,
                            tree_number: Optional[str] = None):
        for child_tree in self.children_nodes:
            if child_tree.value == child_value:
                return child_tree
        new_tree = MeSHTree(child_value, descriptor_name, tree_number)
        self.children_nodes.append(new_tree)
        return new_tree

    def __str__(self):
        ret = self.value if self.value else '-'
        for node in self.children_nodes:
            ret += f" {node}"
        ret += ' |'
        return ret

    @property
    def norm_tree_number(self):
        return self.tree_number.replace(".", '_') if self.tree_number else 'ROOT'

    def build_from_ids(self, tree_str: str, descriptor_name: Optional[str] = None):
        if not descriptor_name and tree_str in top_term_number_to_name:
            descriptor_name = top_term_number_to_name[tree_str]
        top_term = tree_str[0]
        nodes = tree_str[1:].split('.')
        tmp_node = self.get_or_create_child(top_term)
        for node_str in nodes:
            tmp_node = tmp_node.get_or_create_child(node_str)
        tmp_node.descriptor_name = descriptor_name
        tmp_node.tree_number = tree_str

    def size(self):
        if len(self.children_nodes) == 0:
            return 0
        size = 1
        for child in self.children_nodes:
            size += child.size()
        return size

    def is_leaf(self):
        return len(self.children_nodes) == 0

    def __iter__(self):
        yield self
        for child in self.children_nodes:
            for sub_child in child:
                yield sub_child

    def iter_without_leafs(self):
        for node in self:
            if not node.is_leaf():
                yield node

    @property
    def mesh_vocab_size(self):
        return len(set([element.descriptor_name for element in self]))

    @property
    def node_count(self):
        return sum([len(node.children_nodes) for node in self.iter_without_leafs()])

    @staticmethod
    def build_original_mesh_tree(mesh_xml_path: str, mesh_vocabulary: Dict[str, int]):
        et = ET.parse(mesh_xml_path)  # /home/wojtek/Documents/Indexing/desc2022.xml
        root = et.getroot()
        tree = MeSHTree()
        tree.tree_number = 'ROOT'
        desc_dict: Dict[str, List[str]] = {}

        for record in root.findall('DescriptorRecord'):
            descriptor_name = get_text(record.find('DescriptorName/String'))
            if descriptor_name not in mesh_vocabulary.keys():
                continue
            norm_descriptor_name = descriptor_name.replace('.', '_')
            tree_numbers = [get_text(tree_number_node) for tree_number_node in
                            record.findall('TreeNumberList/TreeNumber')]
            if norm_descriptor_name in desc_dict:
                desc_dict[norm_descriptor_name].extend(tree_numbers)
            else:
                desc_dict[norm_descriptor_name] = tree_numbers

            for tree_number in tree_numbers:
                tree.build_from_ids(tree_number, descriptor_name)

        # change descriptor dictionary to list according to given vocabulary
        descriptors_ordered = list(map(lambda el: el[0], sorted(mesh_vocabulary.items(), key=lambda item: item[1])))
        tree.tree_numbers_list = [desc_dict[descriptor_name.replace('.', '_')] for descriptor_name in
                                  descriptors_ordered]
        return tree

    def forward(self, x):
        return torch.sigmoid(self.clf(x))

    def init_classifiers(self, input_length: int):
        for node in self.iter_without_leafs():
            node.clf = nn.Linear(input_length, len(self.children_nodes))
