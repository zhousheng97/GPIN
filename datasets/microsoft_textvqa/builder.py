# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.microsoft_textvqa.dataset import MICROTextVQADataset
from pythia.datasets.vqa.textvqa.builder import TextVQABuilder


@Registry.register_builder("microsoft_textvqa")
class MICROTextVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "microsoft_textvqa"
        self.set_dataset_class(MICROTextVQADataset)
