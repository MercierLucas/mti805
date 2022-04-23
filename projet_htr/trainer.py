import json
import time
from typing import List
from classifiers import Classifier
from datasets import ClassifDataset


class Trainer:
    stats = {}

    def __init__(self, dataset:List[ClassifDataset]) -> None:
        self.datasets = dataset

    def train(self, models:List[Classifier], output=None):
        self.stats = {m.name: {} for m in models}
        for dataset in self.datasets:
            print(f'Start training on {dataset.name}')
            for model in models:
                print(f'> Training {model.name}')
                t = time.time()
                model.train(dataset)
                model.save(model=f'./models/{model.name}_{dataset.name}.pkl')
                elapsed = time.time() - t
                self.stats[model.name]['train_time'] = elapsed

        if output:
            with open(f'{output}.json', 'w') as f:
                json.dump(self.stats, f, sort_keys=True, indent=4)
                print(f'Results saved to {output}.json')
        
        



