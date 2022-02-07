import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, logging
from torch import nn

from dataloader import load

logging.set_verbosity(logging.ERROR)


@hydra.main(config_name='config.yml')
def main():
    
    
if __name__ == "__main__":
    main()
