import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types    

# Import operators
from .dataset_difficulty_operator import DatasetDifficultyScoring

def register(plugin):
    """Register operators with the plugin."""
    # Register the dataset difficulty scoring operator
    foo.register_operator(DatasetDifficultyScoring)