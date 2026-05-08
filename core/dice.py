"""
dice.py — DiCE counterfactual explainer registry and training logic.

This module provides a registry of DiCE explainer configurations,
including sampling strategies and backend settings, along with
handling for model loading and initialization.
"""

from core.models import MODEL_REGISTRY
import dice_ml as dice;
from dice_ml.constants import SamplingStrategy;
from dice_ml.constants import BackEndTypes;
from dice_ml.constants import ModelTypes;
import dice_ml.diverse_counterfactuals;
import dice_ml.counterfactual_explanations;

import pandas as pd;
from core.models import MODEL_REGISTRY;

DICE_REGISTRY = {
    "Random": {
        "sampling_strategy": SamplingStrategy.Random,
        "backend": BackEndTypes.Sklearn,
        "model_type": ModelTypes.Classifier,
        "model_name": "random",
        "model": None,
        "model_path": "",
    },
    "Genetic": {
        "sampling_strategy": SamplingStrategy.Genetic,
        "backend": BackEndTypes.Sklearn,
        "model_type": ModelTypes.Classifier,
        "model_name": "genetic",
        "model": None,
        "model_path": "",
    },
    "KDTree": {
        "sampling_strategy": SamplingStrategy.KdTree,
        "backend": BackEndTypes.Sklearn,
        "model_type": ModelTypes.Classifier,
        "model_name": "kdtree",
        "model": None,
        "model_path": "",
    },
    "Gradient": {
        "sampling_strategy": SamplingStrategy.Gradient,
        "backend": BackEndTypes.Sklearn,
        "model_type": ModelTypes.Classifier,
        "model_name": "gradient",
        "model": None,
        "model_path": "",
    },
}

