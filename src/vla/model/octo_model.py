from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple

import json
import logging
import torch
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vla.model.octo_module import OctoModule
from src.vla.model.components.action_heads import ActionHead
from src.vla.utils.spec import ModuleSpec
from src.vla.utils.typing import Config, Data, Params, PRNGkey, Sequence

@dataclass
class OctoModel:
    """
    Recommended way of interacting with Octo Models

    Usage for inference:

        >>> model = OctoModel.load_pretrained(checkpoint_dir)
        >>> tasks = model.create_tasks(texts=["go to the room"])
        >>> # or tasks = model.create_tasks(goals={'iamge_primary': goal_images})
        >>> actions = model.sample_actions(observations, tasks, rng=torch.random.generate(0))
        >>> # Note: these are normalized actions (processed to mean 0 and std 1). To get correct actions
        >>> # for a particular embodiment, you must additionally specify unnormalization statistics.
        >>> # For example, to get actions for one of Octo's pretraining datasets:
        >>> actions = model.sample_actions(observations, tasks, rng, 
        >>>     unnormalization_statistics = model.dataset_statistics['DATASET_NAME_HERE']['action'])

    Usage for finetuning:

        >>> model = OctoModel.load_pretrained(checkpoint_dir)
        >>> train_state = octo.utils.train_utils.TrainState.create(
            rng = torch.random.rng(0),
            model = model,
            optim = torch.optim.adam(...)
        )
        >>> # access params through train_state.model.params
        >>> train_state, metrics = your_update_function(train_state, batch)
        >>> # when it's time to save (note that this only saves the model parameters, 
        >>> # not the full optimizer state)
        >>> train_state.model.save_pretrained(step, save_dir)

    Usage for pretraining:

        >>> model = OctoModel.from_config(
                config, 
                example_batch,
                text_processor
            ) # initializes params
        >>> # Continue as in train.py and finetune.py 
    """ 

    module: torch.nn.Module
    config: Dict[str, Any]
    example_batch: Dict[str, Any]
    text_processor: Optional[Any] = None
    dataset_statistics: Optional[Dict[str, Any]] = None 
    device: torch.device = torch.device('cuda')

    @torch.no_grad()
    def create_task(
        self, 
        goals: Optional[Dict[str, torch.Tensor]] = None,
        texts: Optional[Sequence[str]] = None 
    ) -> Dict[str, Any]:
        """
        Creates tasks dcit from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit iamges to run the language-conditioned model, and omit texts to run the goal-conditioned model
        """
        assert (goals is not None) or (texts is not None), "Provides goals or texts"

        tasks: Dict[str, Any] = {'pad_mask_dict': { }}

        if goals is not None:
            # goals: dict of [B, ...]
            tasks.update(goals)
            for k, v in goals.items():
                tasks['pad_mask_dict'][k] = torch.ones(v.shape[0], dtype=torch.bool, device=v.device)
            batch_size = next(iter(goals.values())).shape[0] 

        else: 
            # No goals -> fill non-language task keys with zeros per example spec
            batch_size = len(texts)
            for k, v in self.example_batch['task'].items():
                if k in ('pad_mask_dict', 'language_instruction'):
                    continue 
                shape = (batch_size, *v.shape[1:])
                tasks[k] = torch.zeros(shape, dtype = torch.float32, device = self.device)
                tasks['pad_mask_dict'][k] = torch.zeros(batch_size, dtype = torch.bool, device=self.device)

        if texts is not None:
            assert self.text_processor is not None, 'Text Processor required for texts.'
            ## Encode text using our processor (HF). Expecet dict/torch tensor
            encoded = self.text_processor.encode(texts)
            tasks['language_instruction'] = encoded 
            tasks['pad_mask_dict']['language_intsruction'] = torch.zeros(
                batch_size, dtype = torch.bool, device = self.device
            ) 

        _verify_shapes(tasks, 'tasks', self.example_batch['task'], starting_dim = 1)

        return tasks 


    @torch.no_grad()
    def run_transformer(
        self,
        observation: Dict[str, torch.Tensor],
        tasks: Dict[str, Any],
        timestep_pad_mask: torch.Tensor,
        train: bool = False
    ):
        "Checks shapes then run OctoTransformer (module.octo_transformer)."