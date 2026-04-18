from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple 
import logging 
import torch 
import torch.nn as nn
from einops import rearrange
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vla.model.components.base import TokenGroup
from src.vla.model.components.transformer import MAPHead
from src.vla.model.components.block_transformer import AttentionRule, BlockTransformer, PrefixGroup, TimestepGroup

from src.vla.utils.spec import ModuleSpec
from src.vla.utils.typing import Data, Sequence


def _concat_tokengroups(groups, axis: int = -2) -> TokenGroup:
    """
    Concatenate TokenGroup along token axis; masks along same axis.
    """
    if len(groups) == 0:
        raise ValueError('Need at least one TokenGroup to concatenate')
    tokens = torch.cat([g.tokens for g in groups], dim = axis)
    ## mask has one fewer dim than tokens (no embedding dims)
    ## If axis = -2 on tokens (n_tokens), corresponding mask axis = -1
    mask = torch.cat([g.mask for g in groups], dim = axis + 1)
    return TokenGroup(tokens = tokens, mask = mask)

class OctoTransformer(nn.Module):
    """
    This module forms the base of the Octo architecture

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]
    
    The task is tokenized using a set of *task tokenizer* (for example, a tokenizer that processes the
    language instruction into tokens, or one that processes the goal images into tokens)

    The observation at each timestep is tokenized using a set of *observation tokenizer*
    (for example, a tokenizer that processes the primary image into tokens, or one that processes 
    the wrist image into tokens).

    We introduce additional tokens ('readouts') that "read out" the information in the transformer for
    downstream action or value prediction. For example, we may have an 'action' readout that provides 
    embeddings that are useful for predicting actions, and a 'value' readout with embeddings that are useful
    for predicting values.

    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or
    previous timesteps. The easiest way to understand how the model works is to run:

        >>> model(observations, tasks, timestep_pad_mask, verbose = True)

    Generally, the model runs the transformer on something like the following sequences:
        [
            <task language tokens>
            <t=0 "image_primary" tokens>, <t=0 "image_wrist" tokens>, <t=0 readout_action tokens>, ...
            <t=1 "image_primary" tokens>, <t=1 "image_wrist" tokens>, <t=1 readout_action tokens>, ...
            <t=2 "image_primary" tokens>, <t=2 "image_wrist" tokens>, <t=2 readout_action tokens>, ...
            ...
        ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous 
    timesteps. So, "image_wrist" can attend to "image _primary" and vice versa

    Readouts provide a mechanism for "reading out" the information in the transformer. They are designed to
    only *read* from the sequence before it, without the ability to influence (i.e write) the computation for
    any of the non-readout tokens. By design different readouts (e.g. "action" vs "value") are completely 
    independent of each other, meaning they can be run separately without affecting each other. 

    Args:
        observations_tokenziers (Dict[str, nn.Module]): Dictionary of PyTorch modules for tokenizing the observations.
            The output of each other tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Dict[str, nn.Module]): Dictionary of PyTorch modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix
        readouts (Dict[str, int]): Dictionary of {readout_name: n_tokens_for_readout}.
        transformer_kwargs (Dict): Dictionary of kwargs to forward to the Transformer.
        token_embedding_size (int): Dimension of the token embeddings
        max_horizon (int): The maximum number of timesteps that the transformer can be run with. Note that while the 
            transformer can be run with any horizon <= max_horizon, the model will only generate same outputs for
            horizon lengths smaller or equal to the pre-training horizon.
        repeat_task_tokens: If True, repeat the task tokens at each observation timestep 
    """
    def __init__(
            self,
            observation_tokenizers: Dict[str, nn.Module],
            task_tokenizers: Dict[str, nn.Module],
            readouts: Dict[str, int],
            transformer_kwargs: Dict,
            token_embedding_size: int,
            max_horizon: int,
            repeat_task_tokens: bool,
            use_correct_attention: bool = False
    ):
        super().__init__()
        self.observation_tokenizers = nn.ModuleDict(observation_tokenizers)
        self.task_tokenizers = nn.ModuleDict(task_tokenizers)
        self.readout_cfg = readouts
        self.token_embedding_size = token_embedding_size
        self.max_horizon = max_horizon
        self.repeat_task_tokens = repeat_task_tokens
        self.use_correct_attention = use_correct_attention

        self.proj_head = nn.ModuleDict()

        self.pos_embed_prefix: Dict[str, torch.nn.Parameter] = {}
        self.pos_embed_timestep: Dict[str, torch.nn.Parameter] = {}

        num_layers = transformer_kwargs["num_layers"]
        num_heads = transformer_kwargs["num_heads"]
        mlp_dim = transformer_kwargs["mlp_dim"]
        dropout = transformer_kwargs.get("dropout_rate", 0.1)
        attn_dropout = transformer_kwargs.get("attention_dropout_rate", 0.1)

        # def builder(token_dim: int) -> nn.Module:
        #     from src.vla.model.components.block_transformer import TransformerWithFullMask
        #     return TransformerWithFullMask(
        #         num_layers=num_layers,
        #         Token_dim=token_dim,
        #         num_heads=num_heads,
        #         mlp_dim=mlp_dim,
        #         dropout=dropout,
        #         attn_dropout=attn_dropout,
        #     )
        # self.block = BlockTransformer(
        #     transformer_builder=builder,
        #     enforce_causal=True,
        #     use_correct_attention=self.use_correct_attention,
        # )
        self.block = BlockTransformer(transformer_kwargs=transformer_kwargs, enforce_causal=True,
                                      use_correct_attention=use_correct_attention)
        
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Dict[str, torch.Tensor],
        timestep_pad_mask: torch.Tensor,         # [B, horizon] bool
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
    ):
        if readouts is None:
            readouts = list(self.readout_cfg.keys())

        ## Basic shape check
        ## Get horizon from any observation leaf
        first_key = next(iter(observations))
        B, horizon = observations[first_key].shape[:2]
        assert horizon <= self.max_horizon, "Horizon must be <= max_horizon"

        for k,v in observations.items():
            assert v.shape[1] == horizon, f"observations['{k}'] has horizon {v.shape[1]} != {horizon}"

        ## Attention Rules
        task_attention_rules = {'task_*': AttentionRule.CAUSAL}
        observation_attention_rules = {
            'task_*': AttentionRule.CAUSAL,
            'obs_*': AttentionRule.CAUSAL
        }

        all_prefix_groups = []
        all_timestep_groups = []

        ## Task tokenizer to prefix groups
        for name, token in self.task_tokenizers.items():
            group_name = f'task_{name}'
            token_group: TokenGroup = token(observations, tasks, train=train)
            if token_group is None:
                logging.warning(f'Skipping task tokenizer: {group_name}')
                continue