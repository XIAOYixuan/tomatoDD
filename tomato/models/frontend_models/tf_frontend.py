# encoding: utf-8
# Author: Yixuan
#
#

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from tomato.utils import utils, logger

from .base import BaseFrontEnd

class TFW2V2(BaseFrontEnd):

    def __init__(self, device, args=None):
        from transformers import Wav2Vec2Model

        super(TFW2V2, self).__init__(device)
        self.tf_mdl = args.tf_mdl
        self.model = Wav2Vec2Model.from_pretrained(self.tf_mdl)
        self.model.to(device)

    def extract_feat(self, input_data):
        raise NotImplementedError("TFW2V2 frontend extract_featis not implemented")
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        extract_features = self.model.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.model._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.model.feature_projection(extract_features)
        hidden_states = self.model._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.model.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.model.adapter is not None:
            hidden_states = self.model.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        #return Wav2Vec2BaseModelOutput(
        #    last_hidden_state=hidden_states,
        #    extract_features=extract_features,
        #    hidden_states=encoder_outputs.hidden_states,
        #    attentions=encoder_outputs.attentions,
        #)
        return encoder_outputs.hidden_states, attention_mask