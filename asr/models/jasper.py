from copy import copy

import torch.nn as nn

from common.module import MaskedConv1d, JasperEncoder, JasperDecoderForCTC

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, nn.Conv1d):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class JasperAcousticModel(nn.Module):
    def __init__(self, enc: JasperEncoder, dec: JasperDecoderForCTC):
        nn.Module.__init__(self)
        self.jasper_encoder = enc
        self.jasper_decoder = dec

    def forward_encoder(self, x):
        features, length = x
        return self.jasper_encoder(features, length)

    def forward(self, x):
        t_encoded_t, t_encoded_len_t = self.forward_encoder(x)
        out = self.jasper_decoder(encoder_output=t_encoded_t)
        if self.jasper_encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out


class JasperInferenceModel(nn.Module):
    """Contains jasper encoder and decoder
    """

    def __init__(self,
                 acoustic_model: JasperAcousticModel):
        nn.Module.__init__(self)
        self.acoustic_model = acoustic_model

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def feature_count(self):
        return self.acoustic_model.jasper_encoder.input_dim()

    def forward(self, features):
        return self.acoustic_model.forward((features, None))


class Jasper(nn.Module):
    """Contains data jasper encoder and decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self._features_config = kwargs.get("features_config")
        encoder_out_dim = self.jasper_encoder.output_dim()

        self.jasper_decoder = JasperDecoderForCTC(input_dim=encoder_out_dim,
                                                  dictionary=kwargs.get("dictionary"))

        self.acoustic_model = JasperAcousticModel(self.jasper_encoder,
                                                  self.jasper_decoder)

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        features, lengths = x

        if self.jasper_encoder.use_conv_mask:
            acoustic_input = (features, lengths)
        else:
            acoustic_input = (features, None)
        # Forward Pass through Encoder-Decoder
        return self.acoustic_model.forward(acoustic_input)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state["features_config"] = self._features_config
        return state

    def load_state_dict(self, state_dict, strict=True):
        # this should copy just names
        temp = copy(state_dict) if strict else state_dict
        self._features_config = temp["features_config"]
        if strict:
            temp.pop("features_config")
        super().load_state_dict(temp, strict)

    def inference_model(self):
        return JasperInferenceModel(self.acoustic_model)
