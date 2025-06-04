import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PRIAS.resnet import resnet50, count_parameters
import os
from argparse import Namespace


def load_resnet50(which_resnet):
    model_path = os.path.join(os.getcwd(), 'supervised', 'saved_models', which_resnet)
    # 'resnet50_2023-04-28', 'version_0', 'last.pth')

    model = resnet50(Namespace(use_fcn=False, num_classes=2, binary=True))
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=False)
    return model


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_V = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_U = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_V.append(nn.Dropout(0.25))
            self.attention_U.append(nn.Dropout(0.25))

        self.attention_V = nn.Sequential(*self.attention_V)
        self.attention_U = nn.Sequential(*self.attention_U)

        self.attention_w = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_V(x)
        b = self.attention_U(x)
        A = a.mul(b)
        A = self.attention_w(A)  # N x n_classes
        return A, x


class Cumulative_Probability_Layer(nn.Module):
    """
    Cumulative probability layer for predicting future risk
    From https://github.com/yala/Mirai
    """
    def __init__(self, num_features, max_followup, make_probs_indep=False, use_base_pred=True):
        super(Cumulative_Probability_Layer, self).__init__()
        self.make_probs_indep = make_probs_indep
        self.use_base_pred = use_base_pred
        self.hazard_fc = nn.Linear(num_features,  max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)  
        self.sigmoid = nn.Sigmoid()

        """mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)"""

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        #prob_hazard = self.sigmoid(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.make_probs_indep:
            return self.hazards(x)
        hazards = self.hazards(x)
        cum_prob = torch.cumsum(hazards, dim=1)
        
        #B, T = hazards.size() #hazards is (B, T)
        #expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T) #expanded_hazards is (B,T, T)
        #masked_hazards = expanded_hazards * self.upper_triagular_mask # masked_hazards now (B,T, T)
        #cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)

        if self.use_base_pred:
            cum_prob += self.base_hazard_fc(x)
        return cum_prob


class Hazard_Module(nn.Module):
    """
    My own implementation of the hazard module
    """
    # implement component where seperate layers predict future risk, cumulative probability
    def __init__(self, num_features, max_followup):
        super().__init__()
        self.hazard_fc = []
        for i in range(max_followup):
            self.hazard_fc.append(nn.Linear(num_features, 1))

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  #nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = torch.abs(raw_hazard)  #self.relu(raw_hazard)
        #prob_hazard = self.sigmoid(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.make_probs_indep:
            return self.hazards(x)
        hazards = self.hazards(x)
        cum_prob = torch.cumsum(hazards, dim=1)
        if self.use_base_pred:
            cum_prob += self.base_hazard_fc(x)
        return cum_prob


class PRIAS_Model(nn.Module):
    """
    PRIAS Model
    Args:
        config: config file with model parameters
        use_features: whether to use features or not (default: True)
        long_mode: whether to predict long term risk
        feature_size: size of feature vector (default: 2048)
    Need to implement:
        - a better hazard module?
        - include risk factor prediction
    """
    def __init__(self,
                 config,
                 use_features=True,
                 long_mode=False,
                 survival_mode=False,
                 return_attention=False,
                 hidden_layers=0,
                 feature_size=2048,
                 n_follow_up=3):
        super(PRIAS_Model, self).__init__()
        self.config = config
        self.feats = use_features
        self.long_mode = long_mode
        self.survival_mode = survival_mode
        self.return_a = return_attention

        # Get ResNet baseline
        if not self.feats:
            self.backbone = load_resnet50(config.pretrained_resnet)
        else:
            self.backbone = None

        # Attention module
        if self.feats:
            fc = [nn.BatchNorm1d(feature_size), nn.Linear(feature_size, 1024), nn.ReLU()]
        else:
            fc = [nn.Linear(2048, 1024), nn.ReLU()]
        if self.config.dropout:
            fc.append(nn.Dropout(0.25))
        attn_net = Attn_Net_Gated(L=1024, D=256, dropout=self.config.dropout, n_classes=1)
        fc.append(attn_net)
        self.gated_attention = nn.Sequential(*fc)
        count_parameters(self.gated_attention)

        # Cumulative probability layer for predicting future risk
        if self.long_mode or self.survival_mode:
            self.cpl = Cumulative_Probability_Layer(1024, n_follow_up, use_base_pred=True, make_probs_indep=survival_mode)

        # Baseline classifier
        if hidden_layers > 0:
            fc2 = []
            size = 1024
            factor = 1
            for i in range(hidden_layers):
                fc2.append(nn.Linear(size, int(size/factor)))
                fc2.append(nn.ReLU())
                if self.config.dropout:
                    fc2.append(nn.Dropout(0.25))
                size = int(size/factor)
            fc2.append(nn.Linear(size, 1))
            self.classifier = nn.Sequential(*fc2)
        else:
            self.classifier = nn.Linear(1024, 1)

        # can implement a different threshold for classification here
        self.threshold = 0.5

    def forward(self, x):
        if self.feats:
            x_hat = x
        else:
            x_hat = self.backbone(x)
        a, x_hat = self.gated_attention(x_hat)  # NxK (batch x num classes)
        a = F.softmax(a, dim=0)  # softmax over N
        a = torch.transpose(a, 1, 0)  # KxN
        h = torch.mm(a, x_hat)

        if self.long_mode:
            # check that this works properly
            #logits = torch.cat((self.classifier(h), self.cpl(h)), dim=1)
            logits = self.cpl(h)
        else:
            logits = self.classifier(h)

        if self.survival_mode:
            return logits, a
        else:
            y_prob = torch.sigmoid(logits)
            y_hat = torch.where(y_prob > self.threshold, 1, 0)

            if self.return_a:
                return logits, y_prob, y_hat, a
            else:
                return logits, y_prob, y_hat
        

class PRIAS_Model_Stroma(PRIAS_Model):
    """
    PRIAS Model with Stroma prediction
    Args:
        config: config file with model parameters
        use_features: whether to use features or not (default: True)
        long_mode: whether to predict long term risk
        survival_mode: whether to predict survival risk
        return_attention: whether to return attention weights
        hidden_layers: number of hidden layers in the classifier
        feature_size: size of feature vector (default: 2048)
        n_follow_up: number of follow up periods (default: 3)
    """
    def __init__(self,
                 config,
                 use_features=True,
                 long_mode=False,
                 survival_mode=False,
                 return_attention=False,
                 hidden_layers=0,
                 feature_size=2048,
                 n_follow_up=3):
        super(PRIAS_Model_Stroma, self).__init__(config, use_features, long_mode, survival_mode, return_attention, hidden_layers, feature_size, n_follow_up)

    def forward(self, x, p = None):
        if self.feats:
            x_hat = x
        else:
            x_hat = self.backbone(x)
        a, x_hat = self.gated_attention(x_hat)  # NxK (batch x num classes)

        if p is not None:
            p = p.to(a.dtype)
            a = torch.transpose(a, 1, 0) + torch.log(p + 1e-8)  # add stroma prediction to attention weights

        a = F.softmax(a, dim=0)  # softmax over N
        h = torch.mm(a, x_hat)

        if self.long_mode:
            # check that this works properly
            #logits = torch.cat((self.classifier(h), self.cpl(h)), dim=1)
            logits = self.cpl(h)
        else:
            logits = self.classifier(h)
        y_prob = torch.sigmoid(logits)
        y_hat = torch.where(y_prob > self.threshold, 1, 0)

        if self.return_a:
            return logits, y_prob, y_hat, a
        else:
            return logits, y_prob, y_hat




class RiskFactor_Model(nn.Module):
    # Small network to predict risk factors from features
    def __init__(self,
                 input_size,
                 hidden_layers=1,
                 num_factors=2,
                 min_val=0,
                 dropout=True):
        super(RiskFactor_Model, self).__init__()

        self.in_sz = input_size
        self.min = min_val
        self.n_factors = num_factors

        self.attn_model = Attn_Net_Gated(L=input_size, D=256, dropout=dropout)
        #self.classifier = nn.Linear(input_size, num_factors)

        fc = []
        for i in range(hidden_layers):
            out_size = 512//(2**i)
            fc.append(nn.Linear(input_size, out_size))
            fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(0.25))
            input_size = out_size

        """
        fc = [nn.Linear(input_size, 256), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        """

        fc.append(nn.Linear(input_size, self.n_factors))

        self.rf_model = nn.Sequential(*fc)



    def forward(self, x):
        a, x_hat = self.attn_model(x)
        a = F.softmax(a, dim=0)  # softmax over N
        h = torch.mm(torch.transpose(a, 1, 0), x_hat)
        y = self.rf_model(h)
        #y = torch.clamp(y, min=self.min)
        return y


class PRIAS_RP_Model(nn.Module):
    # Figure out how to deal with missing risk factors
    def __init__(self,
                 config,
                 # rf_model: RiskFactor_Model,
                 k_years=5,
                 num_risk_factors=2,
                 use_features=True,
                 feature_size=1920):
        super(PRIAS_RP_Model, self).__init__()
        self.config = config
        self.feats = use_features
        self.K = k_years
        self.n_factors = num_risk_factors
        self.n_classes = k_years + 1 + num_risk_factors  # add one for year 0 (baseline risk)

        # Get ResNet baseline
        if not self.feats:
            self.backbone = load_resnet50(config.pretrained_resnet)
        else:
            self.backbone = None

        # Attention module
        if self.feats:
            fc = [nn.Linear(feature_size, 1024), nn.ReLU()]
        else:
            fc = [nn.Linear(2048, 1024), nn.ReLU()]
        if self.config.dropout:
            fc.append(nn.Dropout(0.25))
        attn_net = Attn_Net_Gated(L=1024, D=256, dropout=self.config.dropout, n_classes=self.n_classes)
        fc.append(attn_net)
        self.gated_attention = nn.Sequential(*fc)
        count_parameters(self.gated_attention)

        self.risk_factors = nn.ModuleList([nn.Linear(1024, 1) for i in range(
            self.n_factors)])  # use an indepdent linear layer to predict each risk factor
        self.classifiers = nn.ModuleList(
            [nn.Linear(1024, 1) for i in range(self.K + 1)])  # K+1 classifiers to predict risk

        # self.classifier = nn.Linear(1024, 1)
        # can implement a different threshold for classification here
        self.threshold = 0.5  # for risk prediction

    def forward(self, x):
        if self.feats:
            x_hat = x
        else:
            x_hat = self.backbone(x)
        a, x_hat = self.gated_attention(x_hat)  # NxK (batch x num classes)
        a = F.softmax(a, dim=0)  # softmax over N
        h = torch.mm(torch.transpose(a, 1, 0), x_hat)

        logits = self.classifier(h)
        y_prob = torch.sigmoid(logits)  # F.softmax(logits, dim = 1)
        y_hat = torch.where(y_prob > self.threshold, 1, 0)  # torch.topk(logits, k=1, dim=1)[1]

        return logits, y_prob, y_hat
