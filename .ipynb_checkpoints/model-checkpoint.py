import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import MultiheadAttention
import config
# from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.models.efficientnet import efficientnet_b4
import timm
import math


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class SelfAttentionHead(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttentionHead, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # Reshape the input tensor to (batch_size, height*width, in_channels)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b, h*w, c)

        # Pass the input tensor through the query, key, and value linear layers
        q = self.query(x_reshaped).reshape(b, h*w, self.num_heads, c//self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x_reshaped).reshape(b, h*w, self.num_heads, c//self.num_heads).permute(0, 2, 3, 1)
        v = self.value(x_reshaped).reshape(b, h*w, self.num_heads, c//self.num_heads).permute(0, 2, 1, 3)

        # Compute the attention scores and apply dropout
        att = (q @ k) / math.sqrt(c//self.num_heads)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Compute the output using the attention scores and the value matrix
        out = (att @ v).transpose(1, 2).reshape(b, h, w, c)
        out = self.out(out)

        return out


class RSNAModel(nn.Module):
    def __init__(self, output_size, n_cols, pos_weight=None):
        super().__init__()
        self.no_columns, self.output_size = n_cols, output_size
        self.use_weights = False if pos_weight is None else True
        self.pos_weight = torch.tensor(pos_weight) if self.use_weights else None
        self.label_smoothing = config.LABEL_SMOOTHING
        
        # Define the backbone model
        # self.backbone = timm.create_model(config.MODEL_BACKBONE,
        #                                   pretrained=True,
        #                                   num_classes=0,
        #                                   global_pool="",
        #                                   in_chans=config.IN_CHANNELS)
        
        self.backbone = timm.create_model(config.MODEL_BACKBONE, pretrained=True)

        # Define the GeM pooling layer
        # self.global_pool = GeM(p_trainable=True)
        
        # Define the self-attention head
        # self.attention_head = SelfAttentionHead(self.backbone.num_features)

        # Define the meta head
#         self.meta_head = nn.Sequential(nn.Linear(self.no_columns, 200),
#                                  nn.BatchNorm1d(200),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.2),
                                 
#                                  nn.Linear(200, 200),
#                                  nn.BatchNorm1d(200),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.1))
        
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # self.classification = torch.nn.Sequential(torch.nn.Linear(2*self.backbone.num_features + 200, output_size))
        self.classification = torch.nn.Linear(self.backbone.num_features, self.output_size)

    def forward(self, image, meta=None, target=None, image_id=None):
        x = image.float()

        # Pass input through the backbone model
        x_backbone = self.backbone.forward_features(x)

        # Apply GeM pooling to the backbone output
        # x_backbone_pool = self.global_pool(x_backbone)
        x_backbone_pool = F.adaptive_avg_pool2d(x_backbone,1)
        x_head = torch.flatten(x_backbone_pool,1,3)

        # Pass input through the self-attention head
        # x_attention = self.attention_head(x_backbone)
        # x_attention = x_attention.mean(dim=[1, 2])  # collapse spatial dimensions
        # x_attention = x_attention.view(x_attention.size(0), -1)  # flatten channel dimension
        

        # Pass meta features through the meta head
        # x_meta = self.meta_head(meta.float())

        # Concatenate layers from image with layers from csv_data
        # x_combined = torch.cat((x_head, x_attention,x_meta), dim=1)
        # x_combined = torch.cat((x_head), dim=1)
        
        # Apply dropout to avoid overfitting
        x_combined = self.dropout(x_head)

        # Classification layer
        out = self.classification(x_combined)
        

        loss = None
        if target is not None:
            # Apply label smoothing to the targets
            if self.label_smoothing > 0:
                target = target.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

            loss = self.loss_fn(out, target.unsqueeze(1).float(), use_weights=self.use_weights)
            out = torch.sigmoid(out)

        return out, loss
    
    def loss_fn(self, output, target, use_weights=False):
        if use_weights:
            return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)
        else:
            return torch.nn.BCEWithLogitsLoss()(output, target)    
    
    
    
    
    
    
    
    
    
    
    
    
# class RSNAModel(nn.Module):
#     def __init__(self, output_size, n_cols, pos_weight=None):
#         super().__init__()
#         self.no_columns, self.output_size = n_cols, output_size
#         self.use_weights = False if pos_weight is None else True
#         self.pos_weight = torch.tensor(pos_weight) if self.use_weights else None
#         self.label_smoothing = config.LABEL_SMOOTHING
        
#         # Define the backbone model
#         self.backbone = timm.create_model(config.MODEL_BACKBONE,
#                                           pretrained=True,
#                                           num_classes=0,
#                                           global_pool="",
#                                           in_chans=config.IN_CHANNELS)

#         # Define the GeM pooling layer
#         self.global_pool = GeM(p_trainable=True)

#         # Define the linear classifier head
#         self.head = torch.nn.Linear(self.backbone.num_features, self.output_size)

#         # Define the rfft head
#         self.rfft_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64, self.output_size)
#         )

#         # Define the meta head
#         self.meta_head = nn.Sequential(
#             nn.Linear(5, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, self.output_size)
#         )


#     def forward(self, image, meta=None, target=None, image_id=None):
#         x = image.float()

#         # Pass input through the backbone model
#         x_backbone = self.backbone(x)

#         # Apply GeM pooling to the backbone output
#         x_backbone_pool = self.global_pool(x_backbone)

#         # Flatten the pooled output and pass through the linear classifier head
#         x_head = x_backbone_pool.view(x_backbone_pool.size(0), -1)
#         x_head = self.head(x_head)

#         # Pass input through the rfft head
#         x_rfft = self.rfft_head(x)

#         # Pass meta features through the meta head
#         if meta is not None:
#             meta = meta.to(x_rfft.device)
#             x_meta = self.meta_head(meta.to(x_rfft.dtype))

#         # Combine the outputs of the three heads
#         x_combined = x_head + x_rfft + x_meta

#         loss = None
#         if target is not None:
#             # Apply label smoothing to the targets
#             if self.label_smoothing > 0:
#                 target = target.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

#             loss = self.loss_fn(x_combined, target.unsqueeze(1).float(), use_weights=self.use_weights)
#             x_combined = torch.sigmoid(x_combined)

#         return x_combined, loss
    
#     def loss_fn(self, output, target, use_weights=False):
#         if use_weights:
#             return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)
#         else:
#             return torch.nn.BCEWithLogitsLoss()(output, target)










# class RSNAModel(torch.nn.Module):
#     def __init__(self, output_size, n_cols,pos_weight=None):
#         super().__init__()
#         self.no_columns, self.output_size = n_cols, output_size
#         self.use_weights = False if pos_weight is None else True 
#         self.pos_weight =  torch.tensor(pos_weight) if self.use_weights else None
#         # self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
#         # self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        
        
#         # Define Feature part (IMAGE)
#         # self.features = EfficientNet.from_pretrained(config.MODEL_BACKBONE)
#         # self.features = timm.create_model(config.MODEL_BACKBONE, pretrained=True,num_classes=config.NUM_CLASSES, drop_rate=0)
#         if config.LOAD_PRETRAINED:
#             self.features = timm.create_model(config.MODEL_BACKBONE, pretrained=False)
#         else:
#             self.features = timm.create_model(config.MODEL_BACKBONE, pretrained=True)
#         self.dropout = nn.Dropout(config.DROPOUT)
        
        
        
#         # (CSV)
# #         self.csv = torch.nn.Sequential(torch.nn.Linear(self.no_columns, 250),
# #                                  torch.nn.BatchNorm1d(250),
# #                                  torch.nn.ReLU(),
# #                                  torch.nn.Dropout(p=0.2),
                                 
# #                                  torch.nn.Linear(250, 250),
# #                                  torch.nn.BatchNorm1d(250),
# #                                  torch.nn.Dropout(p=0.2))
        
#         # Define Classification part
#         # self.attention = self.attention = MultiHeadAttention(self.features.num_features, num_heads=8)
#         # self.classification = torch.nn.Sequential(torch.nn.Linear(self.features.num_features + 250, 1))
#         self.classification = nn.Linear(self.features.num_features,1)
        
        
#     def forward(self, image, meta=None,target=None,image_id=None):   
        
#         # Image CNN
#         x = image
#         # batch_size,C,H,W = x.shape
#         # x = (x - self.mean) / self.std
#         # image = self.features.extract_features(image.float()) # EfficientNet
#         features = self.features.forward_features(x.float())
        
#         x = F.adaptive_avg_pool2d(features,1)
#         x = torch.flatten(x,1,3)
        
        
#         # Apply multi-head attention
#         # x, _ = self.attention(x, x, x)
        
        
#         # Apply dropout to avoid overfitting
#         x = self.dropout(x)
        
#         # CLASSIF
#         out = self.classification(x)
        
        
#         loss = None
#         if target is not None:
#             loss = self.loss_fn(out, target.unsqueeze(1).float(), use_weights=self.use_weights)
#             out = torch.sigmoid(out)
#         else :
#             out = torch.sigmoid(out)
        
#         return out, loss
        
    
#     def loss_fn(self, output, target, use_weights=False):
    
#         if use_weights:
#             # Define the smoothing factor
#             eps = 0.1

#             # Define the class weights
#             class_weights = torch.tensor([1, self.pos_weight], dtype=torch.float)
#             class_weights = class_weights / class_weights.sum()

#             # Define the target distribution
#             num_classes = 2 # binary classification
#             target_dist = torch.ones(target.shape[0], num_classes) * eps / num_classes
#             target_dist[torch.arange(target.shape[0]), target.long()] = 1 - eps + eps / num_classes
#             target_dist = target_dist * class_weights.unsqueeze(0)
#             return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)

#         else:

#             return torch.nn.BCEWithLogitsLoss()(output, target)    

    # def loss_fn(self, output, target, use_weights=False):
    #     if use_weights:
    #         return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)
    #     else:
    #         return torch.nn.BCEWithLogitsLoss()(output, target)


        


# def gem(x, p=3, eps=1e-6):
#     return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


# class GeM(nn.Module):
#     def __init__(self, p=3, eps=1e-6, p_trainable=False):
#         super(GeM, self).__init__()
#         if p_trainable:
#             self.p = Parameter(torch.ones(1) * p)
#         else:
#             self.p = p
#         self.eps = eps

#     def forward(self, x):
#         ret = gem(x, p=self.p, eps=self.eps)
#         return ret

#     def __repr__(self):
#         return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


# class RSNAModel(nn.Module):

#     def __init__(self, output_size, n_cols, pos_weight=None):
#         super().__init__()
#         self.no_columns, self.output_size = n_cols, output_size
#         self.use_weights = False if pos_weight is None else True
#         self.pos_weight = torch.tensor(pos_weight) if self.use_weights else None
#         self.backbone = timm.create_model(config.MODEL_BACKBONE,
#                                           pretrained=True,
#                                           num_classes=0,
#                                           global_pool="",
#                                           in_chans=config.IN_CHANNELS)

#         self.global_pool = GeM(p_trainable=True)
#         self.head = torch.nn.Linear(self.backbone.num_features, self.output_size)

#     def forward(self, image, meta=None, target=None, image_id=None):
#         x = image.float()
#         x = self.backbone(x)
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         out = self.head(x)

#         loss = None
#         if target is not None:
#             loss = self.loss_fn(out, target.unsqueeze(1).float(), use_weights=self.use_weights)
#             out = torch.sigmoid(out)

#         return out, loss        
    
#     def loss_fn(self, output, target, use_weights=False):
    
#         if use_weights:
#             # Define the smoothing factor
#             eps = 0.1

#             # Define the class weights
#             class_weights = torch.tensor([1, self.pos_weight], dtype=torch.float)
#             class_weights = class_weights / class_weights.sum()

#             # Define the target distribution
#             num_classes = 2 # binary classification
#             target_dist = torch.ones(target.shape[0], num_classes) * eps / num_classes
#             target_dist[torch.arange(target.shape[0]), target.long()] = 1 - eps + eps / num_classes
#             target_dist = target_dist * class_weights.unsqueeze(0)
#             return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)

#         else:

#             return torch.nn.BCEWithLogitsLoss()(output, target)
    
    
    
    # def loss_fn(self, output, target, use_weights=False):
    #     if use_weights:
    #         return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(output, target)
    #     else:
    #         return torch.nn.BCEWithLogitsLoss()(output, target)



    
# class CostSensitiveLoss(nn.Module):
#     def __init__(self, costs):
#         super(CostSensitiveLoss, self).__init__()
#         self.costs = costs

#     def forward(self, input, target):
#         pred = torch.argmax(input, dim=1)
#         loss = -torch.mean(torch.sum(self.costs[target] * torch.log(input[range(target.shape[0]), target]), dim=1))
#         return loss

# # define misclassification costs
# costs = torch.Tensor([[0, 1], [2, 0]])

# # create an instance of the custom loss function
# loss_fn = CostSensitiveLoss(costs)