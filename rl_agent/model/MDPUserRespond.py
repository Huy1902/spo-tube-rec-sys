from rl_agent.model.BaseModel import BaseModel
import torch.nn as nn
import torch

class MDPUserResponse(BaseModel):
    def __init__(self, reader, params):
        super().__init__(reader, params)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean')

    def _define_params(self, reader, params):
        stats = reader.get_statistics()
        print(stats)

        self.item_dim     = stats['item_vec_size']
        self.feature_dim  = params['feature_dim']
        self.hidden_dim   = params['hidden_dims']
        self.attn_n_head  = params['attn_n_head']
        self.dropout_rate = params['dropout_rate']

        # Map item feature vectors -> model space
        self.item_emb_layer = nn.Linear(self.item_dim, self.feature_dim)

        # Learned query to read user history (shape [1,1,D])
        self.user_query = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

        # History encoders
        self.seq_self_attn_layer = nn.MultiheadAttention(
            self.feature_dim, self.attn_n_head, batch_first=True
        )
        self.seq_user_attn_layer = nn.MultiheadAttention(
            self.feature_dim, self.attn_n_head, batch_first=True
        )

        self.norm_h = nn.LayerNorm(self.feature_dim)
        self.norm_u = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.loss = []

    @staticmethod
    def _to_key_padding_mask(history_mask):
        """
        Convert history_mask (1=valid, 0=pad) -> key_padding_mask (True=PAD).
        If you don't have masks, return None.
        """
        if history_mask is None:
            return None
        return (history_mask == 0)

    def get_forward(self, feed_dict: dict) -> dict:
        """
        Expected in feed_dict:
          - history_features: [B, L, item_dim]
          - exposure_features: [B, K, item_dim]
          - (optional) history_mask: [B, L] with 1=valid, 0=pad
        Returns:
          - preds (logits): [B, K]
          - reg: scalar regularization term
        """
        history = feed_dict['history_features']      # [B, L, item_dim]
        exposure = feed_dict['exposure_features']    # [B, K, item_dim]
        history_mask = feed_dict.get('history_mask', None)

        # Encode history items
        H = self.item_emb_layer(history)             # [B, L, D]
        H = self.dropout(self.norm_h(H))

        # Self-attention over history (contextualize)
        kpm = self._to_key_padding_mask(history_mask)
        H_ctx, _ = self.seq_self_attn_layer(H, H, H, key_padding_mask=kpm)  # [B, L, D]

        # Learned-query pooling over the encoded history -> user_interest [B,1,D]
        B = H_ctx.size(0)
        Q = self.user_query.expand(B, -1, -1)        # [B,1,D]
        user_interest, _ = self.seq_user_attn_layer(Q, H_ctx, H_ctx, key_padding_mask=kpm)
        user_interest = self.dropout(self.norm_u(user_interest))            # [B,1,D]

        # Score exposure items by dot product with user_interest
        E = self.item_emb_layer(exposure)            # [B, K, D]
        score = torch.sum(E * user_interest, dim=-1) # [B, K]  (logits)

        # Regularize only modules that exist here
        reg = self.get_regularization(
            self.item_emb_layer, self.seq_user_attn_layer, self.seq_self_attn_layer, self.norm_h, self.norm_u
        )

        return {'preds': score, 'reg': reg}

    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        Use BCEWithLogitsLoss on logits directly.
        Targets should be shape-matched with preds: [B, K] of {0,1}.
        """
        preds = out_dict["preds"]        # logits [B, K]
        reg   = out_dict["reg"]

        target = feed_dict['feedback']   # [B, K]
        if not torch.is_floating_point(target):
            target = target.float()

        ce = self.bce_logits(preds, target)     # no sigmoid here
        self.loss.append(float(ce.detach().cpu()))
        loss = ce + self.l2_coef * reg
        return loss
