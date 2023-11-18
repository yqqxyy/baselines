import argparse
import torch
import torch.nn.functional as F
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models

import nmmo
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class Random(pufferlib.models.Policy):
  '''A random policy that resets weights on every call'''
  def __init__(self, envs):
    super().__init__()
    self.envs = envs
    self.decoders = torch.nn.ModuleList(
        [torch.nn.Linear(1, n) for n in envs.single_action_space.nvec]
    )

  def encode_observations(self, env_outputs):
    return torch.randn((env_outputs.shape[0], 1)).to(env_outputs.device), None

  def decode_actions(self, hidden, lookup):
    torch.nn.init.xavier_uniform_(hidden)
    actions = [dec(hidden) for dec in self.decoders]
    return actions, None

  def critic(self, hidden):
    return torch.zeros((hidden.shape[0], 1)).to(hidden.device)


class Baseline(pufferlib.models.Policy):
  def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
    super().__init__(env)

    self.flat_observation_space = env.flat_observation_space
    self.flat_observation_structure = env.flat_observation_structure

    self.tile_encoder = TileEncoder(input_size)
    self.player_encoder = PlayerEncoder(input_size, hidden_size, num_heads = 8)
    self.item_encoder = ItemEncoder(input_size, hidden_size, num_heads = 8)
    self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
    self.market_encoder = MarketEncoder(input_size, hidden_size)
    self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
    self.proj_fc = torch.nn.Linear(5 * input_size, input_size)
    self.action_decoder = ActionDecoder(input_size, hidden_size)
    self.value_head = torch.nn.Linear(hidden_size, 1)

  def encode_observations(self, flat_observations):
    env_outputs = pufferlib.emulation.unpack_batched_obs(flat_observations,
        self.flat_observation_space, self.flat_observation_structure)
    tile = self.tile_encoder(env_outputs["Tile"])

    ticks_repeat = torch.repeat_interleave(env_outputs['CurrentTick'][:, None, :], env_outputs["Entity"].shape[-2], dim=1)
    Entity_with_tick = torch.cat([ticks_repeat, env_outputs["Entity"]], dim=2)
    player_embeddings, my_agent = self.player_encoder(
        Entity_with_tick, env_outputs["AgentId"][:, 0]
    )

    item_embeddings = self.item_encoder(env_outputs["Inventory"])
    inventory = self.inventory_encoder(item_embeddings, env_outputs["Inventory"])

    market_embeddings = self.item_encoder(env_outputs["Market"])
    market = self.market_encoder(market_embeddings, env_outputs["Market"])

    task = self.task_encoder(env_outputs["Task"])

    obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1)
    obs = self.proj_fc(obs)

    return obs, (
        player_embeddings,
        item_embeddings,
        market_embeddings,
        env_outputs["ActionTargets"],
    )

  def decode_actions(self, hidden, lookup):
    actions = self.action_decoder(hidden, lookup)
    value = self.value_head(hidden)
    return actions, value


class TileEncoder(torch.nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.tile_offset = torch.tensor([i * 256 for i in range(3)])
    self.embedding = torch.nn.Embedding(3 * 256, 32)

    self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
    self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
    self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

  def forward(self, tile):
    tile[:, :, :2] -= tile[:, 112:113, :2].clone()
    tile[:, :, :2] += 7
    tile = self.embedding(
        tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
    )

    agents, tiles, features, embed = tile.shape
    tile = (
        tile.view(agents, tiles, features * embed)
        .transpose(1, 2)
        .view(agents, features * embed, 15, 15)
    )

    tile = F.relu(self.tile_conv_1(tile))
    tile = F.relu(self.tile_conv_2(tile))
    tile = tile.contiguous().view(agents, -1)
    tile = F.relu(self.tile_fc(tile))

    return tile


class Dense(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list,
        activation: str,
        final_activation: str = None,
        norm_layer: str = None,
        norm_final_layer: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Save the networks input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # build nodelist
        node_list = [input_size, *hidden_layers, output_size]

        # input and hidden layers
        layers = []

        num_layers = len(node_list) - 1
        for i in range(num_layers):
            is_final_layer = i == num_layers - 1

            # normalisation first
            if norm_layer and (norm_final_layer or not is_final_layer):
                layers.append(getattr(torch.nn, norm_layer)(node_list[i], elementwise_affine=False))

            # then dropout
            if dropout and (norm_final_layer or not is_final_layer):
                layers.append(torch.nn.Dropout(dropout))

            # linear projection
            layers.append(torch.nn.Linear(node_list[i], node_list[i + 1]))

            # activation
            if not is_final_layer:
                layers.append(getattr(torch.nn, activation)())

            # final layer: return logits by default, otherwise apply activation
            elif final_activation:
                layers.append(getattr(torch.nn, final_activation)())

        # build the net
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)

def add_dims(x, ndim: int):
    """Adds dimensions to a tensor to match the shape of another tensor."""
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is larger than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x

def masked_softmax(x, mask, dim: int = -1):
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask,x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = F.softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % self.num_heads == 0

        self.head_dim = embed_dim // self.num_heads

        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return scaled_attention

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # if mask is not None:
        #     scaled_attention_logits += (mask * -1e9)

        # attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        attention_weights = masked_softmax(scaled_attention_logits, mask,dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_layers: int=1, num_heads: int = 8, dense_config = None, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        if dense_config:
            self.dense = Dense(
                input_size=embed_dim,
                output_size=embed_dim,
                **dense_config,
            )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.final_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
          x = x + self.norm2(layer(self.norm1(x), self.norm1(x), self.norm1(x), **kwargs))
          if self.dense:
            x = x + self.dense(x)
        x = self.final_norm(x)
        return x
    

class GlobalAttentionPooling(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = torch.nn.Linear(input_size, 1)

    def forward(self, x, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x), mask, dim=1)
        return (x * weights).sum(dim=1)

class CLSPooling(torch.nn.Module):
    def __init__(self, embed_dim, CLSidx: int = 0, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % self.num_heads == 0

        self.head_dim = embed_dim // self.num_heads

        self.CLSidx = CLSidx

        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_CLS, attention_weights = self.attention_CLS(q, k, v, mask)

        scaled_CLS = scaled_CLS.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return scaled_CLS

    def attention_CLS(self, q, k, v, mask):
        matmul_CLS_k = torch.matmul(q[:,:,self.CLSidx,:].unsqueeze(2), k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_CLS_logits = matmul_CLS_k / torch.sqrt(dk)

        # if mask is not None:
        #     scaled_attention_logits += (mask * -1e9)

        # attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        attention_weight = masked_softmax(scaled_CLS_logits, mask,dim=-1)
        attention_weight = self.dropout(attention_weight)
        output = torch.matmul(attention_weight, v)

        return output, attention_weight

class ResNet(torch.nn.Module):
  def __init__(self, 
        num_layers: int,
        input_size: int,
        hidden_layers: list,
        activation: str,
        norm_layer: str = None,
        dropout=0.0):
    super().__init__()

    layers = []

    self.num_layers = num_layers

    self.activation = getattr(torch.nn, activation)()

    for i in range(self.num_layers):
      layers.append( Dense(input_size= input_size, output_size= input_size, hidden_layers= hidden_layers, activation= activation, norm_layer=norm_layer, dropout=dropout))

      # build the net
    self.net = torch.nn.Sequential(*layers)

  def forward(self, x):

    for i in range(self.num_layers):
      x = self.activation(x + self.net(x))

    return x

class PlayerEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_heads: int = 8):
    super().__init__()
    #self.entity_dim = 31
    self.entity_dim = 32 # 31 entity dims + 1 tick dim

    self.agent_fc = torch.nn.Linear(self.entity_dim * 4, hidden_size)
    self.my_agent_fc = torch.nn.Linear(self.entity_dim * 4, input_size)

    self.dense = Dense(self.entity_dim, self.entity_dim*4, hidden_layers=[self.entity_dim*2], activation="SiLU", norm_layer="LayerNorm")
    self.num_heads = num_heads
    self.transformer = TransformerEncoder(self.entity_dim*4, num_layers=6, num_heads= self.num_heads, dense_config={"hidden_layers":[256], "activation": "SiLU", "norm_layer": "LayerNorm"})

    self.final_norm = torch.nn.LayerNorm(input_size)

  def forward(self, agents, my_id):
    # Pull out rows corresponding to the agent
    agent_ids = agents[:, :, EntityId]
    valid_mask = agent_ids==0
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
    )

    agent_embeddings = self.dense(agents)

    agent_embeddings = self.transformer(agent_embeddings,mask=valid_mask)

    my_agent_embeddings = agent_embeddings[
        torch.arange(agents.shape[0]), row_indices
    ]

    # Project to input of recurrent size
    agent_embeddings = self.agent_fc(agent_embeddings)
    my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
    my_agent_embeddings = F.relu(my_agent_embeddings)

    return self.final_norm(agent_embeddings), self.final_norm(my_agent_embeddings)


class ItemEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_heads: int = 8):
    super().__init__()


    item_embed_dim=16
    hidden_embed_dim1=32
    hidden_embed_dim2=64
    hidden_embed_dim3=128
    self.dense1 = Dense(item_embed_dim, hidden_embed_dim2, hidden_layers=[hidden_embed_dim1], activation="SiLU", norm_layer="LayerNorm")
    self.dense2 = Dense(hidden_embed_dim2, hidden_size, hidden_layers=[hidden_embed_dim3], activation="SiLU", norm_layer="LayerNorm")
    self.final_norm = torch.nn.LayerNorm(hidden_size)
    self.resnet = ResNet(num_layers=4, input_size=hidden_size, hidden_layers=[hidden_embed_dim3], activation="SiLU", norm_layer='LayerNorm',dropout=0.1)

  def forward(self, items):

    item_embeddings = self.dense1(items)
    item_embeddings = self.dense2(item_embeddings)
    item_embeddings = self.resnet(item_embeddings)
    item_embeddings = self.final_norm(item_embeddings)
    return item_embeddings


class InventoryEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.pooling = CLSPooling(embed_dim= input_size, CLSidx=0, num_heads= 8, dropout= 0.0)
    self.cls_token = torch.nn.Parameter(torch.randn(1,1,input_size))
    self.dense = Dense(input_size=input_size,output_size=input_size,hidden_layers=[128],activation='SiLU',norm_layer='LayerNorm')
    self.final_norm = torch.nn.LayerNorm(input_size)

  def forward(self, inventory, items_unembedded):

    batch, items, embed_dim = inventory.shape

    batch2, items2, embed_dim2 = items_unembedded.shape

    cls_item = torch.ones(batch2, 1, embed_dim2,device=inventory.device)
    cls_item = cls_item.int()

    items_unembedded = torch.cat([cls_item,items_unembedded],dim=1)

    inventory_ids = items_unembedded[:, :, EntityId]
    valid_mask = inventory_ids==0

    cls_repeat = torch.repeat_interleave(self.cls_token, batch, dim=0)
    inventory_cls = torch.cat([cls_repeat,inventory],dim=1)

    CLS_token = self.pooling(inventory_cls,inventory_cls,inventory_cls,valid_mask)

    return self.final_norm(self.dense(CLS_token.squeeze(1)) )


class MarketEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    #self.pooling = GlobalAttentionPooling(input_size=input_size)
    self.pooling = CLSPooling(embed_dim= input_size, CLSidx=0, num_heads= 8, dropout= 0.0)
    self.cls_token = torch.nn.Parameter(torch.randn(1,1,input_size))
    self.dense = Dense(input_size=input_size,output_size=input_size,hidden_layers=[128],activation='SiLU',norm_layer='LayerNorm')
    self.final_norm = torch.nn.LayerNorm(input_size)

  def forward(self, market, items_unembedded):

    batch, items, embed_dim = market.shape

    batch2, items2, embed_dim2 = items_unembedded.shape

    cls_item = torch.ones(batch2, 1, embed_dim2,device=market.device)
    cls_item = cls_item.int()

    items_unembedded = torch.cat([cls_item,items_unembedded],dim=1)

    market_ids = items_unembedded[:, :, EntityId]
    valid_mask = market_ids==0

    cls_repeat = torch.repeat_interleave(self.cls_token, batch, dim=0)
    market_cls = torch.cat([cls_repeat,market],dim=1)

    CLS_token = self.pooling(market_cls,market_cls,market_cls,valid_mask)

    return self.final_norm(self.dense(CLS_token.squeeze(1)) )


class TaskEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, task_size):
    super().__init__()

    self.resnet = ResNet(num_layers=16, input_size=task_size, hidden_layers=[1024], activation="SiLU", norm_layer='LayerNorm',dropout=0.1)
    self.fc = torch.nn.Linear(task_size, input_size)
    self.final_norm = torch.nn.LayerNorm(input_size)

  def forward(self, task):
    return self.final_norm(self.fc(self.resnet(task.clone())))


class ActionDecoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.layers = torch.nn.ModuleDict(
        {
            "attack_style": torch.nn.Linear(hidden_size, 3),
            "attack_target": torch.nn.Linear(hidden_size*2, 1),
            "market_buy": torch.nn.Linear(hidden_size*2, 1),
            "inventory_destroy": torch.nn.Linear(hidden_size*2, 1),
            "inventory_give_item": torch.nn.Linear(hidden_size*2, 1),
            "inventory_give_player": torch.nn.Linear(hidden_size*2, 1),
            "gold_quantity": torch.nn.Linear(hidden_size, 99),
            "gold_target": torch.nn.Linear(hidden_size*2, 1),
            "move": torch.nn.Linear(hidden_size, 5),
            "inventory_sell": torch.nn.Linear(hidden_size*2, 1),
            "inventory_price": torch.nn.Linear(hidden_size, 99),
            "inventory_use": torch.nn.Linear(hidden_size*2, 1),
        }
    )

  def apply_layer(self, layer, embeddings, mask, hidden):

    if hidden.dim() == 2 and embeddings is not None:
      hidden = layer(embeddings)
      hidden = hidden.squeeze(-1)
    else:
      hidden = layer(hidden)

    if mask is not None:
      hidden = hidden.masked_fill(mask == 0, -1e9)

    return hidden

  def forward(self, hidden, lookup):
    (
        player_embeddings,
        inventory_embeddings,
        market_embeddings,
        action_targets,
    ) = lookup

    embeddings = {
        "attack_target": torch.cat([player_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), player_embeddings.shape[1], dim=1)],dim=-1),
        "market_buy": torch.cat([market_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), market_embeddings.shape[1], dim=1)],dim=-1),
        "inventory_destroy": torch.cat([inventory_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), inventory_embeddings.shape[1], dim=1)],dim=-1),
        "inventory_give_item": torch.cat([inventory_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), inventory_embeddings.shape[1], dim=1)],dim=-1),
        "inventory_give_player": torch.cat([player_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), player_embeddings.shape[1], dim=1)],dim=-1),
        "gold_target": torch.cat([player_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), player_embeddings.shape[1], dim=1)],dim=-1),
        "inventory_sell": torch.cat([inventory_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), inventory_embeddings.shape[1], dim=1)],dim=-1),
        "inventory_use": torch.cat([inventory_embeddings,torch.repeat_interleave(hidden.unsqueeze(1), inventory_embeddings.shape[1], dim=1)],dim=-1),
    }

    action_targets = {
        "attack_style": action_targets["Attack"]["Style"],
        "attack_target": action_targets["Attack"]["Target"],
        "market_buy": action_targets["Buy"]["MarketItem"],
        "inventory_destroy": action_targets["Destroy"]["InventoryItem"],
        "inventory_give_item": action_targets["Give"]["InventoryItem"],
        "inventory_give_player": action_targets["Give"]["Target"],
        "gold_quantity": action_targets["GiveGold"]["Price"],
        "gold_target": action_targets["GiveGold"]["Target"],
        "move": action_targets["Move"]["Direction"],
        "inventory_sell": action_targets["Sell"]["InventoryItem"],
        "inventory_price": action_targets["Sell"]["Price"],
        "inventory_use": action_targets["Use"]["InventoryItem"],
    }

    actions = []
    for key, layer in self.layers.items():
      mask = None
      mask = action_targets[key]
      embs = embeddings.get(key)
      if embs is not None and embs.shape[1] != mask.shape[1]:
        b, _, f = embs.shape
        zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
        embs = torch.cat([embs, zeros], dim=1)

      action = self.apply_layer(layer, embs, mask, hidden)
      actions.append(action)

    return actions
