import torch
import random
from torch.optim import Optimizer
from typing import Dict
import math
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import numpy as np
import math

class BlockCoordinateDesentOptimizer(Optimizer):
    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        interval_steps = 50,
        include_embedding_and_lm_head=False,
        activated_layers=1,
        misa_eta=1,
        update_order="misa",
        grad_beta = 0.8,
        granularity='module', 
        param_ratio_limit = 0.03, 
        sample_last = 2, 
        G_clip = 5,  
        mix_lora = False,
    ):
        
        self.granularity = granularity
        self.module_names = []
        self.mix_lora = mix_lora
        for name, param in named_parameters_list:
            if "lora" in name:
                self.mix_lora = True
                break
        if self.mix_lora:
            self.granularity = 'module'
        self.param_to_id={}
        self.total_params_num = 0
        block_prefix_list, other_params = self.infer_param_groups(named_parameters_list)
        for name, param in named_parameters_list:
            for i in range(len(block_prefix_list)):
                if block_prefix_list[i][0] in name:
                    assert param not in self.param_to_id
                    self.param_to_id[param] = i
            if not self.mix_lora:
                self.total_params_num += param.numel()
            elif 'lora' in name :
                self.total_params_num += param.numel()

        self.G_clip = G_clip
        self.activated_layers = activated_layers
        self.interval_steps = interval_steps
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.other_params = other_params
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.defaults = base_optimizer.defaults
        self.active_layers_indices = []
        self.include_embedding_and_lm_head = include_embedding_and_lm_head
        self.update_order = update_order
        

        self.skip_nan = False
        self.total_layers = self.total_layers 
        self.block_prefix_list=self.block_prefix_list[0:]
        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        self.lp_to_hp = {}
        self.layers_param_number = [0] * self.total_layers
        self.layer_selected_times = [0] * self.total_layers
        self.embed_param_number = 0
        self.lm_head_param_number = 0
        self.calculate_param_numbers(named_parameters_list)

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict
    

        self.grad_beta = grad_beta
        self.misa_eta = misa_eta
        self.param_ratio_limit = param_ratio_limit

        
        self.sample_last = sample_last
        self.last_used = [-1] * self.total_layers
        self.G = [0] * self.total_layers
        self.G_cur = [0] * self.total_layers

        self.sampled_times = [1] * self.total_layers
        self.p = [1.0/self.total_layers] * self.total_layers

        super().__init__(self.param_groups, base_optimizer.defaults)

        self.update_trainable_params()

    def infer_param_groups(self, named_parameters_list):
        block_prefix_list = []
        other_params = []
        layers_pattern = r'.*layers.[^.]*\.'
        layer_pattern = r'.*layer.[^.]*\.'

        import re
        if self.mix_lora:
            for name, param in named_parameters_list:
                if 'lora' in name :
                    block_prefix_list.append([name])
                    self.module_names.append(name)
        elif self.granularity == 'module' :
            for name, param in named_parameters_list:
                if 'layer' not in name or len(param.shape) < 2 :
                    continue
                block_prefix_list.append([name])
                self.module_names.append(name)
        elif self.granularity == 'layer' : 
            for name, param in named_parameters_list:
                # print(name, block_prefix_list)
                if any(prefix[0] in name for prefix in block_prefix_list):
                    continue
                
                if re.findall(layers_pattern, name) and "lm_head" not in name:
                    block_prefix_list.append(re.findall(layers_pattern, name))
                elif re.findall(layer_pattern, name) and "lm_head" not in name:
                    block_prefix_list.append(re.findall(layer_pattern, name))
                else: 
                    other_params.append(name)
                    
        self.total_layers = len(block_prefix_list)
        return block_prefix_list, other_params
    

    def calculate_param_numbers(self, named_param_list) :
        embed_pattern = r'.*embed[^.]*\.'
        import re

        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        for name, param in self.named_parameters_list:
            is_layer_param = False
            for i in range(self.total_layers) :
                if(self.block_prefix_list[i][0] in name) :
                    self.layers_param_number[i] += param.numel()
                    is_layer_param = True
                    break
            if not is_layer_param :
                if re.findall(embed_pattern, name):
                    self.embed_param_number += param.numel()
                elif "lm_head" in name:
                    self.lm_head_param_number += param.numel()
        


    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]
        
    def step(self, *args, **kwargs) -> None:
        self.record_mark = True
        
        self._update_lr()
        if not self.mix_lora: 
            self._grad_to_hp()
        if "misa" in self.update_order :
            self.calculate_grad_norm_for_each_layer()
        if not self.skip_nan:
            self.base_optimizer.step(*args, **kwargs)
            if not self.mix_lora: 
                self._update_param()
        self._clean_hp_grad()
        self.skip_nan = False
        self.global_step += 1
        if self.global_step  % self.interval_steps == 0 or ("misa" in self.update_order and (-1 in self.last_used)):
            self.update_trainable_params()
        
        torch.cuda.empty_cache()

    def _clean_hp_grad(self) -> None:
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:

        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()
            if clear_lp_grads:
                lp_param.grad = None
                
    def calculate_grad_norm_for_each_layer(self) :
        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        for name, param in self.named_parameters_list:
            if not param.requires_grad or 'embed' in name or 'lm_head' in name : 
                continue
            for j in range(self.total_layers) :
                if(self.block_prefix_list[j][0] in name) :
                    break
            if not self.mix_lora: hp = self.lp_to_hp[param]
            else: hp = param
            current_grad_norm = torch.norm(hp.grad)
            if math.isnan(current_grad_norm):
                current_grad_norm = 1.0
                self.skip_nan = True
            for i in range(self.total_layers) :
                if(self.block_prefix_list[i][0] in name) :
                    self.current_grad_norms[i].append(current_grad_norm)
                    break
        for i in range(self.total_layers) :
            if len(self.current_grad_norms[i]) == 0 :
                continue
            try : 
                current_grad_norm = torch.norm(torch.tensor(self.current_grad_norms[i])).to(torch.float32).item()
            except RuntimeError :
                current_grad_norm = 0.9
            self.G_cur[i] += (current_grad_norm) / (self.interval_steps)


    def weighted_sample_without_replacement(self, population, weights, k):
        population = list(population)
        weights = list(weights)
        choosed_ratio = 0.0
        if self.granularity == "module" :
            k = 1000
        else :
            self.param_ratio_limit = 1
        selected = []
        w = [(x, id) for id, x in enumerate(self.last_used)]
        w = sorted(w, key=lambda item: item[0])
        for i in range(self.total_layers) :
            if (w[i][0] == -1 or i < self.sample_last) and k > 0:
                if choosed_ratio + self.layers_param_number[w[i][1]] / self.total_params_num <= self.param_ratio_limit:
                    choosed_ratio += self.layers_param_number[w[i][1]] / self.total_params_num
                    selected.append(w[i][1])
                    k -= 1
                index = population.index(w[i][1])
                del population[index]
                del weights[index]
        while len(population) and k > 0 :
            chosen = random.choices(population, weights=weights, k=1)[0]
            if choosed_ratio + self.layers_param_number[chosen] / self.total_params_num <= self.param_ratio_limit:
                choosed_ratio += self.layers_param_number[chosen] / self.total_params_num
                selected.append(chosen)
                k -= 1
            index = population.index(chosen)
            del population[index]
            del weights[index]
        return selected

    def update_trainable_params(self) :
        self.last_active_layers_indices=self.active_layers_indices

        if self.update_order == "random" :
            if self.granularity == 'layer':
                self.active_layers_indices = np.random.choice(range(self.total_layers), self.activated_layers, replace=False)
            elif self.granularity == 'module' :
                layer_selection_probabilities = [1] * self.total_layers
                assert len(layer_selection_probabilities) == self.total_layers
                self.active_layers_indices = self.weighted_sample_without_replacement(population=range(len(layer_selection_probabilities)), weights=layer_selection_probabilities, k=self.activated_layers)
            
        elif self.update_order == "ascending" :
            if len(self.active_layers_indices):
                st = self.active_layers_indices[-1] + 1
                self.active_layers_indices = []
                for i in range(st, st + self.activated_layers) :
                    self.active_layers_indices.append(i % self.total_layers)
            else :
                self.active_layers_indices = [i for i in range(self.activated_layers)]
        elif self.update_order == "descending" :
            if len(self.active_layers_indices):
                st = self.active_layers_indices[-1] - 1
                self.active_layers_indices = []
                for i in range(st, st - self.activated_layers, -1) :
                    self.active_layers_indices.append((i + self.total_layers) % self.total_layers)
            else :
                self.active_layers_indices = [i for i in range(self.activated_layers)]
                self.active_layers_indices.reverse()
        elif  'misa' in self.update_order :
            for i in range(self.total_layers) :
                if self.G_cur[i] == 0 :
                    continue
                self.G[i] = (1.0 - self.grad_beta) * self.G_cur[i] + self.grad_beta * self.G[i]
                self.G[i] /= (1.0 - (self.grad_beta**self.sampled_times[i]))
                self.sampled_times[i] += 1
                self.G_cur[i] = 0
            self.G = [g for g in self.G]
            g_sum = sum([math.exp(min(self.G_clip, self.misa_eta*g*g)) for g in self.G])
            layer_selection_probabilities = [ (math.exp(min(self.G_clip, self.misa_eta*g*g))/g_sum ) for g in self.G]
            self.p = layer_selection_probabilities
            assert len(layer_selection_probabilities) == self.total_layers
            self.active_layers_indices = self.weighted_sample_without_replacement(population=range(len(layer_selection_probabilities)), weights=layer_selection_probabilities, k=self.activated_layers)
        self.retain_indices = []
        for i in self.active_layers_indices:
            if i in self.last_active_layers_indices and 'misa' in self.update_order:
                self.retain_indices.append(i)
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        if not self.mix_lora:
            for n, p in self.named_parameters_list:
                if self.include_embedding_and_lm_head and ('embed' in n or 'lm_head' in n):
                    continue
                if p not in self.lp_to_hp :
                    continue
                hp = self.lp_to_hp[p]
                if self.param_to_id[p] not in self.retain_indices :
                    if hp in self.base_optimizer.state:
                        del self.base_optimizer.state[hp]
                    del self.lp_to_hp[p]
                    del hp
            
        print(f"Sample list: {self.active_layers_indices}", flush=True)
        for i in self.active_layers_indices :
            self.layer_selected_times[i] += 1
        self.active_param_prefixs = []
        for i in self.active_layers_indices :
            self.active_param_prefixs.append(self.block_prefix_list[i][0])
            self.last_used[i] = self.global_step
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]
        for i, (name, param) in enumerate(self.named_parameters_list):
            freezing_this_layer = False
            if not self.include_embedding_and_lm_head and not any(p in name for p in self.active_param_prefixs) :
                freezing_this_layer = True
            if self.include_embedding_and_lm_head and not any(p in name for p in self.active_param_prefixs) and ('embed' not in name) and ('lm_head' not in name) :
                freezing_this_layer = True
            if "classifier" in name :
                freezing_this_layer = False
            if self.include_embedding_and_lm_head and ('embed' in name or 'lm_head' in name):
                freezing_this_layer = False
            if freezing_this_layer:
                param.grad = None
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                if not self.mix_lora:
                    if self.include_embedding_and_lm_head and ('embed' in name or 'lm_head' in name) :
                        if param in self.lp_to_hp:
                            param_hp = self.lp_to_hp[param]
                            param_hp.requires_grad = True
                        else :
                            param_hp = param.clone().float().detach().to(param.device)
                            param_hp.requires_grad = True

                    elif self.param_to_id[param] not in self.retain_indices: 
                        param_hp = param.clone().float().detach().to(param.device)
                        param_hp.requires_grad = True
                    else:
                        param_hp = self.lp_to_hp[param]
                        param_hp.requires_grad = True

                    self.param_idx2lp[i] = param
                    self.param_idx2hp[i] = param_hp
                    self.lp_to_hp[param] = param_hp

                else:
                    param_hp = param
                    self.lp_to_hp[param] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)

        self.base_optimizer.param_groups = active_param_groups
