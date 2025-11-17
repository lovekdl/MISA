import torch
from torch.optim import SGD
from transformers.optimization import AdamW
from torch import nn
from transformers import Trainer
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from galore_torch import GaLoreAdamW

from transformers.utils import (
    logging,
)
from misa.optimizer import BlockCoordinateDesentOptimizer
from misa.clip_grad_norm import clip_grad_norm_for_sparse_tensor
from types import MethodType

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

logger = logging.get_logger(__name__)

class CustomizedTrainer(Trainer):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.use_bcd :
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
    
    def create_optimizer(self) -> "torch.optim.Optimizer":
        args = self.args
        print(f"Optimizer is {args.optim}")
        if args.use_galore :
            galore_params = []
            for module_name, module in self.model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                target_modules_list = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj"]
                if not any(target_key in module_name for target_key in target_modules_list):
                    continue

                print('enable GaLore for weights in module: ', module_name)
                galore_params.append(module.weight)

            id_galore_params = [id(p) for p in galore_params]
            regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params]
            param_groups = [{'params': regular_params}, 
                            {'params': galore_params, 'rank': args.galore_r, 'update_proj_gap': 50, 'scale': args.galore_alpha, 'proj_type': 'std'}]
            self.optimizer = GaLoreAdamW(param_groups, lr=args.learning_rate)
        if args.use_bcd == True : 
            if args.optim == "sgd":
                base_optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            elif args.optim == "adamw_hf":
                base_optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
            else : raise NotImplementedError("For BCD, we only support 2 optimizers: sgd and adamw_hf")
            print(f"Creating BCD optimizer...")
            self.optimizer = BlockCoordinateDesentOptimizer(
                base_optimizer=base_optimizer,
                named_parameters_list=list(self.model.named_parameters()),
                activated_layers=args.bcd_activated_layers,
                interval_steps=args.bcd_interval_steps,
                update_order=args.bcd_update_order,
                granularity=args.granularity,
                sample_last=args.sample_last,
                param_ratio_limit=args.param_ratio_limit,
                G_clip = args.G_clip,
                include_embedding_and_lm_head=args.include_embedding_and_lm_head,
                mix_lora = args.mix_lora,
                misa_eta=args.misa_eta, 
            )
                
        return super().create_optimizer()

    