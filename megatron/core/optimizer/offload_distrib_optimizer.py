import torch

from typing import *
from .optimizer import multi_tensor_scale_impl, multi_tensor_applier
from .distrib_optimizer import DistributedOptimizer
from .. import tensor_parallel
from .clip_grads import get_grad_norm_fp32
from ..transformer.module import param_is_not_shared
from .hybrid_adam import CPUAdam
from .optimizer_config import OptimizerConfig
from .grad_scaler import MegatronGradScaler
from ..distributed import ParamAndGradBuffer
from .optimizer import _zero_grad_group_helper

__all__ = ['OffloadDistributedOptimizer']
class OffloadDistributedOptimizer(DistributedOptimizer):

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
        cpu_offload_fraction: float = 0.5
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []
        shard_fp32_from_float32_groups = []

        # NOTE: first pass -- collect numel for each parameter tensor
        numel_for_model_params = {}
        for group_range in opt_group_ranges:
            for model_param in group_range["params"]:
                assert model_param.requires_grad
                assert id(model_param) not in numel_for_model_params
                numel_for_model_params[id(model_param)] = model_param.numel()

        total_numel = sum(numel_for_model_params.values())
        cpu_numel = int(cpu_offload_fraction * total_numel)
        offload_model_param_ids = []
        cur_cpu_numel = 0
        if cpu_numel > 0:
            for k, v in sorted(numel_for_model_params.items(), key=lambda x: x[1], reverse=True):
                if cur_cpu_numel + v <= cpu_numel:
                    offload_model_param_ids.append(k)
                    cur_cpu_numel += v

        # NOTE: Here to log actual cpu faction of this process
        if torch.distributed.get_rank() == 0:
            print(f'Fraction is: {cur_cpu_numel / total_numel}')

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            shard_fp32_from_float32_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)

            # Views of each sharded parameters
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)

            # Hybrid FP32 copies of sharded parameters 
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)
            shard_fp32_from_float32_groups.append(shard_fp32_from_float32_params_this_group)

            for model_param in group_range["params"]:
                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start : param_range.end
                    ]
                    if id(model_param) in offload_model_param_ids:
                        shard_main_param = shard_model_param.float().cpu()
                    else:
                        shard_main_param = shard_model_param.clone().float()

                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    if id(model_param) in offload_model_param_ids:
                        # Clone model -> main.
                        shard_model_param = model_param.detach().view(-1)[
                            param_range.start : param_range.end
                        ]

                        shard_main_param = shard_model_param.cpu()
                        tensor_parallel.copy_tensor_model_parallel_attributes(
                            shard_model_param, model_param
                        )
                        tensor_parallel.copy_tensor_model_parallel_attributes(
                            shard_main_param, model_param
                        )
                        if hasattr(model_param, 'shared'):
                            shard_model_param.shared = model_param.shared
                            shard_main_param.shared = model_param.shared
                    else:
                        shard_main_param = shard_model_param
                        tensor_parallel.copy_tensor_model_parallel_attributes(
                            shard_model_param, model_param
                        )
                        if hasattr(model_param, 'shared'):
                            shard_model_param.shared = model_param.shared

                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    shard_fp32_from_float32_params_this_group.append(shard_main_param)

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params. [Hybrid]
            group_range["orig_group"]["params"] = [
                *shard_fp32_from_float32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            # NOTE: Here is a hack to reuse super().__init__()
            (shard_fp32_from_float16_groups, shard_fp32_from_float32_groups)
        )

        
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            config: OptimizerConfig,
            grad_scaler: MegatronGradScaler,
            init_state_fn: Optional[Callable],
            per_model_buffers: Dict[int, List[ParamAndGradBuffer]],
            data_parallel_group: torch.distributed.ProcessGroup,
            data_parallel_group_gloo: torch.distributed.ProcessGroup,
            data_parallel_group_idx: int,
            cpu_offload_fraction: float = 0.5,
        ):
        self.cpu_offload_fraction = cpu_offload_fraction
        assert 0 <= cpu_offload_fraction <= 1, "Offload fraction should be in [0, 1] !"
        assert isinstance(
            optimizer, CPUAdam
        ), "Only CPUAdam currently supported, due to checkpointing requirements."        
        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            per_model_buffers,
            data_parallel_group,
            data_parallel_group_gloo,
            data_parallel_group_idx
        )
        # TODO: It'd better to combine these two buffers.
        self.shard_fp32_from_float16_groups, self.shard_fp32_from_float32_groups = self.shard_fp32_from_float16_groups
        
        
        
        pass

    def zero_grad(self, set_to_none: bool = True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.

        Args:
            set_to_none (bool): if true, set grads to None.
        """
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_fp32_from_float16_groups,
            self.shard_fp32_from_float32_groups
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)

        # If overlapping param all-gather with forward compute, launch all-gather
        # for first accessed bucket here before forward compute is initiated.
        # The all-gather for the next bucket will be launched in the forward
        # pre-hook when this all-gather finishes (to ensure that the communication
        # kernels don't head-of-line block the compute kernels since we run with
        # CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence parallelism).
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)


    def preprocess_grads(self) -> bool:
        """
            this function temperorarily generates a fp32 grad on cuda
            and use grad_norm_clip then copy them to cpu
        """
        timers = self.config.timers
        # 1. collect fp32 grads from fp16 model
        params = None
        main_grads, main_param_id_to_main_grad_mapping = self._collect_grads()

        # 2. unscale / check inf
        # Reset found inf.
        if self.grad_scaler:
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )

            self.found_inf.fill_(0.0)

            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self.found_inf, self.grad_scaler.inv_scale
            )

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()
            )

            # Check for nan.
            found_inf_flag = self.found_inf.item() > 0
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            if found_inf_flag:
                return False, None, None

        # 3. compute grad norm and clip
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        def _internal_get_main_grads_for_grad_norm(params):
            grads_for_norm = []
            for param in params:
                # O(n) to O(n^2)
                if id(param) not in main_param_id_to_main_grad_mapping:
                    continue
                grad = main_param_id_to_main_grad_mapping[id(param)]
                is_not_shared = param_is_not_shared(param)
                is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
                if is_not_shared and is_not_tp_duplicate:
                    grads_for_norm.append(grad)
            return grads_for_norm
        
        def _internal_clip_grad_by_total_norm_fp32(
            main_grads: Union[List[torch.Tensor], torch.Tensor],
            max_norm: Union[int, float],
            total_norm: float,
        ):
            # Grads.
            grads = []
            for g in main_grads:
                assert g.type() == 'torch.cuda.FloatTensor'
                grads.append(g.detach())

            # Scale.
            clip_coeff = max_norm / (total_norm + 1.0e-6)
            if clip_coeff < 1.0:
                dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
                multi_tensor_applier(multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff)

        if self.config.clip_grad > 0.0:
            params = self.get_parameters()
            grads_for_norm = _internal_get_main_grads_for_grad_norm(params)
            grad_norm = get_grad_norm_fp32(
                grads_for_norm, model_parallel_group=self.get_model_parallel_group()
            )
            _internal_clip_grad_by_total_norm_fp32(main_grads, self.config.clip_grad, grad_norm)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        if timers is not None:
            timers('optimizer-copy-grad-to-cpu-and-gpu', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        # 4. move these grads to CPU
        self._dispatch_grads(params, main_param_id_to_main_grad_mapping)

        if timers is not None:
            timers('optimizer-copy-grad-to-cpu-and-gpu').stop()

        return True, grad_norm, None

    def _get_model_and_main_params_data_float32(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.shard_float16_groups, self.shard_fp32_from_float32_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _collect_grads(self):
        shard_main_param_id_to_shard_main_grad_mapping = {}
        shard_main_grads = []

        # Utility method for copying group grads.
        def collect_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]

                    shard_main_grads.append(shard_model_grad.float())
                    shard_main_param_id_to_shard_main_grad_mapping[id(shard_main_param)] = shard_main_grads[-1]

        # Copy model groups to shard groups.
        collect_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        collect_group_grads(self.model_fp32_groups, self.shard_fp32_from_float32_groups)
        return shard_main_grads, shard_main_param_id_to_shard_main_grad_mapping

    def _dispatch_grads(self, params, main_param_id_to_main_grad_mapping):
        if params is None:
            params = self.get_parameters()
        for param in params:
            if id(param) in main_param_id_to_main_grad_mapping:
                if param.grad is not None:
                    param.grad.data.copy_(main_param_id_to_main_grad_mapping[id(param)])
                else:
                    param.grad = main_param_id_to_main_grad_mapping[id(param)].to(param.device)

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]
                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_from_float32_groups, self.model_fp32_groups)

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_from_float32_groups)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        self.update_successful = super().step_with_ready_grads()

        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)
        # If not overlapping all-gather for parameters, launch synchronous all-gather
        # communication calls here. If overlapping all-gather for parameters, the following
        # call to _gather_all_model_params is a no-op: the first all-gather is launched
        # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
        # are launched in the forward pre-hook.
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
        if timers is not None:
            timers('params-all-gather').stop()

        return self.update_successful

    @torch.no_grad()
    def step(self):
        success, grad_norm, num_zeros_in_grad = self.preprocess_grads()
        if success:
            success = self.step_with_ready_grads()
        # Successful update.
        return success, grad_norm, num_zeros_in_grad
