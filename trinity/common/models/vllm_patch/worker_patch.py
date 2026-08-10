from types import MethodType
from typing import Optional

import torch
import vllm
from packaging.version import parse as parse_version
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from trinity.common.models.vllm_patch import (
    PROMPT_LOGPROBS_START_ARG,
    get_vllm_version,
)


def _get_prompt_logprobs_start(request, num_prompt_tokens: int) -> int:
    """Return the first prompt-logprob row that must be materialized.

    Prompt-logprob row ``i`` scores prompt token ``i + 1``. The default of
    zero deliberately preserves vLLM's normal full-prompt behavior. Trinity's
    distillation workflow sets the extra argument to the first response row so
    that the teacher does not compute or transfer top-k values for history.
    """
    extra_args = request.sampling_params.extra_args or {}
    start = extra_args.get(PROMPT_LOGPROBS_START_ARG, 0)
    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError(f"{PROMPT_LOGPROBS_START_ARG} must be an integer, got {start!r}")
    max_start = num_prompt_tokens - 1
    if not 0 <= start <= max_start:
        raise ValueError(
            f"{PROMPT_LOGPROBS_START_ARG} must be in [0, {max_start}], got {start}"
        )
    return start


def _prompt_logprobs_chunk_overlap(
    start_idx: int,
    num_logits: int,
    output_start: int,
) -> Optional[tuple[int, int, int]]:
    """Map a full-prompt prefill chunk into suffix-only output tensors.

    Returns ``(hidden_offset, selected_count, output_offset)`` or ``None`` if
    the chunk is wholly before the requested suffix.
    """
    overlap_start = max(start_idx, output_start)
    chunk_end = start_idx + num_logits
    if overlap_start >= chunk_end:
        return None
    return (
        overlap_start - start_idx,
        chunk_end - overlap_start,
        overlap_start - output_start,
    )


def patch_vllm_prompt_logprobs(model_runner: GPUModelRunner):  # noqa: C901
    """Patch vLLM model runner to support prompt logprobs extraction."""
    version = get_vllm_version()
    if version < parse_version("0.10.2") or version > parse_version("0.14.1"):
        raise ValueError(
            f"Unsupported vllm version: {vllm.__version__}. "
            "This patch requires vllm version >= 0.10.2, <= 0.14.1."
        )
    is_v0102 = version == parse_version("0.10.2")

    def _get_prompt_logprobs_dict_v11(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        """Patched version of _get_prompt_logprobs_dict.

        This is a monkey-patched version of `_get_prompt_logprobs_dict` from
        `vllm.v1.worker.gpu_model_runner.GPUModelRunner` (vLLM versions
        0.10.2 to 0.11.0).

        The original function does not apply temperature scaling to logits when
        calculating prompt logprobs, which can lead to incorrect logprob values
        when the temperature is not 1.0. This patch adds the missing
        temperature scaling.
        """
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            output_start = _get_prompt_logprobs_start(request, num_prompt_tokens)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Allocate only the requested suffix. If chunked, we'll copy
                # the overlapping rows in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1 - output_start, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            overlap = _prompt_logprobs_chunk_overlap(start_idx, num_logits, output_start)
            if overlap is None:
                # History still participates in Transformer prefill, but we
                # skip its vocabulary logits/top-k diagnostics entirely.
                continue
            hidden_offset, selected_count, output_offset = overlap

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[
                offset + hidden_offset : offset + hidden_offset + selected_count
            ]
            # PATCH START
            if is_v0102:
                logits = self.model.compute_logits(prompt_hidden_states, None)
            else:
                logits = self.model.compute_logits(prompt_hidden_states)

            temp = request.sampling_params.temperature
            if temp >= 1e-5:
                logits.div_(temp)
            # PATCH END

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            target_start_tok = start_tok + hidden_offset
            tgt_token_ids = torch.as_tensor(
                request.prompt_token_ids[
                    target_start_tok : target_start_tok + selected_count
                ],
                dtype=torch.long,
                device=self.device,
            )

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(output_offset, output_offset + selected_count)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_prompt_logprobs_dict_v12(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        """Patched version of _get_prompt_logprobs_dict.

        This is a monkey-patched version of `_get_prompt_logprobs_dict` from
        `vllm.v1.worker.gpu_model_runner.GPUModelRunner` (vLLM versions
        0.12.0 to 0.14.1).

        The original function does not apply temperature scaling to logits when
        calculating prompt logprobs, which can lead to incorrect logprob values
        when the temperature is not 1.0. This patch adds the missing
        temperature scaling.
        """
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            output_start = _get_prompt_logprobs_start(request, num_prompt_tokens)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Allocate only the requested suffix. If chunked, we'll copy
                # the overlapping rows in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1 - output_start, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            overlap = _prompt_logprobs_chunk_overlap(start_idx, num_logits, output_start)
            if overlap is None:
                # History still participates in Transformer prefill, but we
                # skip its vocabulary logits/top-k diagnostics entirely.
                continue
            hidden_offset, selected_count, output_offset = overlap

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[
                offset + hidden_offset : offset + hidden_offset + selected_count
            ]
            logits = self.model.compute_logits(prompt_hidden_states)

            # PATCH START
            temp = request.sampling_params.temperature
            if temp >= 1e-5:
                logits.div_(temp)
            # PATCH END

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            target_start_tok = start_tok + hidden_offset
            tgt_token_ids = torch.as_tensor(
                request.prompt_token_ids[
                    target_start_tok : target_start_tok + selected_count
                ],
                dtype=torch.long,
                device=self.device,
            )

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(output_offset, output_offset + selected_count)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    if get_vllm_version() < parse_version("0.12.0"):
        model_runner._get_prompt_logprobs_dict = MethodType(
            _get_prompt_logprobs_dict_v11, model_runner
        )
    else:
        model_runner._get_prompt_logprobs_dict = MethodType(
            _get_prompt_logprobs_dict_v12, model_runner
        )
