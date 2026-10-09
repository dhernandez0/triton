from ..._core import ir, builtin, _unwrap_if_constexpr
from ..._semantic import _check
from triton.experimental.gluon.language._layouts import DistributedLayout
from ..cdna4.async_copy import commit_group, wait_group
from .._ops import _load_cache_policy, _store_cache_policy

__all__ = ["global_to_shared", "shared_to_global", "commit_group", "wait_group", "mbarrier_arrive"]


@builtin
def global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="", eviction_policy="", _semantic=None):
    """
    Asynchronously copy elements from global memory to shared memory.

    Requires manual syncronization via :func:`wait_group` before accessing the loaded data.

    Args:
        smem (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None(0).
        cache_modifier (str): Load cache modifier: ".ca", ".cg", ".cs" or ".cv". Defaults to "".
        eviction_policy (str): "evict_last" sets the high-temporal (TH_LOAD_HT) hint and "evict_first" sets the
            non-temporal (TH_LOAD_NT) hint. Ignored when cache_modifier is ".cs" or ".cv". Defaults to "".
    """
    _check(pointer.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(pointer.type.layout, DistributedLayout),
           lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)
    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    cache_policy = _load_cache_policy(cache_modifier, eviction_policy, _semantic)
    cache_policy = cache_policy._to_ir(_semantic.builder)
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_policy, False)


@builtin
def shared_to_global(pointer, smem, mask=None, cache_modifier="", eviction_policy="", _semantic=None):
    """
    Asynchronously copy elements from shared memory to global memory.

    Requires manual syncronization via :func:`wait_group` before accessing the stored data.

    Args:
        pointer (tensor): Destination pointer tensor.
        smem (shared_memory_descriptor): Source shared memory descriptor.
        mask (tensor, optional): Mask tensor for predicated stores. Defaults to None.
        cache_modifier (str): Store cache modifier: ".wb", ".cg", ".cs" or ".wt". Defaults to "".
        eviction_policy (str): "evict_last" sets the high-temporal (TH_STORE_HT) hint and "evict_first" sets the
            non-temporal (TH_STORE_NT) hint. Ignored when cache_modifier is ".cs" or ".wt". Defaults to "".
    """
    _check(pointer.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(pointer.type.layout, DistributedLayout),
           lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    mask_handle = mask.handle if mask is not None else ir.value()
    cache_policy = _store_cache_policy(cache_modifier, eviction_policy, _semantic)
    cache_policy = cache_policy._to_ir(_semantic.builder)
    _semantic.builder.create_async_copy_local_to_global(smem.handle, pointer.handle, mask_handle, cache_policy)


@builtin
def mbarrier_arrive(mbarrier, _semantic=None):
    """
    Arrive on the mbarrier once all outstanding async copies are complete.

    Args:
        mbarrier (shared_memory_descriptor): Barrier object to arrive on.
    """
    _semantic.builder.create_async_copy_lds_barrier_arrive(mbarrier.handle)
