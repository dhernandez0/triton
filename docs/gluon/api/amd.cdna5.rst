AMD CDNA 5
==========

.. currentmodule:: triton.experimental.gluon.language.amd.cdna5

GL2 cache residency
-------------------

``eviction_policy="evict_last"`` requests the high-temporal instruction hint,
which asks GL2 to retain the accessed lines preferentially. The hint is
supported by Gluon buffer loads and stores, asynchronous copies between global
and shared memory, and TDM load, store, fused-load, and prefetch operations.

Triton pointer loads and stores use the hint when they lower to AMD buffer
instructions or multicast cluster loads. Plain ``global_load`` and
``global_store`` instructions cannot carry it and ignore the eviction policy.

Streaming and bypass cache modifiers take precedence over an incompatible
eviction policy. In particular, ``.cs`` and ``.cv`` loads and ``.cs`` and
``.wt`` stores ignore ``evict_last`` and preserve the cache modifier's
non-temporal, last-use, or write-through behavior.

.. note::

   The instruction hint does not reserve GL2 capacity by itself. The
   application or framework must also call
   ``hipDeviceSetLimit(hipLimitPersistingL2CacheSize, bytes)`` to reserve GL2
   ways for persisting lines. Without a reservation, the hint is not guaranteed
   to improve residency or performance.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/gluon-module.rst

    async_copy
    cluster
    mbarrier
    tdm


.. autosummary::
    :toctree: generated
    :nosignatures:

    buffer_load
    buffer_store
    get_scaled_upcast_fp4_scale_layout
    get_wmma_scale_layout
    load_shared_fp4_repacked
    make_partitioned_dot_layouts
    scaled_downcast
    scaled_upcast
    wmma
    wmma_scaled
    PartitionedSharedLayout
