# Batch v1 vs v2 semantics

v1: `_flush_pool('stream_end')` called at end of EVERY batch.
v2: pool persists across batches; capacity flush only at 100; terminal flush exactly once at true stream end.

Witness: v1 pool_size_after_batch1=0, v2=2.
