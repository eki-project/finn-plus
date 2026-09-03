"""Structural fpgadataflow ops with no backend specialization.

A transformation lowers or replaces each of these before synthesis:
``StreamingDataflowPartition`` becomes a stitched IP at the partition boundary,
``Shuffle`` is decomposed into ``InnerShuffle`` / ``OuterShuffle``."""
