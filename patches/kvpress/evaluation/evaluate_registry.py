# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Allow importing vector from the parent project
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from vector.kvpress import (
    VECTORPress,
    VECTORWrapperPress,
    KeyDiffVECTORPress,
    KeyDiffVECTORV2KPress,
    KVzipVECTORPress,
    SnapKVVECTORPress,
)


from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from benchmarks.infinite_bench.calculate_metrics import (
    calculate_metrics as infinite_bench_scorer,
)
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import (
    calculate_metrics_e as longbench_scorer_e,
)
from benchmarks.longbenchv2.calculate_metrics import (
    calculate_metrics as longbenchv2_scorer,
)
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from benchmarks.needle_in_haystack.calculate_metrics import (
    calculate_metrics as needle_in_haystack_scorer,
)
from benchmarks.zero_scrolls.calculate_metrics import (
    calculate_metrics as zero_scrolls_scorer,
)

from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "aime25": aime25_scorer,
    "math500": math500_scorer,
}


PRESS_REGISTRY = {
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(
        ExpectedAttentionPress(use_vnorm=False)
    ),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(
        ExpectedAttentionPress(use_vnorm=False)
    ),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "cur": CURPress(),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "fastkvzip": FastKVzipPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvzip": KVzipPress(),
    "kvzip_plus": KVzipPress(kvzip_plus_normalization=True),
    "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
    "kvzap_mlp_head": KVzapPress(model_type="mlp"),
    "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
    "lagkv": LagKVPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "vector_global": VECTORPress(budget_allocation="global_uniform"),
    "vector_uniform": VECTORPress(budget_allocation="per_head_uniform"),
    "vector_dynamic": VECTORPress(budget_allocation="per_head_dynamic"),
    "keydiff_vector": KeyDiffVECTORPress(),
    "keydiff_v2k": KeyDiffVECTORV2KPress(),
    "snapkv_vector": SnapKVVECTORPress(),
    "kvzip_vector": KVzipVECTORPress(),
    "kvzip_vector_plus": KVzipVECTORPress(kvzip_plus_normalization=True),
    # PyramidKV + VECTOR via generic wrapper.
    # __post_init__ sets press.window_size = n_local=4, aligning PyramidKV's
    # budget window with VECTOR's protected local region.
    "pyramidkv_vector": VECTORWrapperPress(press=PyramidKVPress()),
    "no_press": None,
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(
        base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))
    ),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
}
