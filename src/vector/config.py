"""
VECTOR configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture constants for a supported HF model."""

    name: str
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    chunk_size: int = 32
    sink_size: int = 4
    local_size: int = 4
    adapter: str = "auto"

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        chunk_size: int = 32,
        sink_size: int = 4,
        local_size: int = 4,
        adapter: str = "auto",
    ) -> "ModelConfig":
        """Auto-populate dims from the HF model config (no GPU needed)."""
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        # Gemma3 is a multimodal model; AutoConfig returns the vision-language config
        # which wraps the text config. Unwrap to get the text-only architecture dims.
        if "gemma-3-" in name.lower():
            cfg = cfg.text_config
        assert hasattr(cfg, "hidden_size"), f"{name} config missing `hidden_size`"
        assert hasattr(cfg, "num_attention_heads"), (
            f"{name} config missing `num_attention_heads`"
        )
        assert hasattr(cfg, "num_key_value_heads"), (
            f"{name} config missing `num_key_value_heads`"
        )
        assert hasattr(cfg, "num_hidden_layers"), (
            f"{name} config missing `num_hidden_layers`"
        )
        assert hasattr(cfg, "head_dim"), f"{name} config missing `head_dim`"
        hidden_dim: int = cfg.hidden_size
        num_heads: int = cfg.num_attention_heads
        num_kv_heads: int = cfg.num_key_value_heads
        num_layers: int = cfg.num_hidden_layers
        head_dim: int = cfg.head_dim
        return cls(
            name=name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            chunk_size=chunk_size,
            sink_size=sink_size,
            local_size=local_size,
            adapter=adapter,
        )

    @property
    def kv_head_dim(self) -> int:
        """Merged KV dim: num_kv_heads × head_dim."""
        return self.num_kv_heads * self.head_dim

    @property
    def input_dim(self) -> int:
        """Feature dim for linear probe: hidden_dim + chunk_size (one-hot offset)."""
        return self.hidden_dim + self.chunk_size


@dataclass
class ProbeConfig:
    """Offline calibration settings."""

    target_layers: list[int] = field(default_factory=list)
    train_seqs: int = 10000
    seq_len: int = 4096
    lambda_ratio: float = 1e-4
    weights_path: Path = Path("weights/probe.pt")
    batch_size: int = 8
    val_seqs: int = 100
    calib_data: str = "c4"  # key of vector.data.calibration.CALIB_STREAMS

    def __post_init__(self) -> None:
        self.weights_path = Path(self.weights_path)
        assert self.calib_data in ("c4", "wikitext103"), (
            f"unknown calib_data {self.calib_data!r}"
        )


@dataclass
class CompressConfig:
    """KV cache compression settings."""

    mode: str = "vk"  # "v" | "vk"
    granularity: str = "chunk"  # "chunk" | "token"
    rank_family: str | None = (
        None  # "rank" | "weighted_rank" | "union" | "pareto" | "product"
    )
    rank_signals: list[str] | None = None
    rank_weights: list[float] | None = None
    union_split: list[float] | None = None
    norm_alpha: float = 1.0
    ratios: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 0.90])
    strategies: list[str] = field(
        default_factory=lambda: ["probe", "probe_evict", "random", "random_evict"]
    )
    # Baseline method parameters
    snapkv_window_size: int = 64
    snapkv_kernel_size: int = 5
    snapkv_pooling: str = "avgpool"  # "avgpool" | "maxpool"
    # H2O observation window: 0 = use ALL query positions (faithful to official H2O paper);
    # positive = use last N queries (memory-constrained approximation for very long sequences).
    h2o_window_size: int = 0
    # ManifoldKV windowing: 0 = global centroid; positive = window size in absolute token positions.
    manifoldkv_window_size: int = 4096
    # Fast KV Compaction (arXiv:2602.16284): OLS ridge regularisation λ (0 = ordinary LS).
    compaction_ridge_lambda: float = 0.0
    # Fast KV Compaction: observation window for attention scoring.
    # 0 = use ALL query positions (paper-faithful context_prefill default).
    compaction_obs_window_size: int = 0
    # Fast KV Compaction: β (log-weight bias) computation for AM-HighestAttnKeys.
    # "nnls" = solve NNLS to match partition function (paper default).
    # "zero" = β = 0 (simplified, lower quality).
    compaction_beta_method: str = "nnls"
    # Fast KV Compaction: number of projected-gradient-descent iterations for NNLS.
    # 0 = lstsq + clamp (no PGD); 2 = paper default.
    compaction_nnls_iters: int = 2
    # Fast KV Compaction: NNLS bound on B (B = exp(β)).
    # Reference defaults (base._nnls_pg): lower=1e-12, upper=None (no upper bound).
    compaction_nnls_lower_bound: float = 1e-12
    compaction_nnls_upper_bound: float = float("inf")  # no upper bound
    # Fast KV Compaction: Q reference strategy.
    # "context_prefill": use Q captured during prefill (self-study/off-policy, cheap).
    # "on_policy": re-derive Q from compressed earlier-layer states (paper reproduction
    #              standard, requires a second forward pass per layer; not yet implemented).
    compaction_q_ref: str = "context_prefill"
    # KVzip (snu-mllab/KVzip): chunk size for the context reconstruction scoring pass.
    # Each chunk gets one model forward with "Repeat the previous context exactly."
    # 2000 = KVzip paper default.
    kvzip_chunk_size: int = 2000
    # Token budget allocation strategy.
    # "global_uniform"   — same positions evicted for all heads (backward-compat default).
    # "per_head_uniform" — each head independently selects, same COUNT per head
    #                      (paper-faithful for H2O / SnapKV / KeyDiff / ManifoldKV).
    # "per_head_dynamic" — per-layer global top-k with fake-key masking (KVpress-compatible).
    # NOTE: evaluation entry scripts should pass --budget-allocation explicitly.
    budget_allocation: str = "global_uniform"
    # KeyDiff block-wise prefill eviction (arXiv:2504.15364 §2.4).
    # B > 0 triggers paper-faithful online eviction during prefill (paper default B=128).
    # 0 falls back to post-hoc eviction (deprecated, paper-unfaithful).
    keydiff_block_size: int = 128
    # Threeway approx budget: per-ratio fractions of eligible tokens to overwrite with OLS
    # predictions.  Length must match len(ratios) when provided.  None (default) → each
    # ratio's approx budget is ratio / 2.  Each value must satisfy approx_ratio <= the
    # corresponding compress_ratio (enforced per-call in select_threeway).
    threeway_approx_ratios: list[float] | None = None
    # threeway importance signal for Step 1 (eviction axis).
    # "k_cos_post_rope"     — sort by K cosine similarity ascending (low = important/distinctive).
    # "rank_sum_vnorm_kcos" — rank(k_cos_post_rope asc) + rank(v_norm desc).
    threeway_importance: str = "k_cos_post_rope"

    def __post_init__(self) -> None:
        if self.keydiff_block_size < 0:
            raise ValueError(
                f"keydiff_block_size must be >= 0, got {self.keydiff_block_size}"
            )
        _VALID_BUDGET = {"global_uniform", "per_head_uniform", "per_head_dynamic"}
        if self.budget_allocation not in _VALID_BUDGET:
            raise ValueError(
                f"budget_allocation must be one of {_VALID_BUDGET}, "
                f"got {self.budget_allocation!r}"
            )
        valid_rank_family = {"rank", "weighted_rank", "union", "pareto", "product"}
        valid_rank_signals = {
            "v_err",
            "k_err",
            "v_norm",
            "k_norm",
            "v_cos",
            "v_dist2_to_mean",
            "v_rd3",
            "v_rd_dist2",
            "k_cos_pre_rope",
            "k_cos_post_rope",
        }
        valid_modes = {"v", "vk"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode!r}")
        valid_granularity = {"chunk", "token"}
        if self.granularity not in valid_granularity:
            raise ValueError(
                "granularity must be one of "
                f"{valid_granularity}, got {self.granularity!r}"
            )
        if self.norm_alpha < 0.0:
            raise ValueError(f"norm_alpha must be >= 0, got {self.norm_alpha}")

        # Validate baseline params (always, regardless of strategies used)
        if self.snapkv_window_size <= 0:
            raise ValueError(
                f"snapkv_window_size must be > 0, got {self.snapkv_window_size}"
            )
        if self.snapkv_kernel_size < 1:
            raise ValueError(
                f"snapkv_kernel_size must be >= 1, got {self.snapkv_kernel_size}"
            )
        if self.snapkv_pooling not in {"avgpool", "maxpool"}:
            raise ValueError(
                f"snapkv_pooling must be 'avgpool' or 'maxpool', "
                f"got {self.snapkv_pooling!r}"
            )
        if self.h2o_window_size < 0:
            raise ValueError(
                f"h2o_window_size must be >= 0 (0 = use all query positions), "
                f"got {self.h2o_window_size}"
            )
        if self.manifoldkv_window_size < 0:
            raise ValueError(
                f"manifoldkv_window_size must be >= 0 (0 = global centroid), "
                f"got {self.manifoldkv_window_size}"
            )
        if self.compaction_ridge_lambda < 0.0:
            raise ValueError(
                f"compaction_ridge_lambda must be >= 0, got {self.compaction_ridge_lambda}"
            )
        if self.compaction_obs_window_size < 0:
            raise ValueError(
                f"compaction_obs_window_size must be >= 0 (0 = use all query positions), "
                f"got {self.compaction_obs_window_size}"
            )
        if self.compaction_beta_method not in {"zero", "nnls"}:
            raise ValueError(
                f"compaction_beta_method must be 'zero' or 'nnls', "
                f"got {self.compaction_beta_method!r}"
            )
        if self.compaction_nnls_iters < 0:
            raise ValueError(
                f"compaction_nnls_iters must be >= 0, got {self.compaction_nnls_iters}"
            )
        if self.compaction_nnls_lower_bound <= 0.0:
            raise ValueError(
                f"compaction_nnls_lower_bound must be > 0, "
                f"got {self.compaction_nnls_lower_bound}"
            )
        if self.compaction_nnls_upper_bound <= self.compaction_nnls_lower_bound:
            raise ValueError(
                f"compaction_nnls_upper_bound must be > compaction_nnls_lower_bound, "
                f"got upper={self.compaction_nnls_upper_bound}, "
                f"lower={self.compaction_nnls_lower_bound}"
            )
        if self.compaction_q_ref not in {"context_prefill", "on_policy"}:
            raise ValueError(
                f"compaction_q_ref must be 'context_prefill' or 'on_policy', "
                f"got {self.compaction_q_ref!r}"
            )
        if self.kvzip_chunk_size <= 0:
            raise ValueError(
                f"kvzip_chunk_size must be > 0, got {self.kvzip_chunk_size}"
            )

        # Validate strategies
        valid_strategies = {
            "probe",
            "probe_evict",
            "random",
            "random_evict",
            "threeway",
            "h2o",
            "snapkv",
            "keydiff",
            "manifoldkv",
            "compaction",
            "kvzip",
        }
        for s in self.strategies:
            if s not in valid_strategies:
                raise ValueError(f"strategy {s!r} not in {valid_strategies}")
        if "threeway" in self.strategies and self.granularity != "token":
            raise ValueError(
                "strategy 'threeway' requires granularity='token', "
                f"got granularity={self.granularity!r}"
            )
        if "threeway" in self.strategies and self.mode != "vk":
            raise ValueError(
                "strategy 'threeway' requires mode='vk' "
                "(k_cos_post_rope eviction axis requires K weights)"
            )
        if self.threeway_approx_ratios is not None:
            if len(self.threeway_approx_ratios) != len(self.ratios):
                raise ValueError(
                    f"threeway_approx_ratios length ({len(self.threeway_approx_ratios)}) "
                    f"must match ratios length ({len(self.ratios)})"
                )
            for approx_r, compress_r in zip(self.threeway_approx_ratios, self.ratios):
                if not (0.0 <= approx_r <= 1.0):
                    raise ValueError(
                        f"each threeway_approx_ratios value must be in [0, 1], got {approx_r}"
                    )
                if approx_r > 1.0 - compress_r + 1e-9:
                    raise ValueError(
                        f"threeway_approx_ratios value {approx_r} violates constraint: "
                        f"must be <= 1 - compress_ratio = {1.0 - compress_r:.3f} "
                        f"(compress_ratio={compress_r}). n_exact would be < 0."
                    )

        _VALID_THREEWAY_IMPORTANCE = {"k_cos_post_rope", "rank_sum_vnorm_kcos"}
        if self.threeway_importance not in _VALID_THREEWAY_IMPORTANCE:
            raise ValueError(
                f"threeway_importance must be one of {_VALID_THREEWAY_IMPORTANCE}, "
                f"got {self.threeway_importance!r}"
            )
        # Rank schema validation is only required when ranked probe strategies are used.
        uses_ranked_probe = any(s in {"probe", "probe_evict"} for s in self.strategies)
        if not uses_ranked_probe:
            return
        if self.rank_family is None:
            raise ValueError("rank_family must be set for probe/probe_evict strategies")
        if self.rank_family not in valid_rank_family:
            raise ValueError(
                f"rank_family must be one of {valid_rank_family}, got {self.rank_family!r}"
            )
        assert self.rank_signals is not None, "rank_signals must be set"
        if len(self.rank_signals) == 0:
            raise ValueError("rank_signals must be non-empty")
        if "k_cos" in self.rank_signals:
            raise ValueError(
                "k_cos is forbidden. Use explicit signal name: "
                "k_cos_pre_rope or k_cos_post_rope."
            )
        if not set(self.rank_signals).issubset(valid_rank_signals):
            bad = set(self.rank_signals) - valid_rank_signals
            raise ValueError(
                f"rank_signals contains unsupported values: {bad}; "
                f"supported={valid_rank_signals}"
            )

        if self.rank_family == "rank":
            if self.rank_weights is not None:
                raise ValueError("rank does not use rank_weights")
            if self.union_split is not None:
                raise ValueError("rank does not use union_split")
            if len(self.rank_signals) != 1:
                raise ValueError("rank requires exactly 1 rank_signal")
        elif self.rank_family == "weighted_rank":
            if self.rank_weights is None:
                raise ValueError("weighted_rank requires rank_weights")
            if len(self.rank_weights) != len(self.rank_signals):
                raise ValueError(
                    "weighted_rank requires len(rank_weights)==len(rank_signals), got "
                    f"{len(self.rank_weights)} vs {len(self.rank_signals)}"
                )
            if any(w < 0.0 for w in self.rank_weights):
                raise ValueError("rank_weights must all be >= 0")
        elif self.rank_family == "union":
            if self.rank_weights is not None:
                raise ValueError("union does not use rank_weights")
            if len(self.rank_signals) != 2:
                raise ValueError("union requires exactly 2 rank_signals")
            if self.union_split is None:
                raise ValueError("union requires union_split")
            if len(self.union_split) != 2:
                raise ValueError("union_split must contain exactly 2 values")
            if any(x <= 0.0 for x in self.union_split):
                raise ValueError("union_split values must be > 0")
        elif self.rank_family == "pareto":
            assert self.rank_family == "pareto"
            if self.rank_weights is not None:
                raise ValueError("pareto does not use rank_weights")
            if self.union_split is not None:
                raise ValueError("pareto does not use union_split")
            if len(self.rank_signals) != 2:
                raise ValueError("pareto currently requires exactly 2 rank_signals")
            if set(self.rank_signals) != {"v_err", "v_norm"}:
                raise ValueError(
                    "pareto currently only supports rank_signals={'v_err','v_norm'}"
                )
        else:
            assert self.rank_family == "product"
            if self.rank_weights is not None:
                raise ValueError("product does not use rank_weights")
            if self.union_split is not None:
                raise ValueError("product does not use union_split")
            if len(self.rank_signals) != 2:
                raise ValueError("product requires exactly 2 rank_signals")
            if set(self.rank_signals) != {"v_err", "v_norm"}:
                raise ValueError(
                    "product currently only supports rank_signals={'v_err','v_norm'}"
                )

        # Keep interface for per_head_dynamic in config validation;
        # implementation-level paths raise NotImplementedError.


@dataclass
class EvalConfig:
    """Evaluation pipeline settings."""

    benchmark: str  # "ppl" | "niah" | "longbench"
    output_path: Path = Path("results/eval.json")

    # PPL
    ppl_n_seqs: int = 50
    ppl_seq_len: int = 2048
    ppl_prefix_frac: float = 0.8

    # NIAH
    niah_depths: list[int] = field(default_factory=lambda: [10, 30, 50, 70, 90])
    niah_n_trials: int = 5
    niah_context_len: int = 2048
    niah_max_gen: int = 30
    niah_data_dir: Path = Path("data/NIAH")

    # LongBench
    lb_data_dir: Path = Path("data/LongBench/data")
    lb_tasks: list[str] = field(default_factory=list)
    lb_n_samples: int = 200
    lb_max_tokens: int | None = None  # None = no truncation
    strict_first_token: bool = False

    def __post_init__(self) -> None:
        valid_benchmarks = {"ppl", "niah", "longbench"}
        if self.benchmark not in valid_benchmarks:
            raise ValueError(
                f"benchmark must be one of {valid_benchmarks}, got {self.benchmark!r}"
            )
        self.output_path = Path(self.output_path)
        self.niah_data_dir = Path(self.niah_data_dir)
        self.lb_data_dir = Path(self.lb_data_dir)
