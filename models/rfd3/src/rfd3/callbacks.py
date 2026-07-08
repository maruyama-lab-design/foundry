import logging

import pandas as pd
from beartype.typing import Any

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_df_as_table

ranked_logger = RankedLogger(__name__, rank_zero_only=True)
log = logging.getLogger(__name__)


class FreezeBoneAndTrainFunctionTextCallback(BaseCallback):
    """
    Function-text conditioning のファインチューニング用 callback。

    on_fit_start で呼ばれ、以下のパラメータ以外をすべて凍結する:
      - text_proj / text_gate  (TokenInitializer のテキスト投影)
      - lora_A / lora_B        (LocalAttentionPairBias の LoRA アダプタ)
    """

    def on_fit_start(self, trainer: Any):
        model = trainer.state["model"]
        trainable, frozen = 0, 0
        for name, param in model.named_parameters():
            if any(key in name for key in ("text_proj", "text_gate", "lora_A", "lora_B")):
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()

        total = trainable + frozen
        log.info(
            f"[FreezeBoneAndTrainFunctionText] "
            f"学習対象: {trainable:,} params ({trainable / total * 100:.2f}%) / "
            f"凍結: {frozen:,} params"
        )


class LogFunctionTextConditioningMetricsCallback(BaseCallback):
    """
    Function-text conditioning 固有のメトリクスを wandb にログする callback。

    毎 epoch ログする項目:
      - finetune/text_gate        : テキスト条件の有効度 (tanh(gate) ∈ [-1, 1])
      - finetune/lora_norm_mean   : 全 LoRA (B@A) の Frobenius ノルム平均
      - finetune/trainable_params : 学習対象パラメータ数 (初回のみ)
    """

    def on_train_epoch_end(self, trainer: Any):
        if not trainer.fabric.is_global_zero:
            return

        model = trainer.state["model"]
        metrics: dict[str, float] = {}

        # text_gate: tanh を通した有効値を記録
        for name, param in model.named_parameters():
            if "text_gate" in name:
                metrics["finetune/text_gate"] = param.tanh().item()
                break

        # LoRA ノルム: B @ A の Frobenius ノルムの平均
        lora_norms = []
        lora_layers: dict[str, Any] = {}
        for name, param in model.named_parameters():
            layer_key = name.rsplit(".lora_", 1)[0]
            if ".lora_A" in name:
                lora_layers.setdefault(layer_key, {})["A"] = param
            elif ".lora_B" in name:
                lora_layers.setdefault(layer_key, {})["B"] = param

        for layer_key, ab in lora_layers.items():
            if "A" in ab and "B" in ab:
                composed = ab["B"] @ ab["A"]  # (out, in)
                lora_norms.append(composed.norm(p="fro").item())

        if lora_norms:
            import statistics
            metrics["finetune/lora_norm_mean"] = statistics.mean(lora_norms)
            metrics["finetune/lora_norm_max"] = max(lora_norms)

        trainer.fabric.log_dict(metrics, step=trainer.state["current_epoch"])


class LogDesignValidationMetricsCallback(BaseCallback):
    def on_validation_epoch_end(self, trainer: Any):
        # Only log metrics to disk if this is the global zero rank
        if not trainer.fabric.is_global_zero:
            return

        assert hasattr(
            trainer, "validation_results_path"
        ), "Results path not found! Ensure that StoreValidationMetricsInDFCallback is called first."
        df = pd.read_csv(trainer.validation_results_path)

        # ... filter to most recent epoch, drop epoch column
        df = df[df["epoch"] == df["epoch"].max()]
        df.drop(columns=["epoch"], inplace=True)

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset].copy()
            dataset_df.drop(columns=["dataset"], inplace=True)

            print(f"\n+{' ' + dataset + ' ':-^150}+\n")

            remaining_cols = [
                col for col in dataset_df.columns if col not in ["example_id"]
            ]
            remaining_df = dataset_df[remaining_cols].copy()
            remaining_df = remaining_df.dropna(how="all")
            numeric_cols = remaining_df.select_dtypes(include="number").columns

            # Compute means and non-NaN counts for numeric columns
            final_means = remaining_df[numeric_cols].mean()
            non_nan_counts = remaining_df[numeric_cols].count()

            # Convert the Series to a DataFrame and add the count as a new column
            final_means_df = final_means.to_frame(name="mean")
            final_means_df["Count"] = non_nan_counts

            print_df_as_table(
                final_means_df.reset_index(),
                f"{dataset} — {trainer.state['current_epoch']} — Design Validation Metrics",
            )
            if trainer.fabric:
                trainer.fabric.log_dict(
                    {f"val/{dataset}/{col}": final_means[col] for col in numeric_cols},
                    step=trainer.state["current_epoch"],
                )

                if len(dataset_df["example_id"].unique()) <= 25:
                    for eid, df_ in dataset_df.groupby("example_id"):
                        df_ = df_[numeric_cols].mean()
                        trainer.fabric.log_dict(
                            {
                                f"val/{dataset}/{col}/{eid}": df_[col]
                                for col in numeric_cols
                            },
                            step=trainer.state["current_epoch"],
                        )
