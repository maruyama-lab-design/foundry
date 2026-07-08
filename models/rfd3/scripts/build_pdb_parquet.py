"""
build_pdb_parquet.py
=====================
全 PDB CIF ファイルから pn_units_df_train.parquet を生成するスクリプト。

手順:
  1. 全 CIF ファイルを並列処理 → 生データ parquet（途中保存あり）
  2. MMseqs2 で配列クラスタリング → cluster 列を追加
  3. 最終 parquet を出力

使い方:
  # STEP 1: PDB ダウンロード（事前に実行）
  conda run -n rfd3m atomworks pdb sync /home/om/pdb_full

  # STEP 2: parquet ビルド
  conda activate rfd3m
  python models/rfd3/scripts/build_pdb_parquet.py \\
      --pdb-dir /home/om/pdb_full \\
      --output-dir /home/om/pdb_parquet \\
      --workers 28

  # STEP 3 (optional): スパコンに転送
  rsync -avz /home/om/pdb_parquet/ kyushu:/path/to/parquet/

要件:
  - atomworks >= 2.2.1  (conda activate rfd3m)
  - mmseqs2              (conda install -c bioconda -n rfd3m mmseqs2)
  - pandas, pyarrow
"""

from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChainType 定数（atomworks.enums.ChainType より）
# ---------------------------------------------------------------------------
POLYPEPTIDE_TYPES = {5, 6}   # POLYPEPTIDE_D=5, POLYPEPTIDE_L=6
NUCLEIC_TYPES = {3, 4, 7}    # DNA=3, DNA_RNA_HYBRID=4, RNA=7
PEPTIDE_TYPES = {0, 2}       # CYCLIC_PSEUDO_PEPTIDE=0, PEPTIDE_NUCLEIC_ACID=2
NON_POLYMER_TYPE = 8


# ---------------------------------------------------------------------------
# Worker: 1 CIF ファイル → rows のリスト
# ---------------------------------------------------------------------------
def _normalize_value(v):
    """list / set / dict → JSON 文字列に統一して PyArrow の型混在エラーを防ぐ。"""
    import json
    if isinstance(v, (set, frozenset)):
        return json.dumps(sorted(str(x) for x in v))
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str)
    return v


_WORKER_TIMEOUT = 600  # 秒（1構造あたり最大10分）


def _process_one_cif(path: str) -> tuple[str, list[dict]]:
    """単一 CIF ファイルを処理して (path, rows) を返す。SIGALRM でタイムアウト。"""
    import signal

    def _handle_alarm(signum, frame):
        raise TimeoutError(f"timeout after {_WORKER_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _handle_alarm)
    signal.alarm(_WORKER_TIMEOUT)
    try:
        from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
        dp = DataPreprocessor()
        rows = dp.get_rows(path, ligand_scores=[])
        for r in rows:
            for k, v in list(r.items()):
                r[k] = _normalize_value(v)
            r["path"] = str(r.get("path", path))
        signal.alarm(0)
        return path, rows
    except TimeoutError as e:
        signal.alarm(0)
        return path, [{"_error": f"timeout: {e}", "_path": str(path)}]
    except Exception as e:
        signal.alarm(0)
        return path, [{"_error": str(e), "_path": str(path)}]


# ---------------------------------------------------------------------------
# STEP 1: CIF → 生 DataFrame
# ---------------------------------------------------------------------------
def build_raw_df(pdb_dir: Path, output_dir: Path, workers: int, checkpoint_every: int = 5000) -> Path:
    raw_parquet = output_dir / "pn_units_raw.parquet"
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 既存チェックポイントから done_paths だけ収集（行データはメモリに載せない）
    done_paths: set[str] = set()
    for cp in sorted(checkpoint_dir.glob("chunk_*.parquet")):
        df_cp = pd.read_parquet(cp, columns=["_source_path"])
        if "_source_path" in df_cp.columns:
            done_paths.update(df_cp["_source_path"].tolist())
    log.info(f"チェックポイント済み: {len(done_paths)} ファイル ({len(list(checkpoint_dir.glob('chunk_*.parquet')))} chunks)")

    # 未処理ファイルを列挙（-sf.cif.gz は構造因子ファイルなので除外）
    cif_files = sorted(
        f for f in glob.glob(str(pdb_dir / "**" / "*.cif.gz"), recursive=True)
        if "-sf.cif.gz" not in f
    )
    remaining = [f for f in cif_files if f not in done_paths]
    log.info(f"CIF ファイル総数: {len(cif_files)}, 未処理: {len(remaining)}")

    chunk_rows: list[dict] = []
    chunk_idx = len(list(checkpoint_dir.glob("chunk_*.parquet")))
    all_rows = []  # 未使用（後でチェックポイントから PyArrow で結合）
    t_start = time.time()
    n_done = len(done_paths)
    n_total = len(cif_files)

    # maxtasksperchild=50 でワーカーを定期再起動しメモリ蓄積を防ぐ
    # imap_unordered: 完了した順に結果を受け取る（先頭ファイルが遅くてもブロックしない）
    with multiprocessing.Pool(processes=workers, maxtasksperchild=50) as pool:
        for i, (path, rows) in enumerate(
            pool.imap_unordered(_process_one_cif, remaining, chunksize=2), 1
        ):
            for r in rows:
                r["_source_path"] = path
            chunk_rows.extend(rows)
            n_done += 1

            if n_done % 50 == 0:  # 50件ごとに進捗表示
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 1
                remaining_sec = (len(remaining) - i) / rate
                log.info(
                    f"進捗: {n_done}/{n_total} "
                    f"({n_done/n_total*100:.1f}%) "
                    f"速度 {rate:.1f}件/s  残り推定 {remaining_sec/3600:.1f}h"
                )

            # チェックポイント保存
            if len(chunk_rows) >= checkpoint_every:
                cp_path = checkpoint_dir / f"chunk_{chunk_idx:04d}.parquet"
                pd.DataFrame(chunk_rows).to_parquet(cp_path, index=False)
                log.info(f"チェックポイント保存: {cp_path} ({len(chunk_rows)} rows)")
                chunk_rows = []
                chunk_idx += 1

    # 残りを保存
    if chunk_rows:
        cp_path = checkpoint_dir / f"chunk_{chunk_idx:04d}.parquet"
        pd.DataFrame(chunk_rows).to_parquet(cp_path, index=False)

    # チェックポイントを PyArrow で結合（Pythonリストに展開しない → メモリ節約）
    log.info("チェックポイントを結合して raw parquet を作成中...")
    import pyarrow.parquet as pq
    import pyarrow as pa
    cp_files = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    tables = [pq.read_table(str(cp)) for cp in cp_files]
    combined = pa.concat_tables(tables, promote_options="default")
    # エラー行を除外
    if "_error" in combined.schema.names:
        import pyarrow.compute as pc
        mask = pc.is_null(combined.column("_error"))
        n_errors = len(combined) - pc.sum(mask).as_py()
        if n_errors > 0:
            log.warning(f"エラー行: {n_errors} 件 (除外します)")
        combined = combined.filter(mask)
    pq.write_table(combined, str(raw_parquet))
    log.info(f"生 parquet 保存: {raw_parquet} ({len(combined):,} rows)")
    return raw_parquet


# ---------------------------------------------------------------------------
# STEP 2: 派生カラムを追加
# ---------------------------------------------------------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # example_id: generate_example_id(['pdb','pn_unit'], pdb_id, assembly_id, [q_pn_unit_iid])
    # -> "{['pdb', 'pn_unit']}{1abc}{1}{['A_1']}"
    df["example_id"] = (
        "{['pdb', 'pn_unit']}"
        + "{" + df["pdb_id"].astype(str) + "}"
        + "{" + df["assembly_id"].astype(str) + "}"
        + "{['" + df["q_pn_unit_iid"].astype(str) + "']}"
    )

    # n_prot / n_nuc / n_ligand / n_peptide を assembly ごとに集計
    # all_pn_unit_iids_after_processing はアセンブリ内の全 pn_unit iid リスト（JSON）
    # ここでは q_pn_unit_type を使って近似（正確には全 pn_unit の型情報が必要）
    # → 同一 (pdb_id, assembly_id) のグループで q_pn_unit_type をカウント
    type_cols = df.groupby(["pdb_id", "assembly_id"])["q_pn_unit_type"].apply(list).reset_index()
    type_cols.columns = ["pdb_id", "assembly_id", "_types"]

    def _count_types(types):
        return {
            "n_prot": sum(1 for t in types if t in POLYPEPTIDE_TYPES),
            "n_nuc": sum(1 for t in types if t in NUCLEIC_TYPES),
            "n_ligand": sum(1 for t in types if t == NON_POLYMER_TYPE),
            "n_peptide": sum(1 for t in types if t in PEPTIDE_TYPES),
        }

    type_cols[["n_prot", "n_nuc", "n_ligand", "n_peptide"]] = pd.DataFrame(
        type_cols["_types"].apply(_count_types).tolist()
    )
    type_cols.drop(columns="_types", inplace=True)

    df = df.merge(type_cols, on=["pdb_id", "assembly_id"], how="left")
    return df


# ---------------------------------------------------------------------------
# STEP 3: MMseqs2 クラスタリング
# ---------------------------------------------------------------------------
def add_cluster_column(df: pd.DataFrame, tmp_dir: Path) -> pd.DataFrame:
    """MMseqs2 easy-cluster でタンパク質配列をクラスタリングし cluster 列を追加する。"""
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        log.warning(
            "mmseqs2 が見つかりません。cluster 列を None に設定します。\n"
            "インストール: conda install -c bioconda mmseqs2"
        )
        df["cluster"] = None
        return df

    fasta_path = tmp_dir / "sequences.fasta"
    result_prefix = tmp_dir / "mmseqs_result"
    cluster_tsv = tmp_dir / "mmseqs_result_cluster.tsv"

    # 配列ハッシュ → 代表配列 のマッピングを作る
    seq_col = "q_pn_unit_processed_entity_canonical_sequence"
    hash_col = "q_pn_unit_processed_entity_canonical_sequence_hash"

    if seq_col not in df.columns:
        log.warning(f"{seq_col} 列がありません。cluster 列を None に設定します。")
        df["cluster"] = None
        return df

    # ポリマー行のみ（配列あり）
    seq_df = df[[hash_col, seq_col]].dropna(subset=[seq_col])
    seq_df = seq_df[seq_df[seq_col].astype(str).str.len() > 0]
    seq_df = seq_df.drop_duplicates(subset=[hash_col])

    log.info(f"クラスタリング対象配列: {len(seq_df):,} 件")

    # FASTA 書き出し
    with open(fasta_path, "w") as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row[hash_col]}\n{row[seq_col]}\n")

    # MMseqs2 easy-cluster (40% identity, 80% coverage)
    cmd = [
        mmseqs, "easy-cluster",
        str(fasta_path),
        str(result_prefix),
        str(tmp_dir / "tmp"),
        "--min-seq-id", "0.4",
        "-c", "0.8",
        "--cov-mode", "0",
        "--cluster-mode", "0",
        "--threads", str(min(28, os.cpu_count() or 8)),
    ]
    log.info(f"MMseqs2 実行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # クラスタ結果読み込み (TSV: rep_seq \t member_seq)
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    hash_to_cluster = dict(zip(cluster_df["member"], cluster_df["rep"]))

    df["cluster"] = df[hash_col].map(hash_to_cluster)
    n_clustered = df["cluster"].notna().sum()
    log.info(f"クラスタ割り当て: {n_clustered:,} / {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# STEP 4: 最終 parquet を保存
# ---------------------------------------------------------------------------
def save_final_parquet(df: pd.DataFrame, output_dir: Path, split_date: str = "2024-12-16") -> None:
    # 学習/検証 split（日付ベース）
    df["deposition_date"] = pd.to_datetime(df["deposition_date"], errors="coerce")
    df_train = df[df["deposition_date"] < split_date].copy()
    df_val = df[df["deposition_date"] >= split_date].copy()

    train_path = output_dir / "pn_units_df_train.parquet"
    val_path = output_dir / "pn_units_df_val.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    log.info(f"学習 parquet: {train_path} ({len(df_train):,} rows)")
    log.info(f"検証 parquet: {val_path} ({len(df_val):,} rows)")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PDB CIF → pn_units_df_train.parquet ビルダー")
    parser.add_argument("--pdb-dir", required=True, type=Path, help="PDB CIF ミラーディレクトリ")
    parser.add_argument("--output-dir", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("--workers", type=int, default=28, help="並列ワーカー数")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="チェックポイント保存間隔（行数）")
    parser.add_argument("--skip-raw", action="store_true", help="生parquetがあればスキップ")
    parser.add_argument("--split-date", default="2024-12-16", help="学習/検証分割日")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_parquet = args.output_dir / "pn_units_raw.parquet"

    # STEP 1: CIF 処理
    if args.skip_raw and raw_parquet.exists():
        log.info(f"生 parquet をスキップ: {raw_parquet}")
    else:
        raw_parquet = build_raw_df(
            args.pdb_dir, args.output_dir, args.workers, args.checkpoint_every
        )

    # STEP 2: 派生カラム
    log.info("派生カラムを計算中...")
    df = pd.read_parquet(raw_parquet)
    df = add_derived_columns(df)

    # STEP 3: クラスタリング
    with tempfile.TemporaryDirectory() as tmp:
        log.info("MMseqs2 クラスタリング中...")
        df = add_cluster_column(df, Path(tmp))

    # STEP 4: 保存
    save_final_parquet(df, args.output_dir, args.split_date)
    log.info("完了！")


if __name__ == "__main__":
    main()
