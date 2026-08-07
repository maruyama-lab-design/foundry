"""
rebuild_missing_entries.py
===========================
interpro_annotations.json に欠落している PDB 鎖エントリを補完する。

欠落した理由:
  複数プロセスによる同時書き込みでファイルが破損し、復旧時に 72,428 エントリを
  失った。このスクリプトは pdb_chain_uniprot.tsv と比較して欠落を特定し、
  InterPro API + UniProt API で再取得する。

使い方:
  python models/rfd3/scripts/rebuild_missing_entries.py \
      --annotation-path /home/om/OneDrive/data/rfd3/sifts/interpro_annotations.json \
      --sifts-tsv /home/om/OneDrive/data/rfd3/sifts/pdb_chain_uniprot.tsv \
      --checkpoint-every 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# InterPro API
_INTERPRO_URL = "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{uid}/?format=json&page_size=200"
# UniProt API
_UNIPROT_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"

_MAX_RETRIES = 5
_PAUSE = 0.3   # InterPro: 礼儀としてのウェイト


def fetch_interpro(uid: str) -> list[dict]:
    """UniProt ID に対応する InterPro エントリ一覧を取得する。"""
    url = _INTERPRO_URL.format(uid=uid)
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # 空レスポンス = InterPro に登録なし → リトライ不要
            if not resp.content or not resp.text.strip():
                return []
            data = resp.json()
            entries = []
            for r in data.get("results", []):
                meta = r.get("metadata", {})
                accession = meta.get("accession", "")
                name = meta.get("name") or ""
                if name == "None":
                    name = ""
                entries.append({
                    "id": accession,
                    "name": name,
                    "description": "",
                })
            return entries
        except requests.HTTPError as exc:
            if resp.status_code in (404, 204):
                return []  # タンパク質が InterPro に存在しない
            if attempt == _MAX_RETRIES - 1:
                log.warning(f"InterPro API 失敗 ({uid}): {exc}")
                return []
            wait = _PAUSE * (2 ** attempt)
            log.warning(f"InterPro API エラー (attempt {attempt+1}, {uid}): {exc}. {wait:.1f}s 後リトライ")
            time.sleep(wait)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                log.warning(f"InterPro API 失敗 ({uid}): {exc}")
                return []
            wait = _PAUSE * (2 ** attempt)
            log.warning(f"InterPro API エラー (attempt {attempt+1}, {uid}): {exc}. {wait:.1f}s 後リトライ")
            time.sleep(wait)
    return []


def fetch_uniprot_functions(uid_batch: list[str]) -> dict[str, str]:
    """UniProt /accessions エンドポイントでバッチ取得。"""
    params = {
        "accessions": ",".join(uid_batch),
        "fields": "accession,cc_function",
        "format": "json",
        "size": str(len(uid_batch)),
    }
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(_UNIPROT_ACCESSIONS, params=params, timeout=60)
            resp.raise_for_status()
            response_json = resp.json()
            result: dict[str, str] = {}
            for entry in response_json.get("results", []):
                uid = entry.get("primaryAccession", "")
                if not uid:
                    continue
                function_text = ""
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            function_text = texts[0].get("value", "")
                            break
                result[uid] = function_text
            return result
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                log.warning(f"UniProt API 失敗: {exc}")
                return {}
            wait = _PAUSE * (2 ** attempt)
            log.warning(f"UniProt API エラー (attempt {attempt+1}): {exc}. {wait:.1f}s 後リトライ")
            time.sleep(wait)
    return {}


def main():
    parser = argparse.ArgumentParser(description="欠落 PDB 鎖エントリを補完する")
    parser.add_argument("--annotation-path", required=True, type=Path)
    parser.add_argument("--sifts-tsv", required=True, type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="この件数（UniProt ID）ごとに途中保存 (default: 500)")
    parser.add_argument("--interpro-pause", type=float, default=_PAUSE)
    args = parser.parse_args()

    # ── 既存 DB 読み込み ──────────────────────────────────────────
    log.info(f"annotation DB を読み込み中: {args.annotation_path}")
    with open(args.annotation_path) as f:
        db: dict[str, dict] = json.load(f)
    log.info(f"  {len(db):,} エントリ読み込み完了")

    # ── SIFTS TSV 読み込み ────────────────────────────────────────
    log.info(f"SIFTS TSV を読み込み中: {args.sifts_tsv}")
    tsv = pd.read_csv(args.sifts_tsv, sep="\t", comment="#", header=0, low_memory=False)
    tsv = tsv.dropna(subset=["PDB", "CHAIN", "SP_PRIMARY"])
    tsv["key"] = tsv["PDB"].str.lower() + "_" + tsv["CHAIN"].str.lower()
    tsv = tsv.drop_duplicates(subset=["key"])
    log.info(f"  {len(tsv):,} ユニーク PDB 鎖")

    # ── 欠落エントリ特定 ──────────────────────────────────────────
    current_keys = set(db.keys())
    missing_rows = tsv[~tsv["key"].isin(current_keys)]
    log.info(f"欠落 PDB 鎖: {len(missing_rows):,}")

    if missing_rows.empty:
        log.info("欠落エントリなし。終了します。")
        return

    # key → uniprot_id のマッピング
    key_to_uid: dict[str, str] = dict(zip(missing_rows["key"], missing_rows["SP_PRIMARY"]))

    # UniProt ID → 対応する欠落 key のリスト
    uid_to_keys: dict[str, list[str]] = {}
    for key, uid in key_to_uid.items():
        uid_to_keys.setdefault(uid, []).append(key)

    all_uids = sorted(uid_to_keys.keys())
    log.info(f"ユニーク UniProt ID (欠落分): {len(all_uids):,}")

    # 処理済みチェック（再開対応）: DB にエントリがあれば済み
    already_done = {uid for uid in all_uids if all(k in db for k in uid_to_keys[uid])}
    remaining_uids = [uid for uid in all_uids if uid not in already_done]
    log.info(f"処理済み: {len(already_done):,} / 残り: {len(remaining_uids):,}")

    if not remaining_uids:
        log.info("全 UniProt ID 処理済みです。")
        return

    # ── UniProt 機能記述をバッチ取得 ─────────────────────────────
    log.info("UniProt 機能記述をバッチ取得中...")
    uniprot_func_map: dict[str, str] = {}
    batch_size_uniprot = 100
    for start in range(0, len(remaining_uids), batch_size_uniprot):
        batch = remaining_uids[start : start + batch_size_uniprot]
        funcs = fetch_uniprot_functions(batch)
        uniprot_func_map.update(funcs)
        time.sleep(0.2)
    log.info(f"  UniProt 機能記述取得済み: {sum(1 for v in uniprot_func_map.values() if v):,} / {len(remaining_uids):,}")

    # ── InterPro データを 1 UniProt ID ずつ取得 ───────────────────
    added_entries = 0
    failed_uids = 0

    for i, uid in enumerate(remaining_uids):
        keys_for_uid = uid_to_keys[uid]

        # InterPro データ取得
        interpro_entries = fetch_interpro(uid)
        time.sleep(args.interpro_pause)

        # 機能記述
        func_text = uniprot_func_map.get(uid, "")

        # DB に追加
        new_entry = {
            "uniprot_id": uid,
            "interpro": interpro_entries,
        }
        if func_text:
            new_entry["uniprot_function"] = func_text

        for key in keys_for_uid:
            db[key] = new_entry

        added_entries += len(keys_for_uid)
        if not interpro_entries:
            failed_uids += 1

        # 進捗表示
        if (i + 1) % 100 == 0 or (i + 1) == len(remaining_uids):
            pct = (i + 1) / len(remaining_uids) * 100
            log.info(
                f"進捗: {i+1:,}/{len(remaining_uids):,} UniProt ID ({pct:.1f}%) "
                f"| 追加エントリ: {added_entries:,} | InterPro 取得失敗: {failed_uids:,}"
            )

        # チェックポイント保存
        processed = i + 1
        if processed % args.checkpoint_every == 0 or processed == len(remaining_uids):
            with open(args.annotation_path, "w") as f:
                json.dump(db, f)
            log.info(f"  チェックポイント保存: {args.annotation_path} ({len(db):,} エントリ)")

    log.info(
        f"完了: 追加エントリ={added_entries:,} / InterPro 取得失敗={failed_uids:,}"
    )
    log.info(f"最終 DB エントリ数: {len(db):,}")
    log.info("")
    log.info("次のステップ: embeddings_cache.pt を再計算してください")
    log.info("  python -u -c \"")
    log.info("  import sys; sys.path.insert(0,'models/rfd3/src')")
    log.info("  from rfd3.transforms.function_text_transforms import precompute_embeddings")
    log.info(f"  precompute_embeddings(")
    log.info(f"      annotation_path='{args.annotation_path}',")
    log.info(f"      output_path='{args.annotation_path.parent / 'embeddings_cache.pt'}',")
    log.info(f"      batch_size=512, device='cuda',")
    log.info(f"  )\"")


if __name__ == "__main__":
    main()
