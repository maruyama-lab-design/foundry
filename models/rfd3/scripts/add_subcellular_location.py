"""
add_subcellular_location.py
============================
interpro_annotations.json の各エントリに UniProt の
Subcellular Location 情報を追加する。

実行後の JSON フォーマット:
  {
    "4hhb_a": {
      "uniprot_id": "P69905",
      "uniprot_function": "...",
      "uniprot_subcellular_location": "Cell inner membrane; Multi-pass membrane protein",
      "interpro": [...]
    }
  }

使い方:
  /home/om/miniconda3/envs/rfd3m/bin/python -u \
    models/rfd3/scripts/add_subcellular_location.py \
    --annotation-path /home/om/OneDrive/data/rfd3/sifts/interpro_annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_UNIPROT_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"
_BATCH_SIZE = 100
_REQUEST_PAUSE = 0.5
_MAX_RETRIES = 5


def _parse_subcellular_location(comments: list) -> str:
    """SUBCELLULAR LOCATION コメントから証拠コードを除いたクリーンなテキストを返す。"""
    locs = []
    for comment in comments:
        if comment.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        for sl in comment.get("subcellularLocations", []):
            parts = []
            loc_val = (sl.get("location") or {}).get("value", "")
            if loc_val:
                parts.append(loc_val)
            topo_val = (sl.get("topology") or {}).get("value", "")
            if topo_val:
                parts.append(topo_val)
            if parts:
                locs.append("; ".join(parts))
    return ". ".join(locs)


def fetch_subcellular_locations(uid_batch: list[str]) -> dict[str, str]:
    """UniProt /accessions エンドポイントで subcellular location をバッチ取得。"""
    params = {
        "accessions": ",".join(uid_batch),
        "fields": "accession,cc_subcellular_location",
        "format": "json",
        "size": str(len(uid_batch)),
    }
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(_UNIPROT_ACCESSIONS, params=params, timeout=60)
            resp.raise_for_status()
            response_json = resp.json()
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                log.warning(f"UniProt API 失敗（{_MAX_RETRIES}回リトライ後）: {exc}")
                return {}
            wait = _REQUEST_PAUSE * (2 ** attempt)
            log.warning(f"UniProt API エラー (attempt {attempt+1}): {exc}. {wait:.1f}s 後リトライ")
            time.sleep(wait)

    result: dict[str, str] = {}
    for entry in response_json.get("results", []):
        uid = entry.get("primaryAccession", "")
        if not uid:
            continue
        result[uid] = _parse_subcellular_location(entry.get("comments", []))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="interpro_annotations.json に UniProt Subcellular Location を追加する"
    )
    parser.add_argument("--annotation-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--pause", type=float, default=_REQUEST_PAUSE)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    args = parser.parse_args()

    log.info(f"annotation DB を読み込み中: {args.annotation_path}")
    with open(args.annotation_path) as f:
        db: dict[str, dict] = json.load(f)
    log.info(f"  {len(db):,} エントリ読み込み完了")

    # ユニーク UniProt ID を収集
    uid_to_keys: dict[str, list[str]] = {}
    for key, entry in db.items():
        uid = entry.get("uniprot_id", "")
        if uid:
            uid_to_keys.setdefault(uid, []).append(key)

    all_uids = sorted(uid_to_keys.keys())
    log.info(f"ユニーク UniProt ID 数: {len(all_uids):,}")

    # 再開対応: uniprot_subcellular_location キーが存在するものはスキップ
    already_done = {
        uid for uid in all_uids
        if all("uniprot_subcellular_location" in db.get(k, {}) for k in uid_to_keys[uid])
    }
    remaining = [uid for uid in all_uids if uid not in already_done]
    log.info(f"処理済み: {len(already_done):,} / 残り: {len(remaining):,}")

    if not remaining:
        log.info("全 UniProt ID 処理済みです。")
        return

    updated = 0
    empty_count = 0

    for batch_num, start in enumerate(range(0, len(remaining), args.batch_size)):
        batch = remaining[start: start + args.batch_size]
        locations = fetch_subcellular_locations(batch)

        for uid in batch:
            loc_text = locations.get(uid, "")
            for key in uid_to_keys[uid]:
                db[key]["uniprot_subcellular_location"] = loc_text
            if loc_text:
                updated += 1
            else:
                empty_count += 1

        processed = start + len(batch)

        if (batch_num + 1) % 10 == 0 or processed >= len(remaining):
            pct = processed / len(remaining) * 100
            log.info(
                f"進捗: {processed:,}/{len(remaining):,} ({pct:.1f}%) "
                f"| 取得成功: {updated:,} | 空: {empty_count:,}"
            )

        if processed % args.checkpoint_every == 0 or processed >= len(remaining):
            with open(args.annotation_path, "w") as f:
                json.dump(db, f)
            log.info(f"  チェックポイント保存: {args.annotation_path} ({len(db):,} エントリ)")

        time.sleep(args.pause)

    log.info(f"完了: 取得成功={updated:,} / 空={empty_count:,}")


if __name__ == "__main__":
    main()
