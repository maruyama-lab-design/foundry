"""
add_uniprot_descriptions.py
============================
既存の interpro_annotations.json に UniProt 機能記述を追加する。

UniProt REST API（バッチ）で各タンパク質の機能説明を取得し、
各エントリに "uniprot_function" フィールドとして追加する。

実行後の JSON フォーマット:
  {
    "4hhb_a": {
      "uniprot_id": "P69905",
      "uniprot_function": "Involved in oxygen transport from the lung ...",
      "interpro": [...]
    }
  }

使い方:
  conda activate rfd3m
  python models/rfd3/scripts/add_uniprot_descriptions.py \\
      --annotation-path /home/om/OneDrive/data/rfd3/sifts/interpro_annotations.json

完了後、embeddings_cache.pt を再計算すること:
  python -c "
  from rfd3.transforms.function_text_transforms import precompute_embeddings
  precompute_embeddings(
      annotation_path='/home/om/OneDrive/data/rfd3/sifts/interpro_annotations.json',
      output_path='/home/om/OneDrive/data/rfd3/sifts/embeddings_cache.pt',
      batch_size=256, device='cpu',
  )
  "
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

# UniProt REST API
_UNIPROT_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"
_BATCH_SIZE = 100       # 1リクエストあたりの UniProt ID 数
_REQUEST_PAUSE = 0.5    # リクエスト間の待機秒数（レート制限対策）
_MAX_RETRIES = 5


def fetch_uniprot_functions(uid_batch: list[str]) -> dict[str, str]:
    """UniProt /accessions エンドポイントでバッチ取得。{uid: function_text} を返す。"""
    params = {
        "accessions": ",".join(uid_batch),
        "fields": "accession,cc_function",
        "format": "json",
        "size": str(len(uid_batch)),
    }
    response_json: dict = {}

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
        # cc_function コメントを探す
        function_text = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                    break
        result[uid] = function_text

    return result


def main():
    parser = argparse.ArgumentParser(
        description="interpro_annotations.json に UniProt 機能記述を追加する"
    )
    parser.add_argument(
        "--annotation-path",
        required=True,
        type=Path,
        help="interpro_annotations.json のパス（上書き更新される）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_BATCH_SIZE,
        help=f"1リクエストあたりの UniProt ID 数（デフォルト: {_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=_REQUEST_PAUSE,
        help=f"リクエスト間の待機秒数（デフォルト: {_REQUEST_PAUSE}）",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="この件数ごとに途中保存する（デフォルト: 5000）",
    )
    args = parser.parse_args()

    # --- JSON 読み込み ---
    log.info(f"annotation DB を読み込み中: {args.annotation_path}")
    with open(args.annotation_path) as f:
        db: dict[str, dict] = json.load(f)
    log.info(f"  {len(db):,} エントリ読み込み完了")

    # --- ユニーク UniProt ID を収集 ---
    uid_to_keys: dict[str, list[str]] = {}
    for key, entry in db.items():
        uid = entry.get("uniprot_id", "")
        if uid:
            uid_to_keys.setdefault(uid, []).append(key)

    all_uids = sorted(uid_to_keys.keys())
    log.info(f"ユニーク UniProt ID 数: {len(all_uids):,}")

    # --- 未処理 ID を特定（再開対応）---
    already_done = {
        uid for uid in all_uids
        if all(
            "uniprot_function" in db.get(k, {})
            for k in uid_to_keys[uid]
        )
    }
    remaining = [uid for uid in all_uids if uid not in already_done]
    log.info(f"処理済み: {len(already_done):,} / 残り: {len(remaining):,}")

    if not remaining:
        log.info("全 UniProt ID 処理済みです。")
        return

    # --- バッチ処理 ---
    updated = 0
    empty_count = 0
    n_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for batch_num, start in enumerate(range(0, len(remaining), args.batch_size)):
        batch = remaining[start : start + args.batch_size]
        functions = fetch_uniprot_functions(batch)

        for uid in batch:
            func_text = functions.get(uid, "")
            for key in uid_to_keys[uid]:
                db[key]["uniprot_function"] = func_text
            if func_text:
                updated += 1
            else:
                empty_count += 1

        processed = start + len(batch)

        # 進捗表示
        if (batch_num + 1) % 10 == 0 or processed >= len(remaining):
            pct = processed / len(remaining) * 100
            log.info(
                f"進捗: {processed:,}/{len(remaining):,} UniProt ID ({pct:.1f}%) "
                f"| 取得成功: {updated:,} | 空: {empty_count:,}"
            )

        # 途中保存
        if processed % args.checkpoint_every == 0 or processed >= len(remaining):
            with open(args.annotation_path, "w") as f:
                json.dump(db, f)
            log.info(f"  チェックポイント保存: {args.annotation_path}")

        time.sleep(args.pause)

    log.info(
        f"完了: UniProt 機能記述 取得={updated:,} / 空={empty_count:,} "
        f"→ {args.annotation_path}"
    )
    log.info("")
    log.info("次のステップ: embeddings_cache.pt を再計算してください")
    log.info("  python -c \"")
    log.info("  from rfd3.transforms.function_text_transforms import precompute_embeddings")
    log.info(f"  precompute_embeddings(")
    log.info(f"      annotation_path='{args.annotation_path}',")
    log.info(f"      output_path='{args.annotation_path.parent / 'embeddings_cache.pt'}',")
    log.info(f"      batch_size=256, device='cpu',")
    log.info(f"  )\"")


if __name__ == "__main__":
    main()
