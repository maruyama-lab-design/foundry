# newly created by Maruyama on 2026-06-09.


"""
function_text_transforms.py
============================
Transform for adding function text embeddings derived from
InterPro domain annotations to RFD3 training data.

Pipeline position (pipelines.py):
    After  : AddGlobalIsNonLoopyFeature
    Before : EncodeAF3TokenLevelFeatures

Data flow:
    atom_array metadata
        → lookup key (e.g. "4hhb_a")
        → annotation JSON  →  domain description text
        → frozen PubMedBERT  →  [CLS] embedding  (768-d)
        → data["feats"]["function_text_emb"]

Prerequisites (run once before training):
    1. build_annotation_db.py   : PDB → UniProt → InterPro の対応表を作成
    2. precompute_embeddings.py : 全エントリの埋め込みを事前計算してキャッシュ

Usage in pipelines.py:
    from rfd3.transforms.function_text_transforms import AddFunctionTextEmbedding

    # AddGlobalIsNonLoopyFeature の直後に追加
    TrainingRoute(
        AddFunctionTextEmbedding(
            annotation_path="/path/to/interpro_annotations.json",
            embedding_cache_path="/path/to/embeddings_cache.pt",  # 推奨
            dropout_prob=0.1,
        )
    ),
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# CFG null condition として使うゼロ埋め込みのデフォルト次元
DEFAULT_EMBED_DIM = 768


# =============================================================================
# メイン Transform クラス
# =============================================================================


class AddFunctionTextEmbedding:
    """
    InterPro ドメイン記述を PubMedBERT でエンコードし、
    data["feats"]["function_text_emb"] に格納する Transform。

    Classifier-Free Guidance (CFG) のため、dropout_prob の確率で
    埋め込みをゼロベクトル（null condition）に置き換える。

    推奨運用:
        事前に precompute_embeddings.py で全エントリの埋め込みを計算し、
        embedding_cache_path に保存しておく。
        これにより学習中の BERT 推論コストをゼロにできる。

    Args:
        annotation_path:
            事前構築したアノテーション JSON のパス。
            形式:
              {
                "4hhb_a": {
                  "uniprot_id": "P69905",
                  "interpro": [
                    {
                      "id": "IPR001062",
                      "name": "Globin",
                      "description": "The globin superfamily ..."
                    },
                    ...
                  ]
                },
                ...
              }
            キーは "{pdb_id}_{chain_id}" の小文字。

        encoder_name:
            HuggingFace モデル名。デフォルトは PubMedBERT-base。

        embed_dim:
            エンコーダ出力次元（PubMedBERT-base は 768）。
            embedding_cache_path を指定する場合はキャッシュと一致させること。

        dropout_prob:
            CFG null condition に落とす確率（学習時のみ有効）。

        max_length:
            テキストエンコーダへの最大トークン長。

        device:
            エンコーダを実行するデバイス。DataLoader worker との VRAM 競合を
            避けるため "cpu" を推奨。

        embedding_cache_path:
            事前計算済み埋め込みキャッシュ（.pt ファイル）のパス。
            形式: {"4hhb_a": tensor(768,), ...}
            指定した場合、キャッシュヒット時はエンコーダを呼ばない。
    """

    def __init__(
        self,
        annotation_path: str | Path,
        encoder_name: str = (
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        ),
        embed_dim: int = DEFAULT_EMBED_DIM,
        dropout_prob: float = 0.1,
        max_length: int = 256,
        device: str = "cpu",
        embedding_cache_path: Optional[str | Path] = None,
    ):
        self.annotation_path = Path(annotation_path)
        self.encoder_name = encoder_name
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        self.device = device
        self.embedding_cache_path = (
            Path(embedding_cache_path) if embedding_cache_path else None
        )

        # 以下はすべて遅延ロード（DataLoader の fork 問題を回避するため）
        self._annotations: Optional[dict] = None
        self._tokenizer = None
        self._encoder = None
        self._embedding_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # 遅延ロード
    # ------------------------------------------------------------------

    def _load_annotations(self) -> None:
        if self._annotations is not None:
            return
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_path}\n"
                "Run build_annotation_db.py to create it."
            )
        with open(self.annotation_path) as f:
            self._annotations = json.load(f)
        logger.info(
            f"[AddFunctionTextEmbedding] Loaded {len(self._annotations):,} "
            f"annotations from {self.annotation_path}"
        )

    def _load_encoder(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for AddFunctionTextEmbedding.\n"
                "Install: pip install transformers"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self._encoder = AutoModel.from_pretrained(self.encoder_name)
        self._encoder.eval()
        for param in self._encoder.parameters():
            param.requires_grad = False
        self._encoder.to(self.device)
        logger.info(
            f"[AddFunctionTextEmbedding] Loaded frozen encoder: {self.encoder_name}"
        )

    def _load_embedding_cache(self) -> None:
        if self._embedding_cache is not None:
            return
        if self.embedding_cache_path and self.embedding_cache_path.exists():
            self._embedding_cache = torch.load(
                self.embedding_cache_path, map_location="cpu"
            )
            logger.info(
                f"[AddFunctionTextEmbedding] Loaded {len(self._embedding_cache):,} "
                f"cached embeddings from {self.embedding_cache_path}"
            )
        else:
            self._embedding_cache = {}

    # ------------------------------------------------------------------
    # テキスト構築
    # ------------------------------------------------------------------

    def _build_description(self, entry: dict) -> str:
        """
        UniProt 機能記述 + Subcellular Location + InterPro エントリ名を連結した文字列を返す。

        フォーマット:
            "{UniProt function} | {Subcellular location} | {InterPro name 1} | ..."

        UniProt function がない場合は Subcellular location / InterPro 名のみ。
        InterPro 名は重複除去する（異なる DB が同名エントリを持つため）。
        """
        parts = []

        # 1. UniProt 機能記述（自然言語・最優先）
        uniprot_func = (entry.get("uniprot_function") or "").strip()
        if uniprot_func:
            parts.append(uniprot_func)

        # 2. Subcellular Location（膜タンパク質識別に有効）
        subcellular = (entry.get("uniprot_subcellular_location") or "").strip()
        if subcellular:
            parts.append(subcellular)

        # 3. InterPro エントリ名（重複除去）
        hits = entry.get("interpro", [])
        seen_names: set[str] = set()
        for hit in hits:
            name = hit.get("name", "") or ""
            desc = hit.get("description", "") or ""
            name = name.strip() if name not in ("None", "none") else ""
            desc = desc.strip()
            if name and name not in seen_names:
                seen_names.add(name)
                if desc:
                    parts.append(f"{name}: {desc}")
                else:
                    parts.append(name)

        # 4. UniProt キュレーション品質タグ
        # Swiss-Prot (手動キュレーション) は 6 文字、TrEMBL (自動注釈) は 10 文字
        uid = entry.get("uniprot_id", "")
        if uid:
            parts.append("[reviewed]" if len(uid) == 6 else "[unreviewed]")

        return " | ".join(parts)  # 空のとき "" を返す

    # ------------------------------------------------------------------
    # エンコード
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        テキスト → [CLS] 埋め込み（shape: (embed_dim,)）。
        空文字列のときはゼロベクトルを返す（null condition）。
        """
        if not text:
            return torch.zeros(self.embed_dim)

        self._load_encoder()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        ).to(self.device)
        outputs = self._encoder(**inputs)
        # last_hidden_state の先頭が [CLS] トークン
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return cls_emb  # shape: (embed_dim,)

    def _get_embedding(self, lookup_key: str) -> torch.Tensor:
        """
        lookup_key に対応する埋め込みをキャッシュまたはエンコーダから取得する。
        lookup_key は "{pdb_id}_{chain_id}" の小文字（例: "4hhb_a"）。
        """
        self._load_embedding_cache()

        # キャッシュヒット
        if lookup_key in self._embedding_cache:
            return self._embedding_cache[lookup_key]

        # キャッシュミス: アノテーションからテキストを生成してエンコード
        self._load_annotations()
        entry = self._annotations.get(lookup_key, {})

        if not entry:
            logger.debug(
                f"[AddFunctionTextEmbedding] No annotation for '{lookup_key}'. "
                "Using null embedding."
            )
            return torch.zeros(self.embed_dim)

        text = self._build_description(entry)
        emb = self._encode_text(text)

        # オンザフライでキャッシュに追加（次回以降高速化）
        self._embedding_cache[lookup_key] = emb
        return emb

    # ------------------------------------------------------------------
    # ルックアップキーの取得
    # ------------------------------------------------------------------

    def _get_lookup_key(self, data: dict) -> Optional[str]:
        """
        data["atom_array"] のメタデータから "{pdb_id}_{chain_id}" を取り出す。
        AtomWorks の annotation 名は複数のバリエーションがあるため順番に試みる。
        """
        atom_array = data.get("atom_array")
        if atom_array is None:
            return None

        # PDB エントリ ID の候補フィールド名
        for id_field in ("entry_id", "pdb_id", "structure_id"):
            entry_id = getattr(atom_array, id_field, None)
            if not entry_id:
                continue
            entry_id = str(entry_id).lower()

            # チェーン ID の候補フィールド名
            chain = None
            for chain_field in ("auth_chain_id", "chain_id", "label_chain_id"):
                try:
                    ann = atom_array.get_annotation(chain_field)
                    if ann is not None and len(ann) > 0:
                        chain = str(ann[0]).lower()
                        break
                except Exception:
                    continue

            if chain:
                return f"{entry_id}_{chain}"
            return entry_id  # チェーン不明の場合はエントリ ID のみ

        # フォールバック: extra_info に entry_id があれば使う
        extra_info = data.get("extra_info", {})
        key = extra_info.get("entry_id")
        return str(key).lower() if key else None

    # ------------------------------------------------------------------
    # Transform エントリーポイント
    # ------------------------------------------------------------------

    def __call__(self, data: dict) -> dict:
        """
        data["feats"]["function_text_emb"] に埋め込みを追加して返す。

        CFG: dropout_prob の確率でゼロベクトルに置き換える。
        """
        # CFG: null condition に落とす
        if random.random() < self.dropout_prob:
            emb = torch.zeros(self.embed_dim)
        else:
            lookup_key = self._get_lookup_key(data)
            if lookup_key is None:
                logger.debug(
                    "[AddFunctionTextEmbedding] lookup_key が取得できません。"
                    "null embedding を使用します。"
                )
                emb = torch.zeros(self.embed_dim)
            else:
                try:
                    emb = self._get_embedding(lookup_key)
                except Exception as exc:
                    logger.warning(
                        f"[AddFunctionTextEmbedding] '{lookup_key}' の埋め込み取得失敗: "
                        f"{exc}. null embedding を使用します。"
                    )
                    emb = torch.zeros(self.embed_dim)

        if "feats" not in data:
            data["feats"] = {}
        data["feats"]["function_text_emb"] = emb  # shape: (embed_dim,)
        return data


# =============================================================================
# 前処理スクリプト 1: アノテーション DB の構築
# =============================================================================


def build_annotation_db(
    sifts_tsv: str | Path,
    output_path: str | Path,
    interpro_api_pause: float = 0.2,
) -> None:
    """
    SIFTS TSV + InterPro REST API から annotation JSON を構築する。

    Args:
        sifts_tsv:
            SIFTS の PDB-UniProt マッピング TSV ファイルのパス。
            https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html からダウンロード。
            (pdb_chain_uniprot.tsv など)

        output_path:
            出力 JSON のパス。

        interpro_api_pause:
            InterPro API への連続リクエスト間の待機秒数（レート制限対策）。

    Usage:
        python -c "
        from function_text_transforms import build_annotation_db
        build_annotation_db(
            sifts_tsv='pdb_chain_uniprot.tsv',
            output_path='interpro_annotations.json',
        )
        "
    """
    import time

    import requests

    sifts_tsv = Path(sifts_tsv)
    output_path = Path(output_path)

    # --- SIFTS TSV 読み込み: PDB+Chain → UniProt ID ---
    pdb_to_uniprot: dict[str, str] = {}
    with open(sifts_tsv) as f:
        for line in f:
            if line.startswith("#") or line.startswith("PDB"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 3:
                continue
            pdb_id, chain_id, uniprot_id = cols[0], cols[1], cols[2]
            key = f"{pdb_id.lower()}_{chain_id.lower()}"
            pdb_to_uniprot[key] = uniprot_id

    logger.info(f"SIFTS: {len(pdb_to_uniprot):,} PDB chain → UniProt マッピング")

    # --- UniProt ID ごとに InterPro を取得 ---
    uniprot_to_interpro: dict[str, list[dict]] = {}
    unique_uniprot_ids = set(pdb_to_uniprot.values())

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    uid_iter = sorted(unique_uniprot_ids)
    iterable = _tqdm(uid_iter, desc="InterPro API", unit="uid") if _tqdm else uid_iter

    for i, uid in enumerate(iterable):
        if uid in uniprot_to_interpro:
            continue
        url = f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/{uid}/"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                uniprot_to_interpro[uid] = []
                continue
            resp.raise_for_status()
            if not resp.text.strip():
                uniprot_to_interpro[uid] = []
                continue
            payload = resp.json()
            hits = []
            for result in payload.get("results", []):
                meta = result.get("metadata", {})
                name_field = meta.get("name", "")
                name = (
                    name_field.get("name", "")
                    if isinstance(name_field, dict)
                    else str(name_field)
                )
                desc_field = meta.get("description")
                if isinstance(desc_field, list) and desc_field:
                    first = desc_field[0]
                    description = (
                        first.get("text", "") if isinstance(first, dict) else str(first)
                    )
                else:
                    description = ""
                hits.append(
                    {
                        "id": meta.get("accession", ""),
                        "name": name,
                        "description": description,
                    }
                )
            uniprot_to_interpro[uid] = hits
        except Exception as exc:
            logger.warning(f"InterPro API エラー ({uid}): {exc}")
            uniprot_to_interpro[uid] = []

        if i % 500 == 0:
            logger.info(f"  {i:,} / {len(unique_uniprot_ids):,} UniProt IDs 処理完了")
        time.sleep(interpro_api_pause)

    # --- JSON 出力 ---
    db: dict[str, dict] = {}
    for key, uid in pdb_to_uniprot.items():
        db[key] = {
            "uniprot_id": uid,
            "interpro": uniprot_to_interpro.get(uid, []),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f)
    logger.info(f"アノテーション DB を保存しました: {output_path} ({len(db):,} エントリ)")


# =============================================================================
# 前処理スクリプト 2: 埋め込みの事前計算
# =============================================================================


def precompute_embeddings(
    annotation_path: str | Path,
    output_path: str | Path,
    encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_every: int = 5000,
) -> None:
    """
    annotation JSON 内の全エントリの埋め込みを事前計算して .pt ファイルに保存する。
    学習時はこのキャッシュを参照することで BERT の推論コストをゼロにできる。

    出力ファイルが既に存在する場合は再開（resume）する。
    checkpoint_every バッチごとに途中保存するので中断しても安全。

    Args:
        annotation_path : build_annotation_db() で作成した JSON のパス。
        output_path     : 出力キャッシュ (.pt) のパス。
        encoder_name    : HuggingFace モデル名。
        batch_size      : エンコード時のバッチサイズ。
        device          : GPU が使えるなら "cuda" を推奨（事前計算時のみ）。
        checkpoint_every: この件数（エントリ数）ごとに途中保存する。

    Usage:
        python -c "
        from function_text_transforms import precompute_embeddings
        precompute_embeddings(
            annotation_path='interpro_annotations.json',
            output_path='embeddings_cache.pt',
            batch_size=256,
            device='cuda',
        )
        "
    """
    from transformers import AutoModel, AutoTokenizer

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    annotation_path = Path(annotation_path)
    output_path = Path(output_path)

    with open(annotation_path) as f:
        annotations = json.load(f)

    # 既存キャッシュがあれば読み込んで再開
    if output_path.exists():
        cache: dict[str, torch.Tensor] = torch.load(output_path, map_location="cpu")
        logger.info(f"既存キャッシュを再開: {len(cache):,} エントリ読み込み済み")
    else:
        cache = {}

    # 未処理キーのみ対象にする
    all_keys = list(annotations.keys())
    keys = [k for k in all_keys if k not in cache]
    logger.info(
        f"総エントリ: {len(all_keys):,} / 未処理: {len(keys):,} / 済み: {len(cache):,}"
    )

    if not keys:
        logger.info("全エントリ処理済みです。")
        return

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).eval().to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    def _build_text(entry: dict) -> str:
        parts = []
        uniprot_func = (entry.get("uniprot_function") or "").strip()
        if uniprot_func:
            parts.append(uniprot_func)
        subcellular = (entry.get("uniprot_subcellular_location") or "").strip()
        if subcellular:
            parts.append(subcellular)
        seen_names: set[str] = set()
        for hit in entry.get("interpro", []):
            name = (hit.get("name") or "").strip()
            if name in ("None", "none"):
                name = ""
            desc = (hit.get("description") or "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                if desc:
                    parts.append(f"{name}: {desc}")
                else:
                    parts.append(name)
        uid = entry.get("uniprot_id", "")
        if uid:
            parts.append("[reviewed]" if len(uid) == 6 else "[unreviewed]")
        return " | ".join(parts)

    texts = [_build_text(annotations[k]) for k in keys]

    @torch.no_grad()
    def _encode_batch(batch_texts: list[str]) -> torch.Tensor:
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)
        outputs = encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    batches = range(0, len(keys), batch_size)
    iterable = _tqdm(batches, desc="Precompute embeddings", unit="batch") if _tqdm else batches

    embed_dim = 768  # PubMedBERT-base hidden size

    for batch_idx, start in enumerate(iterable):
        batch_keys = keys[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]

        # 空テキストはゼロベクトル、非空テキストのみ GPU でエンコード
        nonempty_indices = [i for i, t in enumerate(batch_texts) if t]
        nonempty_texts = [batch_texts[i] for i in nonempty_indices]

        if nonempty_texts:
            encoded = _encode_batch(nonempty_texts)
            encoded_iter = iter(encoded)
        else:
            encoded_iter = iter([])

        for i, (key, text) in enumerate(zip(batch_keys, batch_texts)):
            if text:
                cache[key] = next(encoded_iter)
            else:
                cache[key] = torch.zeros(embed_dim)

        processed = start + len(batch_keys)
        if processed % checkpoint_every == 0 or processed == len(keys):
            torch.save(cache, output_path)
            logger.info(
                f"チェックポイント保存: {processed:,} / {len(keys):,} エントリ → {output_path}"
            )

    torch.save(cache, output_path)
    logger.info(
        f"埋め込みキャッシュを保存しました: {output_path} ({len(cache):,} エントリ)"
    )


# =============================================================================
# 前処理スクリプト 3: 失敗エントリの再試行
# =============================================================================


def retry_failed_annotations(
    annotation_path: str | Path,
    interpro_api_pause: float = 1.0,
    max_retries: int = 5,
    checkpoint_every: int = 200,
) -> None:
    """
    build_annotation_db() で通信失敗により ``"interpro": []`` になったエントリを
    再試行し、annotation JSON を上書き更新する。

    404（アノテーション本来なし）と通信エラーを区別し、
    404 は再試行対象から除外する。
    指数バックオフ付きリトライで安定性を高める。

    Args:
        annotation_path:
            build_annotation_db() が出力した JSON のパス（上書き更新される）。
        interpro_api_pause:
            リクエスト間の基本待機秒数（デフォルト 1.0 秒）。
        max_retries:
            1 ID あたりの最大リトライ回数。
        checkpoint_every:
            この件数ごとに途中保存する。

    Usage:
        from rfd3.transforms.function_text_transforms import retry_failed_annotations
        retry_failed_annotations(
            annotation_path="/path/to/interpro_annotations.json",
        )
    """
    import time

    import requests

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    annotation_path = Path(annotation_path)
    with open(annotation_path) as f:
        db: dict[str, dict] = json.load(f)

    # interpro が空のエントリから UniProt ID を収集（重複除去）
    uid_to_keys: dict[str, list[str]] = {}
    for key, entry in db.items():
        if not entry.get("interpro"):
            uid = entry.get("uniprot_id", "")
            if uid:
                uid_to_keys.setdefault(uid, []).append(key)

    unique_uids = sorted(uid_to_keys)
    logger.info(
        f"再試行対象: {len(unique_uids):,} UniProt ID "
        f"({sum(len(v) for v in uid_to_keys.values()):,} PDB エントリ)"
    )

    def _fetch_interpro(uid: str) -> Optional[list[dict]]:
        """成功時はヒットリスト、404 は None（確定空）、失敗は例外を raise。"""
        url = (
            f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/{uid}/"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None  # 確定的に空
        resp.raise_for_status()
        if not resp.text.strip():
            return []
        payload = resp.json()
        hits = []
        for result in payload.get("results", []):
            meta = result.get("metadata", {})
            name_field = meta.get("name", "")
            name = (
                name_field.get("name", "")
                if isinstance(name_field, dict)
                else str(name_field)
            )
            desc_field = meta.get("description")
            if isinstance(desc_field, list) and desc_field:
                first = desc_field[0]
                description = (
                    first.get("text", "") if isinstance(first, dict) else str(first)
                )
            else:
                description = ""
            hits.append(
                {
                    "id": meta.get("accession", ""),
                    "name": name,
                    "description": description,
                }
            )
        return hits

    updated = 0
    confirmed_empty = 0
    still_failed = 0

    iterable = _tqdm(unique_uids, desc="Retry InterPro", unit="uid") if _tqdm else unique_uids

    for i, uid in enumerate(iterable):
        hits: Optional[list[dict]] = None
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                hits = _fetch_interpro(uid)
                break
            except Exception as exc:
                last_exc = exc
                backoff = interpro_api_pause * (2 ** attempt)
                time.sleep(backoff)

        if hits is None and last_exc is None:
            # 404: 確定的に空
            confirmed_empty += 1
        elif hits is not None:
            # 成功: 全関連 PDB エントリを更新
            for key in uid_to_keys[uid]:
                db[key]["interpro"] = hits
            if hits:
                updated += len(uid_to_keys[uid])
        else:
            # 全リトライ失敗
            logger.warning(f"再試行失敗 ({uid}): {last_exc}")
            still_failed += 1

        time.sleep(interpro_api_pause)

        # 途中保存
        if (i + 1) % checkpoint_every == 0:
            with open(annotation_path, "w") as f:
                json.dump(db, f)
            logger.info(
                f"  チェックポイント保存 ({i + 1:,}/{len(unique_uids):,}) "
                f"更新={updated:,} 確定空={confirmed_empty:,} 失敗={still_failed:,}"
            )

    with open(annotation_path, "w") as f:
        json.dump(db, f)
    logger.info(
        f"完了: 更新={updated:,} 確定空={confirmed_empty:,} "
        f"再試行失敗={still_failed:,} → {annotation_path}"
    )