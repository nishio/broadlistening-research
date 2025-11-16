# LLM呼び出し費用見積もり

## データ概要

- **総コメント数**: 27,657件
- **平均コメント長**: 40.5文字（日本語）
- **既存分類体系**: 3大分類 × 22中分類

## 実施内容

### オプション1: Categorization のみ（既存分類を使用）

**想定作業:**
- 既存の22カテゴリに対して、27,657件のargumentsを分類
- Topic Identification は実施しない（ユーザー提供のトピックを使用）

**トークン見積もり（FEEDBACK_2.mdより）:**
- ベースライン: 1,000件で 130K input / 50K output
- 今回のデータ: 27,657件 = 約27.7倍

計算:
- Input tokens: 130K × 27.7 = **約3,601,000 tokens**
- Output tokens: 50K × 27.7 = **約1,385,000 tokens**

**費用見積もり（Gemini 1.5 Pro想定）:**

料金表（2024年時点の一般的な料金）:
- Input: $1.25 per 1M tokens
- Output: $5.00 per 1M tokens

計算:
- Input cost: 3.601M × $1.25 = **$4.50**
- Output cost: 1.385M × $5.00 = **$6.93**
- **総費用: 約$11.43**

---

### オプション2: Full pipeline（Topic Identification + Categorization + Summarization）

**トークン見積もり:**
- Topic Identification: 130K input / 50K output × 27.7 = 3,601K / 1,385K
- Categorization: 130K input / 50K output × 27.7 = 3,601K / 1,385K
- Summarization: 80K input / 7.5K output × 27.7 = 2,216K / 208K

合計:
- Input tokens: **約9,418,000 tokens**
- Output tokens: **約2,978,000 tokens**

**費用見積もり:**
- Input cost: 9.418M × $1.25 = **$11.77**
- Output cost: 2.978M × $5.00 = **$14.89**
- **総費用: 約$26.66**

---

## 推奨オプション

今回の実験目的は「既存の分類体系に新しいデータをはめ込む」ことなので、**オプション1: Categorization のみ**を推奨します。

**推定費用: 約$11.43**

---

## 注意事項

1. **実際の費用は変動する可能性があります:**
   - 上記はFEEDBACK_2.mdの情報（英語ベースの見積もり）を基にした概算
   - 日本語テキストはトークン数が異なる可能性がある
   - Vertex AIの料金は地域やプロジェクトによって異なる

2. **データサイズによる調整:**
   - 日本語は1文字あたり約1.5-2トークン程度
   - 平均40.5文字 × 27,657件 = 約1,119,000文字
   - 推定トークン数: 約1.7M-2.2M tokens（input）
   - 実際のトークン数は上記見積もりより少ない可能性がある

3. **Vertex AI認証が必要:**
   ```bash
   gcloud config set project <your-project-id>
   gcloud auth application-default login
   ```

4. **代替案:**
   - より安価なモデル（Gemini 1.5 Flash）を使用する選択肢もある
   - サンプリング（例: 1,000件のみで実験）することで費用を削減可能

---

## 実行前の確認事項

- [ ] Google Cloud Projectの設定
- [ ] Vertex APIの有効化
- [ ] 予算アラートの設定
- [ ] 実行するデータ量の確認（全データ vs サンプル）

---

**作成日**: 2025-11-15
**データ**: 27,657件のarguments
