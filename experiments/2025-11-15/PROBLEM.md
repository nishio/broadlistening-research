# 未解決の問題

実験日: 2025-11-15

## 問題1: Vertex AI で Gemini 1.5 系モデルが見つからない ✅ 解決

### 現象

以下のmodel IDで404エラーが発生:

```
404 Not Found: Publisher Model `projects/gen-lang-client-0654710684/locations/us-central1/publishers/google/models/gemini-1.5-pro` not found.
```

同様に `gemini-1.5-flash` も見つからない。

### 原因（判明）

**Gemini 1.5 系は2025年に完全廃止済み**

公式情報:
- `gemini-1.5-pro-001`: 2025-05-24 廃止
- `gemini-1.5-pro-002`: 2025-09-24 廃止
- `gemini-1.5-flash-001/002`: 同様に廃止

参考:
- [Model versions and lifecycle](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)
- [Vertex AI release notes](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/release-notes)
- [Firebase: Retired モデルは404を返す](https://firebase.google.com/docs/ai-logic/models)

**結論**: 404エラーは「model IDが間違っている」のではなく、**サービス側がもう受け付けていない**ため。

### 利用可能なモデル（2025-11-15時点）

Vertex AI で現在使える安定版Geminiモデル:

1. **`gemini-2.5-flash-lite`** ⭐ 推奨
   - 最安・高速（入力 $0.10/100万トークン、出力 $0.40/100万トークン）
   - 大規模処理向け、分類タスクに最適

2. **`gemini-2.5-flash`**
   - バランス型（価格と性能のコスパ重視）

3. **`gemini-2.5-pro`**
   - 最強の推論モデル（Thinking機能あり）
   - 複雑タスク向け

4. **`gemini-2.0-flash`**
   - 旧世代だが現役のコスパモデル

5. **`gemini-2.0-flash-lite`**

参考: [Google models | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models)

### 今回の実験での推奨モデル

分類タスク（既存トピックツリーへの振り分け）には:

**第一候補**: `gemini-2.5-flash-lite`
- 最もコスト効率が良い
- 分類タスクには十分な性能

**代替案**:
- 精度重視なら `gemini-2.5-flash`
- さらに慎重なら `gemini-2.0-flash`

### 影響

- ✅ 「正しい1.5のmodel ID」を探す作業は不要（廃止済みのため）
- ✅ 2.5 Flash-Lite なら見積もりより大幅に安くなる可能性
- ⚠️ コスト見積もりの再計算が必要

---

## 問題2: 元のモデル gemini-2.5-pro-preview-06-05 のコスト ❓

### 背景

- Jigsaw Sensemakerのデフォルトモデル: `gemini-2.5-pro-preview-06-05`
- ユーザーは明示的に「1.5-pro にして」と指示していた
- 理由: コスト削減

### 未確認事項

1. **Gemini 2.5 Pro preview の料金**
   - 公式料金表で確認できなかった
   - Gemini 2.5 Pro (GA) の料金は？
   - Preview版は GA版と料金が異なる？

2. **見積もりへの影響**
   - 1,000件で $0.41 の見積もりは Gemini 1.5 Pro ベース
   - Gemini 2.5 Pro preview だと何倍になる？

### 必要なアクション

- Gemini 2.5 Pro preview の料金を確認
- Gemini 1.5 Pro と比較して、コスト増加率を計算

---

## 問題3: 10件テストの失敗を見逃した ❌

### 何が起きたか

1. **Task 10 で「10件テスト成功」と判断**
   - バックグラウンド実行でexit code 0を確認
   - `small_test.log` が0バイトで空だったことを見逃した
   - 実際の出力内容を一度も確認していない

2. **実際には全て失敗していた**
   - ログが空 = 何も出力されずに終了
   - おそらく404エラーで即座に終了していた

### 根本原因

1. **exit code 0 だけで成功と判断してしまった**
   - TypeScriptのエラーハンドリングが不十分？
   - プロセスが即座に終了している

2. **バックグラウンド実行の落とし穴**
   - 出力が見えないため、問題に気づきにくい
   - ログファイルのサイズ確認を怠った

3. **検証の不足**
   - 結果ファイル（categorized_results.csv）の存在確認をしなかった
   - ログの内容を読まなかった

### 改善策

今後の実験では:
1. ✅ ログファイルのサイズを確認
2. ✅ 結果ファイルの存在を確認
3. ✅ 実際の出力内容を読む
4. ✅ バックグラウンド実行せず、直接実行して出力を確認

---

## 問題4: 次のステップの選択肢 ✅ 方針決定

### 決定事項

Gemini 1.5 系は廃止済みなので、**Gemini 2.0/2.5 系に移行する**

### 推奨モデル（優先順）

#### オプション1: `gemini-2.5-flash-lite` ⭐ 最推奨

**メリット:**
- **最もコスト効率が良い**（入力 $0.10/100万トークン、出力 $0.40/100万トークン）
- 分類タスクには十分な性能
- 見積もりより大幅に安くなる

**設定:**
```typescript
modelName: "gemini-2.5-flash-lite"
location: "global"  // または "us-central1"
```

**想定コスト（1,000件）:**
- 元の見積もり（Gemini 1.5 Pro）: $0.41
- Flash-Lite での推定: **約$0.05-0.10** （大幅削減）

#### オプション2: `gemini-2.5-flash`

**メリット:**
- バランス型（性能とコストのトレードオフ）
- Flash-Lite より精度が必要な場合に

**設定:**
```typescript
modelName: "gemini-2.5-flash"
location: "global"
```

#### オプション3: `gemini-2.0-flash`

**メリット:**
- 旧世代だが現役で安定
- 実績が豊富

**設定:**
```typescript
modelName: "gemini-2.0-flash"
location: "us-central1"  // 通常のリージョンで利用可能
```

### ~~廃止されたオプション~~

- ~~Gemini 1.5 系の正しいmodel IDを見つける~~ → **廃止済みのため不可能**
- ~~Gemini 2.5 Pro preview を使う~~ → 安定版GAが利用可能なため不要

---

## 推奨される次のアクション ✅ 更新版

### 1. モデル設定の変更

**`jigsaw-sensemaker/library/src/models/vertex_model.ts`** を編集:

```typescript
constructor(
  project: string,
  location: string,
  modelName: string = "gemini-2.5-flash-lite"  // ← 変更
) {
  super();
  this.vertexAI = new VertexAI({
    project: project,
    location: location,
    apiEndpoint: "aiplatform.googleapis.com",
  });
  this.modelName = modelName;
  // ...
}
```

**location** の設定:
- 2.5 系: `global` または `us-central1` どちらでも可
- 2.0 系: `us-central1` が一般的

### 2. エラーハンドリングの強化

失敗時に確実に異常終了するよう、スクリプトを修正:

```typescript
try {
  const categorizedComments = await sensemaker.categorizeComments(...);
  // 成功処理
} catch (error) {
  console.error('\n❌ Error:');
  console.error(error);
  process.exit(1);  // ← 確実に異常終了
}
```

### 3. 小規模テスト（10件）で動作確認

**重要**: 必ず直接実行して出力を目視確認

```bash
cd /Users/nishio/broadlistening-research/experiments/2025-11-15
export GOOGLE_CLOUD_PROJECT="gen-lang-client-0654710684"
export DEFAULT_VERTEX_PARALLELISM=10
npx ts-node run_categorization_small.ts
```

**確認項目**:
- ✅ エラーなく完了
- ✅ 結果ファイル（categorized_results.*）が生成される
- ✅ ログに分類結果が表示される
- ✅ 10件すべてにトピックが割り当てられている

### 4. 1,000件実験の実行

10件テストが成功したら:

```bash
npx ts-node run_categorization.ts
```

### 5. コスト実測と全データ実験の判断

1,000件の実測コストを確認:
- Vertex AI コンソールで実際の使用量を確認
- 27,657件での推定コストを計算
- ユーザーと相談して全データ実験の可否を判断

---

## 現在のファイル状態

### 変更が必要なファイル

1. **`jigsaw-sensemaker/library/src/models/vertex_model.ts`**
   - 現在: `modelName: string = "gemini-1.5-flash"`
   - 正しいmodel IDに変更が必要

2. **`run_categorization.ts`**
   - 現在: `location: 'global'`
   - モデルに応じて適切なlocationに変更

3. **`run_categorization_small.ts`**
   - 現在: `location: 'us-central1'`
   - モデルに応じて適切なlocationに変更

### 追加で作成したファイル

- `tsconfig.json`: TypeScript設定（必要）

---

## メモ

- ユーザーからの明示的な指示: 「gemini-2.5-pro-preview-06-05 は使えないの 1.5-proにして」
- コスト見積もり (Gemini 1.5 Pro, 1,000件): $0.41
- 実験の目的: 東京都の既存分類体系への分類を再現
