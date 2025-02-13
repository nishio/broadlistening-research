# ノートの命名規則

## 基本規則
1. ファイル名の形式
   - パターン: `{category}-{description}.md`
   - 例: `data-restoration.md`, `experiment-kmeans-clustering.md`

2. カテゴリー一覧
   - `data`: データ関連（データ処理、保存、復元など）
   - `experiment`: 実験関連（クラスタリング、評価など）
   - `setup`: セットアップ関連（環境構築、依存関係など）
   - `doc`: ドキュメント関連（API仕様、使用方法など）
   - `meeting`: ミーティング関連（議事録、決定事項など）

3. 説明部分の規則
   - 英語の小文字のみを使用
   - 単語間はハイフン（-）で区切る
   - 簡潔かつ内容を表す名前を使用

## 例
```
data-restoration.md          # データ復元に関するノート
experiment-kmeans-1226.md    # 1226クラスタのk-means実験に関するノート
setup-python-env.md         # Python環境のセットアップに関するノート
doc-api-specification.md    # API仕様に関するドキュメント
meeting-weekly.md          # 週次ミーティングノート
```

## 既存のノートの扱い
1. 既存のノートの名前は変更しない（リンク切れを防ぐため）
2. 新規作成するノートのみ、この命名規則に従う
