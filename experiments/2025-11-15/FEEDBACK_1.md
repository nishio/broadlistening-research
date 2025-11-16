 ## human


>『「未来の東京」の実現に向けた重点政策方針2024』 >デジタルブック形式で公開されています。 >URL は PR TIMES の本文中に載っていて、そこから飛べます（…/houshin_2024/）。 これをデータとして取得したい

## ai

いちばん手っ取り早いのは、**デジタルブック経由ではなく「公式PDF」を取る**ことです。

---

## 1. 公式PDFの場所

PR TIMES からリンクされている **「未来の東京」戦略の推進** ページに

> 「未来の東京」の実現に向けた重点政策方針2024（PDF：12.1MB）

というリンクがあります。([政策企画部][1])

### ブラウザでの手順

1. ブラウザで
   `https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/choki-plan/suishin`
   を開く。
2. ページ内で
   「**『未来の東京』の実現に向けた重点政策方針2024（PDF：12.1MB）」
   というリンクを探す。
3. そのリンクを **右クリック→「リンク先を保存」** でPDFをダウンロード。

これで、「デジタルブック」と同内容のPDFがまるごと手元に入るはずです。

---

## 2. コマンドラインで取りたい場合

PDFリンクの URL を上の手順で「リンクのアドレスをコピー」してから、例えば macOS / Linux なら：

```bash
curl -L -o houshin_2024.pdf '<<さっきコピーしたPDFのURL>>'
```

のようにすればスクリプトからも取得できます。

---

## 3. どうしても `…/houshin_2024/` から取りたい場合

`spt.metro.tokyo.lg.jp/seisakukikaku/houshin_2024/` のデジタルブックには、ビューア上部に **「PDF」ボタン** が出ているはずです。([東京都交通局スポーツイベント情報][2])

1. デジタルブックを開く
2. ビューアのツールバーから「**PDF**」を押す
3. ブラウザが開いたPDFタブで「保存」 or 右クリック保存

よりハックっぽくやるなら、PDFボタンを押したときに:

* ブラウザの「開発者ツール → Network」タブを開いて
* `*.pdf` へのリクエストURL をコピー

すれば、そのURLをそのまま `curl` などから叩けます。

---

## 4. 取得後どうするか

* PDFをそのままここにアップロードしてくれれば、
  → 見出し構造の抽出 / テキスト化 / diff などはこっちでやります。
* ローカルでテキスト化するなら

  * `pdftotext`（poppler）
  * `pdfplumber`（Python）
    あたりを使うのが扱いやすいです。

まずは suishin ページの **PDFリンクを右クリックで保存**、が一番ラクなルートだと思います。

[1]: https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/choki-plan/suishin?utm_source=chatgpt.com "「未来の東京」戦略の推進 - 政策企画局"
[2]: https://www.spt.metro.tokyo.lg.jp/seisakukikaku/houshin_2024/?utm_source=chatgpt.com "「未来の東京」の実現に 向けた重点政策方針2024"


## human

「『未来の東京』戦略」も取得したい

##ai

「『未来の東京』戦略」は、**フルセットPDF（book.pdf）**が公式に公開されています。

### 1. 公式PDFのURL

東京都の別資料の脚注に、公式URLが明記されています：([Digital Innovation City協議会][1])

```text
https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/mirainotokyo-senryaku/book.pdf
```

これが **『未来の東京』戦略 本編（約400ページ）のPDF** です。

---

### 2. ブラウザから取得する場合

1. 上記URLをブラウザのアドレスバーに貼る
2. PDFが開いたら

   * 「ファイル → 名前を付けて保存」
   * あるいは右クリック → 「名前を付けて保存」

でローカルに保存できます。

---

### 3. コマンドラインで取得する場合（macOS / Linux）

```bash
curl -L -o mirainotokyo-senryaku.pdf \
  'https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/mirainotokyo-senryaku/book.pdf'
```

あるいは `wget` を使うなら：

```bash
wget -O mirainotokyo-senryaku.pdf \
  'https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/mirainotokyo-senryaku/book.pdf'
```

---

### （補足）version up 2024も欲しい場合

直近版の **「『未来の東京』戦略 version up 2024」** も、同様に book.pdf が公開されています：([東京都交通局スポーツイベント情報][2])

```text
https://www.spt.metro.tokyo.lg.jp/seisakukikaku/versionup2024/book.pdf
```

こちらも同じ要領で保存できます。

---

もしこのPDFを後で、

* 特定章だけ抜き出す
* 2050東京戦略との diff を取りたい

などあれば、その前提で処理フローも一緒に組みましょう。

[1]: https://tokyo-dic.jp/conference/pdf/04mezasubekisugata.pdf?utm_source=chatgpt.com "第1回Digital Innovation City協議会 資料 令和3年4月28日（水）"
[2]: https://www.spt.metro.tokyo.lg.jp/seisakukikaku/versionup2024/book.pdf?utm_source=chatgpt.com "「未来の東京」戦略 version up 2024"
