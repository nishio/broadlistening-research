# Context

「東京都の成功ケースでの既存の分類との照らし合わせはGTTがやったので詳細不明」というままでは良くなくて、今後の広聴AIの発展のためにもこちらの手元で再現実験が必要な気持ちになってきた
大前提として、Jigsaw Sensemakerは「与えられたデータから分類ツリーを作成」「その分類ツリーに与えられたデータをはめ込み」の2段階になっており、この後半部分がほぼGTTのやったことと同じであると思っている

東京都のケース: https://www.seisakukikaku.metro.tokyo.lg.jp/basic-plan/choki-plan/ikenbosyu

ブロードリスニング前の公式な出発点として扱われているのは、「『未来の東京』戦略」「『未来の東京』の実現に向けた重点政策方針2024」

# Plan

1: https://broadlistening.seisakukikaku.metro.tokyo.lg.jp/20250131/index.html をcurlなどで取得する
2: この中にあるargumentsのデータを取得するスクリプトを作る、抽出する
3: ブロードリスニング前の情報を取得する
4: Jigsaw SenseMakerをgit cloneし、(3)から得た「以前の分類」に対して(2)のデータをはめ込み分類する。この時にうまくはまらないものは「その他」に入れると良い