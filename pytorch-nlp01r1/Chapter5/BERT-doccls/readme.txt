---------------------------------------------------------
BERT を使った文書分類のデータ
（xtrain.pkl, ytrain.pkl, xtest.pkl, ytest.pkl）
の作り方
---------------------------------------------------------
(1)

以下の URL から ldcc-20140209.tar.gz をダウンロードする。
https://www.rondhuit.com/download.html#ldcc

(2)

Lhaplus などで ldcc-20140209.tar.gz を解凍する。
以下のディレクトリとファイルができる。

    text/
      dokujo-tsushin/
      it-life-hack/
      kaden-channel/
      livedoor-homme/
      movie-enter/
      peachy/
      smax/
      sports-watch/
      topic-news/
      CHANGES.txt
      README.txt

(3)

text のあるディレクトリで mkldcc.py を実行する。
> python mkldcc.py
==>  train.tsv  と test.tsv  が作成される。

(4)

train.tsv  と test.tsv を  BERT-doccls のディレクトリ下にコピーする。

(5)

BERT-doccls のディレクトリ下で以下を実行する。

> python mkdata11.py
==>  xtrain.pkl と ytrain.pkl  が作成される。

> python mkdata12.py
==>  xtest.pkl と ytest.pkl  が作成される。

