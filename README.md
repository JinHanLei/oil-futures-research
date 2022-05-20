# Oil Futures Research

## Setup

    $ pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

## Download news data

    $ wget https://github.com/JinHanLei/oil-futures-research/releases/download/v1/china5e_news.csv
    $ mv china5e_news.csv ./data/

## Run

    $ python train_GRU.py
    $ python train_RGRU.py
