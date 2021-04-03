# -*- coding: utf-8 -*-
from enum import Enum, unique

@unique
class Tokens(Enum):
    UNK = '<unk>'
    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'

@unique
class IDs(Enum):
    UNK = 0
    BOS = 1
    EOS = 2
    PAD = 3

@unique
class Phases(Enum):
    Train = "train"
    Valid = "valid"
    Test = "test"

@unique
class META(Enum):
    TIME = "time"
    WEEK = "week"
    AREA = "area"
    DAY = "day"
    MONTH = "month"

PRECIPITATION = 0
AIR_PRESSURE = 1
AIR_PRESSURE_AT_SEA_LEVEL = 2
AIR_TEMPERATURE = 3
CLOUD_AREA_FRACTION = 4
HIGH_TYPE_CLOUD_AREA_FRACTION = 5
LOW_TYPE_CLOUD_AREA_FRACTION = 6
MEDIUM_TYPE_CLOUD_AREA_FRACTION = 7
RELATIVE_HUMIDITY = 8
X_WIND = 9
Y_WIND = 10

weather_keyword_list = {
    "晴れ": ["晴れ", "日差し", "回復", "日和", "陽気", "秋晴れ", "晴天", "青空", "晴れ間", "晴れる", "太陽", "五月晴れ"],
    "曇り": ["曇り", "雲", "曇"], 
    "雨": ["雨", "雷雨", "暴風雨", "にわか雨", "大雨", "雨風", "荒天", "台風", "傘"],
    "雪": ["雪", "吹雪", "吹雪く", "小雪", "ふぶく"],
}

# 1文目用
first_weather_keyword_list = {
    "晴れ": ["晴れ", "日差し", "回復", "日和", "陽気", "秋晴れ", "晴天", "青空", "晴れ間", "晴れる", "太陽", "五月晴れ"],
    "曇り": ["曇り", "雲", "曇"], 
    "雨": ["雨", "雷雨", "暴風雨", "にわか雨", "大雨", "雨風", "荒天", "台風", "傘"],
    "雪": ["雪", "吹雪", "吹雪く", "小雪", "ふぶく"],
}
# 2文目以降用
other_weather_keyword_list = {
    "晴れ": ["晴れ", "日差し", "回復", "日和", "陽気", "秋晴れ", "晴天", "青空", "晴れ間", "晴れる"],
    "曇り": ["曇り", "雲", "曇"], 
    "雨": ["雨", "雷雨", "暴風雨", "にわか雨", "大雨", "雨風"],
    "雪": ["雪", "吹雪", "吹雪く", "小雪", "ふぶく"],
}

first_keyword2weather = {}
for weather, keyword_list in first_weather_keyword_list.items():
    for keyword in keyword_list:
        first_keyword2weather[keyword] = weather

other_keyword2weather = {}
for weather, keyword_list in other_weather_keyword_list.items():
    for keyword in keyword_list:
        other_keyword2weather[keyword] = weather

first_keyword_list = list(first_keyword2weather.keys())
other_keyword_list = list(other_keyword2weather.keys())

# メタ単語
meta_keywords = ["明日", "今日", "午前", "午後", "( 月 )", "( 火 )", 
    "( 水 )", "( 木 )", "( 金 )", "( 土 )", "( 日 )", "春", "夏", "秋", "冬"]