from sklearn.metrics import precision_score, recall_score, f1_score
from lib.constant import (
    first_keyword2weather, other_keyword2weather, 
    first_weather_keyword_list, other_weather_keyword_list,
    first_keyword_list, other_keyword_list
)

def classification_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return p, r, f1

def extract_weather_labels_from_comment(comment):
    """天気予報コメントから天気ラベルを抽出

    Args:
        comment (str): [description]

    Returns:
        [list]: 抽出した天気ラベル
    """    
    # 1文目
    first_sentence_words = comment.split("。")[0].split()
    extracted_keyword_list = list(filter(lambda x: x in first_keyword_list, first_sentence_words)) # キーワードマッチした単語を抽出
    extracted_weather_list = [first_keyword2weather[keyword] for keyword in extracted_keyword_list]

    # 2分目以降
    other_sentence_words = " ".join(comment.split("。")[1:]).split()
    extracted_keyword_list = list(filter(lambda x: x in other_keyword_list, other_sentence_words)) # キーワードマッチした単語を抽出
    extracted_weather_list += [other_keyword2weather[keyword] for keyword in extracted_keyword_list]

    # 天気情報を先頭から取り出す（重複禁止）
    weather_labels = []
    for weather in extracted_weather_list:
        if weather not in weather_labels:
            weather_labels.append(weather)

    return weather_labels

def calc_weather_labels_accuracy(hyp_sentences, ref_sentences):
    hyp_weather_labels = {"晴れ": [], "雨": [], "曇り": [], "雪": []}
    ref_weather_labels = {"晴れ": [], "雨": [], "曇り": [], "雪": []}

    for hyp_sen, ref_sen in zip(hyp_sentences, ref_sentences):
        # 天気ラベルの抽出
        hyp_labels = extract_weather_labels_from_comment(hyp_sen)
        ref_labels = extract_weather_labels_from_comment(ref_sen)

        # 生成テキストの天気ラベル
        for label, v in hyp_weather_labels.items():
            if label in hyp_labels:
                hyp_weather_labels[label].append(1)
            else:
                hyp_weather_labels[label].append(0)

        # 参照テキストの天気ラベル
        for label, v in ref_weather_labels.items():
            if label in ref_labels:
                ref_weather_labels[label].append(1)
            else:
                ref_weather_labels[label].append(0)

    results = {}
    for label in ref_weather_labels.keys():
        f1 = f1_score(ref_weather_labels[label], hyp_weather_labels[label])
        p = precision_score(ref_weather_labels[label], hyp_weather_labels[label])
        r = recall_score(ref_weather_labels[label], hyp_weather_labels[label])
        results[label+"_F1"] = f1
        results[label+"_P"] = p
        results[label+"_R"] = r

    return results