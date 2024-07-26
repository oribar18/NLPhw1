from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


def q_function(sentence, k, t, u, v, pre_trained_weights, feature2id, tags):
    nominator = 0
    denominator = 0
    for tag in tags:
        if k == len(sentence) - 1:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, "~")
        else:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
        if history not in feature2id.histories_features.keys():
            features = represent_input_with_features(history, feature2id.feature_to_idx)
        else:
            features = feature2id.histories_features[history]
        current_calc = np.exp(sum([pre_trained_weights[feature] for feature in features]))
        if tag == v:
            nominator = current_calc
        denominator += current_calc

    return nominator / denominator


def memm_viterbi(sentence, pre_trained_weights, feature2id, num_of_beam):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence) - 1
    sentence = sentence[:n]  # remove the ~
    pi = {}
    bp = {}
    tags = feature2id.feature_statistics.tags  # all possible tags
    for k in range(n):
        for u in tags:
            for v in tags:
                pi[(k, u, v)] = 0
                bp[(k, u, v)] = ''
    pi[(1, '*', '*')] = 1
    beam_tags = {1: '*'}  # using beam search
    for k in range(2, n):
        scores = {}
        if k == 2:
            t = '*'
            u = '*'
            for v in tags:
                q = q_function(sentence, k, t, u, v, pre_trained_weights, feature2id, tags)
                score = pi[(k-1, t, u)] * q  # calculate the score for those tags
                pi[(k, u, v)] = score
                bp[(k, u, v)] = t
                scores[v] = score
            top_2_tags = sorted(scores, key=scores.get, reverse=True)[:num_of_beam]
            beam_tags[k] = ('*', top_2_tags)
            continue
        elif k == 3:
            t_possible = ['*']
        else:
            t_possible = beam_tags[k-1][0]

        for u in beam_tags[k-1][1]:
            for v in tags:
                max_score = -1
                max_tag = None
                for t in t_possible:
                    q = q_function(sentence, k, t, u, v, pre_trained_weights, feature2id, tags)
                    score = pi[(k-1, t, u)] * q  # calculate the score for those tags
                    if score > max_score:
                        max_score = score
                        max_tag = t
                scores[(u, v)] = max_score
                pi[(k, u, v)] = max_score
                bp[(k, u, v)] = max_tag
        top_2_keys = sorted(scores, key=scores.get, reverse=True)[:num_of_beam]
        top_2_u = [key[0] for key in top_2_keys]
        top_2_v = [key[1] for key in top_2_keys]
        beam_tags[k] = [top_2_u, top_2_v]
    max_score = -1
    max_tags = None
    for key in pi.keys():
        if key[0] == n - 1:
            if pi[key] > max_score:
                max_score = pi[key]
                max_tags = (key[1], key[2])
    predicted_tags = [""] * n  # list of predicted tags
    predicted_tags[n-2] = max_tags[0]
    predicted_tags[n - 1] = max_tags[1]
    for k in range(n-3, -1, -1):  # find all predicted tags
        predicted_tags[k] = (bp[(k+2, predicted_tags[k+1], predicted_tags[k+2])])
    return predicted_tags[2:]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, file_type):
    tagged = file_type in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, 2)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


def calculate_confusion_matrix_and_accuracy(true_labels_path, predictions_path):
    true_labels = []
    predicted_labels = []

    # Read true labels and predictions
    with open(predictions_path, 'r') as preds, open(true_labels_path, 'r') as trues:
        for pred_line, true_line in zip(preds.readlines(), trues.readlines()):
            if pred_line.endswith('\n'):
                pred_parts = pred_line[:-1].split(' ')
            else:
                pred_parts = pred_line.split(' ')
            if true_line.endswith('\n'):
                true_parts = true_line[:-1].split(' ')
            else:
                true_parts = true_line.split(' ')

            # Iterate over pairs of predicted and true labels
            for pred_part, true_part in zip(pred_parts, true_parts):
                _, predicted_tag = pred_part.rsplit('_', 1)
                _, true_tag = true_part.rsplit('_', 1)
                predicted_labels.append(predicted_tag)
                true_labels.append(true_tag)

    # Compute confusion matrix
    labels = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Get top 10 confused tags
    top_10_confused = top_10_confused_tags(cm, labels)

    # Filter confusion matrix to include only top 10 confused tags
    top_10_confused_cm = cm_df.loc[top_10_confused, top_10_confused]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    return top_10_confused_cm, accuracy


def top_10_confused_tags(cm, tags):
    top_indexes = []  # List of tuples of the sum of the row (without diagonal value) and the index of the row
    for i in range(len(cm)):
        row = cm[i]
        top_indexes.append((sum([row[j] for j in range(len(cm)) if j != i]), i))
    top_indexes.sort(reverse=True)
    top_10_confused_tags = [tags[i] for _, i in top_indexes[:10]]
    return top_10_confused_tags