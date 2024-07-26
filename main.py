import pickle
from sklearn.model_selection import KFold
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test, calculate_confusion_matrix_and_accuracy, top_10_confused_tags
import time
import numpy as np


def main(threshold, lam, test_path, train_path, file_type, predictions_path):
    start = time.time()
    threshold = threshold
    lam = lam

    train_path = train_path
    test_path = test_path

    weights_path = 'weights.pkl'
    predictions_path = predictions_path

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
    end = time.time()
    print(f'train time: {end - start} seconds')

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    with open(predictions_path, 'w') as f:
        f.write('')
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, file_type)
    if 'comp' not in test_path:
        cm, acc = calculate_confusion_matrix_and_accuracy(test_path, predictions_path)
        print('Confusion Matrix:\n', cm)
        print('Accuracy:', acc)
        return acc


def cross_validation():
    thresholds = [1, 2, 5, 10, 20]
    lams = [0.1, 0.5, 1, 2]
    thresh_acc = []
    for threshold in thresholds:
        print(f'threshold: {threshold}')
        thresh_acc.append((main(threshold, 1), threshold))
    max_acc_thresh = max(thresh_acc, key=lambda x: x[1])
    lam_acc = []
    for lam in lams:
        print(f'lambda: {lam}')
        lam_acc.append((main(max_acc_thresh[1], lam), lam))
    max_acc_lam = max(lam_acc, key=lambda x: x[1])
    print(f'best threshold: {max_acc_thresh[1]}, best lambda: {max_acc_lam[1]}')


def k_fold_cross_validation(data_path, weights_path='weights.pkl', predictions_path='predictions.wtag', k=7,
                            threshold=1, lam=1, file_type='test'):
    """
    Perform k-fold cross validation on a dataset of sentences.
    :param data_path: The path to the dataset.
    :param weights_path: The path to save the weights.
    :param predictions_path: The path to save the predictions.
    :param sentences: Array of sentences.
    :param k: Number of folds.
    :return: The average accuracy across all folds.
    """
    with open(data_path, 'r') as file:
        data = file.readlines()  # Each line is a sentence
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold = 0
    accuracies = []

    for train_index, test_index in kf.split(data):
        fold += 1
        print(f"Fold #{fold}")

        # Split data into training and test for the current fold
        train_data = [data[i] for i in train_index]  # Adjusted to use list comprehension
        test_data = [data[i] for i in test_index]

        fold_train_data = 'fold_train_data.wtag'
        fold_test_data = 'fold_test_data.wtag'
        with open(fold_train_data, 'w') as f:
            f.write(''.join(train_data))
        with open(fold_test_data, 'w') as f:
            f.write(''.join(test_data))

        # Note: Adjust preprocess_train, get_optimal_vector, and tag_all_test functions to work with data directly instead of file paths

        # Process training data
        statistics, feature2id = preprocess_train(fold_train_data, threshold)

        # Perform training and save weights
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        # Load trained weights
        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        # Evaluate on the test data of the current fold
        with open(predictions_path, 'w') as f:
            f.write('')
        tag_all_test(fold_test_data, pre_trained_weights, feature2id, predictions_path, file_type)
        accuracy = calculate_confusion_matrix_and_accuracy(fold_test_data, predictions_path)[1]
        accuracies.append(accuracy)

        print(f"Accuracy for fold #{fold}: {accuracy}")
        # draw_confusion(all_tags, all_pred)
    for i in range(len(accuracies)):
        print(f"Accuracy for fold #{i}: {accuracies[i]}")
    # Calculate and print the average accuracy over all folds
    print(f"Average accuracy over {k} folds: {np.mean(accuracies)}")


if __name__ == '__main__':
    # print('train2:')
    # main(1, 1, "data/train2.wtag", "data/train2.wtag", 'train', "predictions.wtag")
    # print('test1:')
    # main(1, 1, "data/test1.wtag", "data/train1.wtag", 'test', "predictions.wtag")
    # print('test2:')
    # k_fold_cross_validation("data/train2.wtag")
    # print('train1:')
    # main(1, 1, "data/train1.wtag", "data/train1.wtag", 'train', "predictions.wtag")
    print('comp1:')
    main(1, 1, "data/comp1.words", "data/train1.wtag", 'test', "comp_m1_316137371_314968595.wtag")
    print('comp2:')
    main(1, 1, "data/comp2.words", "data/train2.wtag", 'test', "comp_m2_316137371_314968595.wtag")