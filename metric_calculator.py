def overall_accuracy(test_target, test_prediction):
    correct_prediction_count = 0
    for target, prediction in zip(test_target, test_prediction):
        if prediction[0] == prediction[1]:
            if target[0] == target[1]:
                if prediction[0] in target:
                    correct_prediction_count += 2
            else:
                if prediction[0] in target:
                    correct_prediction_count += 1
        else:
            for p in prediction:
                if p in target:
                    correct_prediction_count += 1
    return correct_prediction_count / (2 * len(test_target))


def both_emoji_accuracy(test_target, test_prediction):
    correct_prediction_count = 0
    for target, prediction in zip(test_target, test_prediction):
        if prediction[0] == prediction[1]:
            if target[0] == target[1]:
                if prediction[0] in target:
                    correct_prediction_count += 2
    return correct_prediction_count / (2 * len(test_target))
