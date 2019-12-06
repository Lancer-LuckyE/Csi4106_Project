import sys

sys.path.insert(1, '../')
import metric_calculator


def accuracy(prediction_path):
    # prediction_path = '../tmp/test_30kdic.output'  # output of NMT
    target_path = '../data/t2e_test.emoji'  # gold standard
    test_size = sum(1 for _ in open(target_path))

    predictions = []
    targets = []

    with open(prediction_path, 'r') as p, open(target_path, 'r') as t:
        counter = 0

        while counter < test_size:
            prediction = p.readline().strip()
            target = t.readline().strip()

            prediction = prediction.split(' ')
            target = target.split(' ')

            if prediction == ['']:  # no prediction output
                prediction = ['DUMMY', 'DUMMY']  # give it dummy values
            elif len(prediction) == 1:  # only one prediction
                prediction.append('DUMMY')  # append a dummy value

            predictions.append(prediction)
            targets.append(target)

            counter += 1

    print('NMT Accuracy: %s' % metric_calculator.overall_accuracy(targets, predictions))


def count_empty_output(prediction_path):
    empty, total = 0, 0
    with open(prediction_path, 'r') as p:
        for line in p:
            total += 1
            if line.strip() == '':
                empty += 1
    return empty, total


if __name__ == '__main__':
    path = '../tmp/test_30kdic.output'
    print('Empty: %s, total: %s' % count_empty_output(path))
    accuracy(path)
    path = '../tmp/test.output'
    print('Empty: %s, total: %s' % count_empty_output(path))
    accuracy(path)
