import sys

sys.path.insert(1, '../')
import metric_calculator

if __name__ == '__main__':
    prediction_path = '../tmp/test.output'  # output of NMT
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

    print('NMT Accuracy: %s' % metric_calculator.accuracy(targets, predictions))
