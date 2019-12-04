import pandas as pd
import numpy as np
import nltk
import rltk

from collections import Counter
import math



def eidtdistanceCal(trainResults,test):
    results = []
    for i, trainResult in enumerate(trainResults) :
        score = nltk.edit_distance(trainResult, test)
        results.append((score,i))
    return results

def hummingdistanceCal(trainResults,test):
    results = []
    for i, trainResult in enumerate(trainResults) :
        score = rltk.dice_similarity(set(trainResult), set(test))
        results.append((score,i))
    return results


def cosineSimilarityCal(trainResults,test):
    results = []
    for i, trainResult in enumerate(trainResults) :
        score = rltk.dice_similarity(set(trainResult), set(test))
        results.append((score,i))
    return results

def accscore(test_tags,predicted):
    count=0
    for tag, pred in zip(test_tags, predicted):
        if pred[0] == pred[1]:
            if tag[0] == tag[1]:
                if pred[0] in tag:
                    	count+=2
            else:
                if pred[0] in tag:
                    	count+=1
        else:
            for p in pred:
                if p in tag:
                    count+=1               
    return count/(2*len(test_tags))

def tagsplit(tmp_tags):
    result_tags=[]
    for tag in tmp_tags:

        tmp_tag = tag[0].split()
        result_tags.append((tmp_tag[0],tmp_tag[1]))
    return result_tags



def knn(data, query, k):
    tokenizedResult =[]
    print('testing ', query)
    # 3. For each example in the data
    for ele in data:
        tokenizedResult.append(nltk.word_tokenize(ele))
    testex=nltk.word_tokenize(query)
    neighbor_distances_and_indices = hummingdistanceCal(tokenizedResult,testex)
    neighbor_distances_and_indices2 = cosineSimilarityCal(tokenizedResult,testex)
    combineScore = []
    for c,d in zip(neighbor_distances_and_indices,neighbor_distances_and_indices2):
        sum_squared_distance = math.pow(c[0], 2) + math.pow(d[0], 2)
        combineScore.append((math.sqrt(sum_squared_distance),c[1]))
    
    #fromedResult = transform(tokenizedResult, query)
    # 3.1 Calculate the distance between the query example and the current
    # example from the data.
    #neighbor_distances_and_indices = eucldistanceCal(fromedResult)

    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(combineScore)
    sorted_neighbor_distances_and_indices.sort(reverse = True)

    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    

    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices, k_nearest_labels

def readfile(target):
    with open(target, 'r', encoding='utf-8') as f:
       lines = f.read().splitlines()
    return lines
def simplylist(target):
    result=[]
    for t in target:
        tmp = t[0]
        result.append(tmp)
    return result


def main():
    train_reviews = pd.read_csv("data/t2e_train.text",header = None)
    train_tags = pd.read_csv("data/t2e_train.emoji",header = None)
    test_reviews = pd.read_csv("data/t2e_test.text",header = None)
    test_tags = pd.read_csv("data/t2e_test.emoji",header = None)
    train_reviews = train_reviews.to_numpy()
    # Testing set (what we will use to test the trained model)
    tmp_train_tags = train_tags.to_numpy()
    tmp_test_tags = test_tags.to_numpy()
    train_tags = tagsplit(tmp_train_tags)
    test_tags = tagsplit(tmp_test_tags)
    test_reviews = test_reviews.to_numpy()
    
    train_reviews=simplylist(train_reviews)
    test_reviews=simplylist(test_reviews)

    emojiResult=[]
    for test_review in test_reviews:
        reg_query = test_review
        reg_k_nearest_neighbors, reg_prediction = knn(
            train_reviews, reg_query, k=2
        )
        emojiResult.append((train_tags[reg_k_nearest_neighbors[0][1]][0],train_tags[reg_k_nearest_neighbors[1][1]][0]))

    print(accscore(test_tags,emojiResult))
 

if __name__ == '__main__':

    main()
