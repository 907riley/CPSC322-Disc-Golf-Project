import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier

def test_random_forest_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    interview_dataset = []
    # first need to recombine into one list
    for i in range(len(X_train_interview)):
        temp = X_train_interview[i]
        temp.append(y_train_interview[i])
        interview_dataset.append(temp)

    forest = MyRandomForestClassifier(20, 7, 2)
    forest.fit(interview_dataset, 4)

    # since it's random the most testing we can really do is to look
    # at the size of the self.forest var
    # we can also just look at the different trees to see
    # if its really working


    assert len(forest.forest) == 20
    assert len(forest.best_trees) == 7



def test_random_forest_predict():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    interview_dataset = []
    # first need to recombine into one list
    for i in range(len(X_train_interview)):
        temp = X_train_interview[i]
        temp.append(y_train_interview[i])
        interview_dataset.append(temp)

    forest = MyRandomForestClassifier(20, 7, 2)
    forest.fit(interview_dataset, 4)

    # using test instances from previous tests
    test_instances_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    interview_expected = ['True', 'False']
    interview_predicted = forest.predict(test_instances_interview)
    print(interview_predicted)

    assert len(interview_predicted) == 2