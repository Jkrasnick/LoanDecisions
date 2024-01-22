# This is a sample Python script.
import pandas
import matplotlib.pyplot as plt
from sklearn import tree, __all__, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def get_bp(x):
    if x == ("Y"):
        return 1
    else:
        return 0
def getIncome(x):
    if (x > 7000):
        return 1
    elif (x>6000):
        return 0.9
    elif (x>5000):
        return 0.8
    elif (x>4000):
        return 0.5
    elif (x>3000):
        return 0.2
    else:
        return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pandas.read_csv("loan-train.csv")
    df = df.dropna()
    education = {'Graduate':1, 'Not Graduate':0}
    employed = {'Yes': 1, 'No': 0}
    credit = {1: 1, 0: 0}
    df['Self_Employed'] = df['Self_Employed'].map(employed)
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map(education)
    df['IncomeLoanRatio'] = df['ApplicantIncome'] / df['LoanAmount']
    df['Credit_History'] = df['Credit_History'].map(credit)
    df['Loan_Status'] = df['Loan_Status'].map(get_bp)
    #features = ['Self_Employed', 'Education', 'Credit_History', 'ApplicantIncome', 'Married']
    features = ['IncomeLoanRatio']
    X = df[features]
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    dtree = DecisionTreeClassifier(max_depth=4)
    dtree.fit(X, y)
    plt.figure(figsize=(15, 10))
    tree.plot_tree(dtree, feature_names=features, filled=True)
    y_pred = dtree.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    plt.show()


    # Model Accuracy, how often is the classifier correct?


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
