# %%
#======Libraries======
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn

#======Main======
def main():
    #Reads data
    df = pd.read_csv("spam_ham_dataset.csv")
    df.drop("Unnamed: 0", axis = 1, inplace = True) #Removes unnecessary column
    df.columns = ("Label", "Text", "Class") #Changes column names

    #Plots the number of ham and spam emails
    plt.figure(figsize = (12, 6))
    sns.countplot(data = df, x = "Label")

    #Filters out stop words from the data
    stop_words = set(stopwords.words("english"))
    df["Text"] = df["Text"].apply(lambda x: " ".join([word for word in word_tokenize(x) if not word in stop_words]))

    #Splits data into train and test in 80:20
    X = df.loc[:, "Text"]
    y = df.loc[:, "Class"]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 11)

    #Preprocess text to build the ML model
    cVect = sklearn.feature_extraction.text.CountVectorizer()
    cVect.fit(X_train)
    dtv = cVect.transform(X_train).toarray() #Training document term vector

    #Hyperparameter tuning
    lr = sklearn.linear_model.LogisticRegression(verbose = 1) #Logistic regression

    grid = {"C": [float(i) for i in range(1, 3)], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]} #???
    logreg_cv = sklearn.model_selection.GridSearchCV(lr, grid, cv = 4) #???
    logreg_cv.fit(dtv, y_train)

    lr = sklearn.linear_model.LogisticRegression(solver = "liblinear", penalty = "l2" , C = 1.0)
    lr.fit(dtv, y_train)

    #Evaluate on the test data
    test_dtv = cVect.transform(X_test).toarray() #Testing document term vector
    pred = lr.predict(test_dtv)

    #Visualizing model performance
    print(sklearn.metrics.classification_report(y_test, pred)) #Classification report

    #Confusion matrix
    cmat = sklearn.metrics.confusion_matrix(y_test, pred)
    plt.figure(figsize = (6, 6))
    sns.heatmap(cmat, annot = True, cmap = "Paired", cbar = False, fmt = "d", xticklabels = ["Not Spam", "Spam"], yticklabels = ["Not Spam", "Spam"])
    
    #Evaluation
    text = input("Enter Text(Subject of the mail): ")
    text = [" ".join([word for word in word_tokenize(text) if not word in stop_words])]

    t_dtv = cVect.transform(text).toarray()
    prob = lr.predict_proba(t_dtv) * 100

    print("Predicted Class: ", end = "")
    print("Spam" if lr.predict(t_dtv)[0] else "Not Spam")

    print(f"Not Spam: {prob[0][0]}%")
    print(f"Spam: {prob[0][1]}%")

    #Plots probabilities
    plt.figure(figsize = (12, 6))
    sns.barplot(x = ["Not Spam", "Spam"] , y = [prob[0][0], prob[0][1]])
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()

#======Execution Check======
if __name__ == "__main__":
    sns.set_style("whitegrid") #Sets background style for graphs
    nltk.download("stopwords") #Downloads a list of stop words if none is found
    nltk.download("punkt") #Downloads a list of language tokens if none is found
    main()
    # quit() #Exits program
# %%
