# %%
#======Libraries======
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn

# %%
#======Function Definitions======
def process_text(column, dataFrame = pd.DataFrame()): #Text Processing Function
    #Variables
    text_data = [] #Empty list
    stop_words = set(stopwords.words("english")) #Set of english stopwords from the Natural Language Toolkit (nltk)
    special_characters = set(string.punctuation) #Set of special characters
    unfiltered_set = stop_words | special_characters #Union set of stopwords and special characters

    if dataFrame.empty:
        # text_data = [" ".join([word for word in word_tokenize(text) if not word in stop_words])]
        filtered_words = []
        words = word_tokenize(column)
        for word in words: #Loops through each word
            if not word in unfiltered_set:
                filtered_words.append(word.lower())
        filtered_text = " ".join(filtered_words)
        text_data.append(filtered_text)
    else:
        # dataFrame[column] = dataFrame[column].apply(lambda x: " ".join([word for word in word_tokenize(x) if not word in stop_words]))
        for row in dataFrame[column]: #Loops through each row
            filtered_words = []
            words = word_tokenize(row)
            for word in words: #Loops through each word
                if not word in unfiltered_set:
                    filtered_words.append(word.lower())
            filtered_text = " ".join(filtered_words)
            text_data.append(filtered_text)

    return text_data

def summarize_dataframe(dataFrame, column): #Summarize Dataframe Function
    counts = dataFrame[column].value_counts() #Counts the number of occurances
    num_categories = len(counts) #Counts the possible occurances
    print(f"Number of categories: {num_categories}")
    print(counts.rename({0: 'Not Spam', 1: 'Spam'}))
    #Plots the data
    plt.figure(figsize = (4, 4))
    sns.countplot(data = dataFrame, x = "Label")

def evaluate(cVect, lr): #Evaluate Function
    text = input("Enter Text(Subject of the mail): ")
    text = process_text(text)

    t_dtv = cVect.transform(text).toarray()
    prob = lr.predict_proba(t_dtv) * 100

    print("Predicted Class: ", end = "")
    print("Spam" if lr.predict(t_dtv)[0] else "Not Spam")

    print(f"Not Spam: {prob[0][0]}%")
    print(f"Spam: {prob[0][1]}%")

    #Plots probabilities
    plt.figure(figsize = (4, 4))
    sns.barplot(x = ["Not Spam", "Spam"] , y = [prob[0][0], prob[0][1]])
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()
    
# %%
#======Main======
def main():
    #Reads in data
    df = pd.read_csv("spam_ham_dataset.csv")
    df.drop("Unnamed: 0", axis = 1, inplace = True) #Removes unnecessary column
    df.columns = ("Label", "Text", "Class") #Changes column names

    #Processes the data
    df["Text"] = process_text("Text", df)

    #Summarizes the data
    summarize_dataframe(df, "Class")

    #Splits data into train and test in 80:20
    X = df.loc[:, "Text"]
    y = df.loc[:, "Class"]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 11)

    #Preprocess text to build the ML model
    cVect = sklearn.feature_extraction.text.CountVectorizer()
    cVect.fit(X_train)
    dtv = cVect.transform(X_train).toarray() #Training document term vector

    #Hyperparameter tuning
    lr = sklearn.linear_model.LogisticRegression() #Logistic regression

    grid = {"C": [float(i) for i in range(1, 3)], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]} #???
    logreg_cv = sklearn.model_selection.GridSearchCV(lr, grid, cv = 4) #???
    logreg_cv.fit(dtv, y_train)

    lr = sklearn.linear_model.LogisticRegression(solver = "liblinear", penalty = "l2" , C = 1.0)
    lr.fit(dtv, y_train)

    #Validate on the test data
    test_dtv = cVect.transform(X_test).toarray() #Testing document term vector
    pred = lr.predict(test_dtv)

    #Visualizing model performance
    print(sklearn.metrics.classification_report(y_test, pred)) #Classification report

    #Confusion matrix
    cmat = sklearn.metrics.confusion_matrix(y_test, pred)
    plt.figure(figsize = (4, 4))
    sns.heatmap(cmat, annot = True, cmap = "Paired", cbar = False, fmt = "d", xticklabels = ["Not Spam", "Spam"], yticklabels = ["Not Spam", "Spam"])
    
    #Evaluation
    evaluate(cVect, lr)

# %%
#======Execution Check======
if __name__ == "__main__":
    sns.set_style("whitegrid") #Sets background style for graphs
    nltk.download("stopwords") #Downloads a list of stop words if none is found
    nltk.download("punkt") #Downloads a list of language tokens if none is found
    main()
    # quit() #Exits program
# %%
