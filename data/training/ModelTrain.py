from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle



# Load Iris dataset
iris_dict = load_iris()
X = iris_dict['data']
y =iris_dict['target']

# Shuffle
X_new, y_new = shuffle(X, y, random_state=0)

# Sample between Train and Test
n_samples_train = 120 # number of samples for training (--> #samples for testing = len(y_new) - 120 = 30)
X_train = X_new[:n_samples_train, :]
y_train = y_new[:n_samples_train]

X_test = X_new[n_samples_train:, :]
y_test = y_new[n_samples_train:]

# Apply classification using logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Measure accuracy on test set of train data
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Creating a pickle file for serialization
with open("/Users/santhoshkumarjagadish/Documents/Github/MLProdDemo/data/Configuration/iris_trained_model.pkl", 'wb') as f:
    pickle.dump(clf, f)

#with open("/Users/santhoshkumarjagadish/Documents/Github/MLProdDemo/data/Configuration/iris_trained_model.pkl", 'rb') as f:
    #clf_loaded = pickle.load(f)
