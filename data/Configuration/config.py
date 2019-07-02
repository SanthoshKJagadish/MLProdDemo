import pickle

model = None

def load_model():

    global model
    # model variable refers to the global variable
    with open("/Users/santhoshkumarjagadish/Documents/Github/MLProdDemo/data/Configuration/iris_trained_model.pkl", 'rb') as f:
        model = pickle.load(f)
