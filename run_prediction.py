import pandas as pd
from get_features import get_features
from run_training import start_training
TEST_PATH = "test2.csv"

def start_prediction(model):
    df = pd.read_csv(TEST_PATH)
    print(df)
    features = get_features(df)
    print("Test features len")
    print(len(features))
    predictions = model.predict(features)
    
    with open("submission.csv",'w') as f:
        f.write("id,target\n")
        for i in range(0,len(predictions)):
            f.write(str(df.at[i,'id'])+","+str(predictions[i])+"\n")


if __name__ == "__main__":
    model = start_training()
    print("training done")
    start_prediction(model)