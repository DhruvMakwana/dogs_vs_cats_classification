#importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
	help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[Info] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3)
model.fit(db["features"][:i], db["labels"][:i])
print("[Info] best hyperparameters: {}".
	format(model.best_params_))

#evaluating
prit("[Info] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, 
	target_names=db["label_names"]))

#accuracy
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

#serialising
print("[Info] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()