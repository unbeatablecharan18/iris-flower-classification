import joblib, os, json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = os.path.join(os.getcwd(), 'models')
FEATURES = ['sepal_length','sepal_width','petal_length','petal_width']

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = FEATURES + ['species']
label_mapping = {name: idx for idx,name in enumerate(sorted(df['species'].unique()))}
X = df[FEATURES]; y = df['species'].map(label_mapping)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
results = {}
for name,file in [('Logistic Regression','logistic_regression_pipeline.pkl'),('Decision Tree','decision_tree_pipeline.pkl'),('Random Forest','random_forest_pipeline.pkl'),('Best Model','best_model_pipeline.pkl')]:
    path=os.path.join(MODEL_DIR,file)
    if not os.path.exists(path):
        print(f"MISSING:{file}")
        continue
    p=joblib.load(path)
    y_pred = p.predict(X_test)
    acc = float(accuracy_score(y_test,y_pred))
    prec,recall,f1,_ = precision_recall_fscore_support(y_test,y_pred,average='macro',zero_division=0)
    cm = confusion_matrix(y_test,y_pred).tolist()
    results[name] = {'accuracy':acc,'precision':float(prec),'recall':float(recall),'f1':float(f1),'confusion_matrix':cm}
print(json.dumps(results, indent=2))
