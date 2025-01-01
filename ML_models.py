
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from  data.process_data import get_data_ML
import pandas as pd 
import numpy as np
from utils import get_perfrom

list_name = ['CHURCHMAN BIBLE', 'LUKE G U', 'CROSS H CATTLE', 'Recruit F9', 'NOLAN', 'NEWBY', 'SHRIMPLIN', 'SHANKLE']
for name in list_name:
    X, y = get_data_ML(name)
    X_train = X['train']
    X_test  = X['test'] 
    y_train = y['train']
    y_test  = y['test']
    
    SVM_model = svm.SVC(C=10, gamma=1,  probability=True)
    SVM_model.fit(X_train,y_train)
    yhat_SVM = SVM_model.predict(X_test)
    y_prob_SVM = SVM_model.predict_proba(X_test)
   
    GPC_model = GaussianProcessClassifier().fit(X_train, y_train)
    yhat_GPC = GPC_model.predict(X_test)
    y_prob_GPC = GPC_model.predict_proba(X_test)

    RFC_model = RandomForestClassifier(max_depth=12, n_estimators=20, max_features=6).fit(X_train, y_train)
    yhat_RFC = RFC_model.predict(X_test)
    y_prob_RFC = RFC_model.predict_proba(X_test)  

    NNC_model = MLPClassifier(alpha=0.001, max_iter=1000, learning_rate_init=0.001, 
                            solver='adam', batch_size=10, hidden_layer_sizes=200 ).fit(X_train, y_train)

    yhat_NNC = NNC_model.predict(X_test)
    y_prob_NNC = NNC_model.predict_proba(X_test)
    Ks =15
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))
    for n in range(1,Ks):
        KNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat = KNN_model.predict(X_test)
        
        mean_acc[n-1]= np.mean(yhat==y_test)
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

    k = 5
    KNN_model = KNeighborsClassifier(n_neighbors=k ,leaf_size=50, p=1,  weights='distance' ).fit(X_train, y_train)
    KNN_model
    yhat_KNN = KNN_model.predict(X_test)
    y_prob_KNN = KNN_model.predict_proba(X_test)
   
    DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 25, min_samples_split=3 )
    DT_model.fit(X_train,y_train)
    yhat_DT = DT_model.predict(X_test)
    y_prob_DT = DT_model.predict_proba(X_test)
    

    model_name = ['SVM', 'GPC', 'RFC', 'NNC', 'KNN', 'DT']
    list_predict = [yhat_SVM, yhat_GPC, yhat_RFC, yhat_NNC, yhat_KNN, yhat_DT]
    list_probs = [y_prob_SVM, y_prob_GPC, y_prob_RFC, y_prob_NNC, y_prob_KNN, y_prob_DT]
    roc_auc, pr_auc, accuracy, balanced_ACC, mcc, precision, recall, f1 = [], [], [], [], [], [], [], []
    for predict, prob in zip(list_predict, list_probs):
        test_perform = get_perfrom(y_test, predict, prob)
        roc_auc.append(test_perform[0])
        pr_auc.append(test_perform[1])
        accuracy.append(test_perform[2])
        balanced_ACC.append(test_perform[3])
        mcc.append(test_perform[4])
        precision.append(test_perform[5])
        recall.append(test_perform[6])
        f1.append(test_perform[7])

    
    # data = pd.DataFrame()
    # data['Model'] = model_name
    # data['AUC'] = roc_auc
    # data['PR_AUC'] = pr_auc
    # data['ACC'] = accuracy
    # data['BA_ACC'] = balanced_ACC
    # data['MCC'] = mcc
    # data['F1'] = f1
    # data['recall'] = recall
    # data['precision'] = precision
    # data.to_csv(f'./result_ML_{name}.csv', index=False)  
