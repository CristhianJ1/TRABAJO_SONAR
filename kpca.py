import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    dt_heart = pd.read_csv('./datos/FUSION.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['%Toxicos'], axis=1)
    dt_target = dt_heart['%Toxicos']

    imputer = SimpleImputer(strategy='mean')
    dt_features = imputer.fit_transform(dt_features)

    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    kernel = ['linear', 'poly', 'rbf']
    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel=k)
        kpca.fit(X_train)

        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)

        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(dt_train, y_train)

        print("SCORE KPCA " + k + ":", logistic.score(dt_test, y_test))
