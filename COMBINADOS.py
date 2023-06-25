import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/hhho.csv')
    dt_features = dt_heart.drop(['%Toxicos'], axis=1)
    dt_target = dt_heart['%Toxicos']

    imputer = SimpleImputer(strategy='mean')
    dt_features = imputer.fit_transform(dt_features)
    dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    kernel = ['linear', 'poly', 'rbf']

    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel=k)
        kpca.fit(X_train)

        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)

        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(dt_train, y_train)

        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/hhho.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Imputar los valores faltantes con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Resultados con los datos originales
    boosting = GradientBoostingClassifier(loss='deviance', learning_rate=0.15, n_estimators=188, max_depth=5).fit(X_train_imputed, y_train)
    boosting_pred = boosting.predict(X_test_imputed)
    print('=' * 64)
    print('Datos Originales')
    print(accuracy_score(boosting_pred, y_test))

    # Resultados con los datos normalizados
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_imputed)
    X_test_normalized = scaler.transform(X_test_imputed)

    boosting_normalized = GradientBoostingClassifier(n_estimators=188).fit(X_train_normalized, y_train)
    boosting_pred_normalized = boosting_normalized.predict(X_test_normalized)
    print('=' * 64)
    print('Datos Normalizados')
    print(accuracy_score(boosting_pred_normalized, y_test))

    # Resultados con los datos normalizados y discretizados
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_normalized)
    X_test_discretized = discretizer.transform(X_test_normalized)

    boosting_discretized = GradientBoostingClassifier(n_estimators=188).fit(X_train_discretized, y_train)
    boosting_pred_discretized = boosting_discretized.predict(X_test_discretized)
    print('=' * 64)
    print('Datos Normalizados y Discretizados')
    print(accuracy_score(boosting_pred_discretized, y_test))
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/hhho.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Eliminar las muestras con valores faltantes
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]

    # Resultados con los datos originales
    estimators = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)
    }

    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('=' * 64)
        print('Datos Originales')
        print('SCORE Bagging with {}: {}'.format(name, accuracy_score(bag_predict, y_test)))

    # Resultados con los datos normalizados y discretizados
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_normalized)
    X_test_discretized = discretizer.transform(X_test_normalized)

    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train_discretized, y_train)
        bag_predict = bag_class.predict(X_test_discretized)
        print('=' * 64)
        print('Datos Normalizados y Discretizados')
        print('SCORE Bagging with {}: {}'.format(name, accuracy_score(bag_predict, y_test)))
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    dataset = pd.read_csv('./datos/hhho.csv')
    print(dataset.head(5))
    
    X = dataset.drop(['%Toxicos'], axis=1)
    y = dataset[['%Toxicos']]
    
    # Codificación ordinal de las etiquetas de clase
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }
    
    warnings.simplefilter("ignore")
    
    for name, estimator in estimadores.items():
        # Entrenamiento
        estimator.fit(X_train, y_train)
        
        # Predicciones del conjunto de prueba
        predictions = estimator.predict(X_test)
        
        print("=" * 64)
        print(name)
        
        # Medimos el error, datos de prueba y predicciones
        print("MSE: " + "%.10f" % float(mean_squared_error(y_test, predictions)))
        
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions, 'r--')
        plt.show()

