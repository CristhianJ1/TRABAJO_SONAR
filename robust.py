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
    dataset = pd.read_csv('./datos/FUSION.csv')
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


