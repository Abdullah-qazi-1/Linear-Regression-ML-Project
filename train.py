import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from utils import save_model

def load_data(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df

def prepare_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    
    df = load_data("data/Admission_Predict.csv")
    X, y = prepare_data(df, target_column='Chance of Admit ')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, model_type='random_forest')
    
    y_pred = model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    
    save_model(model, "artifacts/model.pkl")
    print("Training complete!")