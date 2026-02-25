from src.components.model_trainer import load_data, prepare_data, split_data, train_model, evaluate_model
from src.utils import save_model
from src.logger import logging
if __name__ == "__main__":
    logging.info("Starting model training process")

    df = load_data("data/Admission_Predict.csv")
    X, y = prepare_data(df, target_column='Chance of Admit ')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, model_type='random_forest')
    
    r2, mae = evaluate_model(model, X_test, y_test)
    
    save_model(model, "artifacts/model.pkl")
    logging.info(f"Model training process completed successfully with R2: {r2:.4f} and MAE: {mae:.4f}")