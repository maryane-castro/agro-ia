import joblib
import numpy as np
import pandas as pd

class ModelPredictorRegression:
    def __init__(self, best_model_path, ensemble_model_path):
        self.best_model = joblib.load(best_model_path)
        self.ensemble = joblib.load(ensemble_model_path)

    def preprocess_data(self, file_path):
        X_new = pd.read_csv(file_path)
        
        # Remover colunas desnecessárias
        columns_to_remove = ['Animal', 'ID', 'Frame', 'Real']
        X_new = X_new.drop(columns=columns_to_remove, axis=1, errors='ignore')
        
        # Criar novas features
        X_new['Height_Ratio'] = X_new['Height_Centroid'] / X_new['Height_average']
        X_new['Width_Height_Ratio'] = X_new['Width'] / X_new['Height_average']
        X_new['Length_Height_Ratio'] = X_new['Length'] / X_new['Height_average']
        X_new['Volume_Area_Ratio'] = X_new['Volume'] / (X_new['Width'] * X_new['Length'])
        X_new['Normalized_Width_Height'] = (X_new['Width'] * X_new['Height_average']) / X_new['Length']
        X_new['Length_Width_Ratio'] = X_new['Length'] / X_new['Width']
        X_new['Sin_Width'] = np.sin(X_new['Width'])
        X_new['Cos_Length'] = np.cos(X_new['Length'])
        
        return X_new
    
    def predict(self, file_path, output_path):
        X_new = self.preprocess_data(file_path)
        
        y_pred_best_model = self.best_model.predict(X_new)
        y_pred_ensemble = self.ensemble.predict(X_new)
        
        df_pred = pd.DataFrame({
            'Previsão_Best_Model': y_pred_best_model,
            'Previsão_Ensemble': y_pred_ensemble
        })
        
        df_pred.to_csv(output_path, index=False)
        print(f"Previsões salvas em '{output_path}'.")
