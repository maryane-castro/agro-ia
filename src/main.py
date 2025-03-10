from src.utils.extract_png_csv import master_extract
from src.utils.get_best_frames import master_best_frame
from src.utils.extract_features_animals import master_features
from models.class_regression import ModelPredictorRegression

# extract svos
master_extract(svo_directory="data/svos", 
               output_base_dir="data/processed/extractions_svos")


# get best frame
master_best_frame(inputs_png="data/processed/extractions_svos/pngs",
                  inputs_csv="data/processed/extractions_svos/csvs",
                  dynamic_margin=5 # valor padrão
                  )


# extract features
master_features(inputs_paste_pngs="data/processed/melhoresFrames/pngs",
                inputs_paste_csvs="data/processed/melhoresFrames/csvs")



# model regression
predictor = ModelPredictorRegression('models/weights/best_xgboost_model.pkl', 'models/weights/ensemble_model.pkl')
predictor.predict(file_path='data/processed/datasets/dataset.csv', 
                  output_path='data/previsoes_modelos.csv') 

