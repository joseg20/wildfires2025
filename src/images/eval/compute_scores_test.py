import os
import numpy as np
import pandas as pd
import glob
import argparse
from compute_perf import (
    evaluate_predictions,
    evaluate_multiple_pred_folders,
)

def load_conf_values(conf_file):
    # Cargar el archivo CSV en un DataFrame
    conf_df = pd.read_csv(conf_file)
    
    # Convertir el DataFrame en un diccionario con el nombre del modelo como clave y el conf como valor
    conf_dict = dict(zip(conf_df['model_name'], conf_df['conf']))
    
    return conf_dict

def main(args):
    gt_folder = args.gt_folder  # Ruta al folder de ground truth
    pred_folder = args.pred_folder  # Ruta al folder de predicciones
    conf_dict = load_conf_values(args.conf_file)  # Cargar el diccionario de conf

    # Verificar que existan archivos en la ruta de predicciones
    prediction_files = glob.glob(os.path.join(pred_folder, "*"))
    pred_folder = [os.path.join(dir, 'labels') for dir in prediction_files]
    
    results = []
    for pred_foldera in pred_folder:
        # Extraer el nombre del modelo desde la ruta
        # model_name = os.path.basename(os.path.dirname(os.path.dirname(pred_foldera)))
        model_name = pred_foldera.split('/')[-2]
        if model_name == "results.csv":
            # no continuar
            continue
        conf_thres = conf_dict.get(model_name)
        # Obtener el conf correspondiente para el modelo
        # si conf_thres is None:
        if conf_thres is None:
            # Si no se encuentra el conf en el diccionario, no evaluar el modelo
            print(f"Model: {model_name} not found in the confidence dictionary. Skipping evaluation.")
            conf_thres = 0.05
        print(f"Model: {model_name}, Confidence Threshold: {conf_thres}")

        metrics = evaluate_predictions(pred_foldera, gt_folder, conf_th=conf_thres)
        results.append({
            "Prediction Folder": "test",
            "Model Name": model_name,
            "Confidence Threshold": conf_thres,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"]
        })

    # Convertir los resultados a un DataFrame para su fácil visualización
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False)
    # save results to a csv file in pred_folder
    results_df.to_csv(os.path.join(args.pred_folder, "results.csv"), index=False)

    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction folders with specific confidence thresholds.")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to the ground truth labels folder.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Path to the prediction labels folder.")
    parser.add_argument("--conf_file", type=str, required=True, help="Path to the CSV file containing confidence thresholds.")
    args = parser.parse_args()
    main(args)