import os
import numpy as np
import pandas as pd
import glob
import argparse
from compute_perf import (
    evaluate_predictions,
    evaluate_multiple_pred_folders,
    find_best_conf_threshold_and_plot,
)

def main(args):
    gt_folder = args.gt_folder  # Ruta al folder de ground truth
    pred_folder = args.pred_folder  # Ruta al folder de predicciones

    # Define a range of possible confidence threshold values
    conf_thres_range = np.linspace(0.01, 0.5, 1)
    
    # Verificar que existan archivos en la ruta de predicciones
    prediction_files = glob.glob(os.path.join(pred_folder, "*"))
    pred_folder = [os.path.join(dir, 'labels') for dir in prediction_files]
    
    
    
    # Obtener resultados de la evaluación de predicciones
    results_df = evaluate_multiple_pred_folders(pred_folder, gt_folder, conf_thres_range)
    results_df = results_df.sort_values(by="Best F1 Score", ascending=False)

    print(results_df)

    # Obtener los mejores valores utilizando la función proporcionada
    for pred_foldera in pred_folder:
        best_conf_thres, best_f1_score, best_precision, best_recall = (
            find_best_conf_threshold_and_plot(pred_foldera, gt_folder, conf_thres_range, True)
        )

        print(
            f"Best Confidence Threshold: {best_conf_thres}\nBest F1 Score: {best_f1_score}\nPrecision: {best_precision}\nRecall: {best_recall}"
        )
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction folders and find the best confidence threshold.")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to the ground truth labels folder.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Path to the prediction labels folder.")
    
    args = parser.parse_args()
    
    main(args)
