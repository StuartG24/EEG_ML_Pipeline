# ML Utility Functions
# ml_utilities.py
# v1.0 3/8/25

# Imports
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from IPython.display import display
import pandas as pd
import numpy as np
import shap

# Inspect ML Model Parameters
#
def ml_model_pipeline_details(model_pipeline):
    print("Model Training Pipeline Steps:")
    for name, step in model_pipeline.named_steps.items():
        print(f"- {name}: {step}")

    print('All Pipeline Parameters:')
    for param, value in model_pipeline.get_params().items():
        print(f"- {param}: {value}")

# Display Grid Search Results
#
def grid_search_results(search, duration):
    print('Grid Search Results')
    all_search_results = pd.DataFrame(search.cv_results_)
    print(f"Score: {search.best_score_:.4f}. Mean: {np.mean(all_search_results['mean_test_score']):.4f} and STD {np.std(all_search_results['mean_test_score']):.4f}")
    print(f'Search Took: {duration:.2f} seconds')
    print(f"Best Parameters: {search.best_params_}")
    print(f'Best C-V Score: {search.best_score_}')
    
    top_n = 10
    print(f"Top {top_n} out of {len(all_search_results)} combinations:")
    display(all_search_results[['rank_test_score', 'mean_test_score', 'mean_fit_time', 'mean_score_time', 'params']].sort_values(by='rank_test_score').head(top_n))


# Inspect the Evaluation Metrics for a Classification Model
#
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

def classification_metrics(for_Model, X_test, y_test, y_pred):
    plt.style.use('default')

    print('Classification Results')

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mcc = matthews_corrcoef(y_test, y_pred)

    # Print various metrics
    print('-----------')
    print(f'Recall (Sensitivity, TP Rate): {metrics.recall_score(y_true=y_test, y_pred=y_pred, pos_label=1):.4f}')
    print(f'Precision: {metrics.precision_score(y_true=y_test, y_pred=y_pred, pos_label=1):.4f}')
    print(f'F1 Score {metrics.f1_score(y_true=y_test, y_pred=y_pred, pos_label=1):.4f}')
    print(f'Specificity: {tn / (tn + fp):.4f}')
    print(f'MCC: {mcc:.4f}')
    print('-----------')
    print(f'Accuracy: {metrics.accuracy_score(y_true=y_test, y_pred=y_pred):.4f}')
    print(f'Fall Out (FPR): {fp / (fp + tn):.4f}')
    print(f'Hamming Loss {metrics.hamming_loss(y_true=y_test, y_pred=y_pred):.4f}')

    y_probabilities = for_Model.predict_proba(X_test)[:, 1]
    roc_auc_score = metrics.roc_auc_score(y_true=y_test, y_score=y_probabilities)
    print(f'ROC-AUC Score {roc_auc_score:.4f}')
    gini_score = 2 * roc_auc_score - 1
    print(f'Gini Index: {gini_score:.4f}')

    # Plot Confusion Matrix & ROC Curve
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    fig.suptitle(f'Model Prediction Results', fontsize=20)

    axes[0].set_title('Confusion Matrix')    
    class_labels = for_Model.classes_
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(ax=axes[0])

    axes[1].set_title('ROC Curve')
    roc_display = RocCurveDisplay.from_estimator(for_Model, X_test, y_test, ax=axes[1], pos_label=1)

    plt.tight_layout()
    plt.show()
    plt.style.use('ggplot')


    # class_labels = for_Model.classes_
    # fig, ax = plt.subplots(figsize=(12,4))
    # ax.set_title('Confusion Matrix')
    # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(ax=ax)
    # plt.show

    # y_probabilities = for_Model.predict_proba(X_test)[:, 1]
    # roc_auc_score = metrics.roc_auc_score(y_true=y_test, y_score=y_probabilities)
    # print(f'ROC-AUC Score {roc_auc_score:.4f}')
    # gini_score = 2 * roc_auc_score - 1
    # print(f'Gini Index: {gini_score:.4f}')

    # # Plot the ROC curve
    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.set_title('ROC Curve')
    # roc_display = RocCurveDisplay.from_estimator(for_Model, X_test, y_test, ax=ax, pos_label=1)
    # plt.show()

    # plt.style.use('ggplot')

# Collate & Print Probabilities for a Prediction
#
def get_prediction_probabilities(y_pred, y_probs, y_true):
    predicted_prob = [y_probs[i, pred] for i, pred in enumerate(y_pred)]
    results_df = pd.DataFrame({
    'Prediction': y_pred,
    'Predicted_Probability': predicted_prob,
    'True': y_true
    })
    print(results_df.shape)
    display(results_df.head(10))

    print('Mismatches')
    mismatches = results_df[results_df['Prediction'] != results_df['True']]
    display(mismatches)

    return results_df

# Get Feature Importance for a RandomForest
#
def feature_importance(model_rf, data_pipeline):
    # Get feature importances
    importances = model_rf.named_steps['classifier'].feature_importances_
    feature_names = data_pipeline.named_steps['data_preprocess'].get_feature_names_out()


    # Map feature importances to transformed feature names
    # transformed_feature_names = cols_transform.get_feature_names_out()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    
    # Importance
    print('Importance')
    print(importance_df.shape)
    display(importance_df.head())
    
    # Keep only the top 25 most important features & plot
    importance_df = importance_df.head(25)

    # Plot the feature importances with names horizontally
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances (Sorted)')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()

# Get Feature Importance Using SHAP Values
#

def get_shap_importance(X_train, X_test, model, data_pipeline):

    def model_predict(X):
        return model.predict_proba(X)[:, 1] 
    
    # Select a small background dataset (e.g., 100 samples)
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

    # Get SHAP Values
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_test)

    # Get mean absolute SHAP values for global importance
    global_importance = np.abs(shap_values).mean(axis=0)
    importance_percentages = (global_importance / global_importance.sum()) * 100
    # feature_ranking = np.argsort(global_importance)[::-1]

    # Create df with names and importance
    feature_names = data_pipeline.named_steps['data_preprocess'].get_feature_names_out()
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_%': importance_percentages
    }).sort_values(by='importance_%', ascending=False).reset_index(drop=True)
    # print(shap_importance_df.head())

    # # Get feature names for the transformed data
    # print('Top 5 Features by SHAP importance')
    # for idx in feature_ranking[:5]:
    #     print(f"{feature_names[idx]}: {global_importance[idx]:.4f}")
    
    # Plot the Top 25
    print('SHAP Values Importance')
    print(shap_importance_df.shape)
    display(shap_importance_df.head())

    importance_df = shap_importance_df.head(25)
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance_%'], color='skyblue')
    plt.xlabel('Importance %')
    plt.ylabel('Feature')
    plt.title('Feature Importances (Sorted)')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()
    
    return shap_importance_df