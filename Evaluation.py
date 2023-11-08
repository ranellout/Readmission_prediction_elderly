
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

def eval_models(model_list, X_valid, y_valid):

  # for ROC curves

  plt.figure()

  for model in model_list:
    if type(model).__name__!="TabNetClassifier":
      y_pred=model.predict(X_valid) # predict the validation data

      # Compute False postive rate, and True positive rate
      fpr, tpr, thresholds = roc_curve(y_valid, model.predict_proba(X_valid)[:,1])

      # Calculate Area under the curve to display on the plot
      roc_auc = auc(fpr, tpr)

      # Now, plot the computed values
      plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (type(model).__name__, roc_auc))

      print("for", type(model).__name__, "ROC curve:")
      print(f'accuracy is: {accuracy_score(y_valid, y_pred)}')
      print(f'precision is: {precision_score(y_valid, y_pred)}')
      print(f'recall is: {recall_score(y_valid, y_pred)}')
      print(f'f1 is: {f1_score(y_valid, y_pred)}')

      print(f"optimal threshold is {round(tpr[np.where(thresholds==thresholds[np.argmin(abs(tpr-(1-fpr)))])[0]][0],3)} sensitivity and {round(1-fpr[np.where(thresholds==thresholds[np.argmin(abs(tpr-(1-fpr)))])[0]][0],3)} specificity")
      print("")

    else:
      y_pred=model.predict(X_valid.values) # predict the validation data

      # Compute False postive rate, and True positive rate
      fpr, tpr, thresholds = roc_curve(y_valid.values, model.predict_proba(X_valid.values)[:,1])

      # Calculate Area under the curve to display on the plot
      roc_auc = auc(fpr, tpr)

      # Now, plot the computed values
      plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (type(model).__name__, roc_auc))

      print("for", type(model).__name__, "ROC curve:")
      print(f'accuracy is: {accuracy_score(y_valid.values, y_pred)}')
      print(f'precision is: {precision_score(y_valid.values, y_pred)}')
      print(f'recall is: {recall_score(y_valid.values, y_pred)}')
      print(f'f1 is: {f1_score(y_valid.values, y_pred)}')

      print(f"optimal threshold is {round(tpr[np.where(thresholds==thresholds[np.argmin(abs(tpr-(1-fpr)))])[0]][0],3)} sensitivity and {round(1-fpr[np.where(thresholds==thresholds[np.argmin(abs(tpr-(1-fpr)))])[0]][0],3)} specificity")
      print("")

  # Custom settings for the plot
  plt.plot([0, 1], [0, 1],'r--', color = 'gray')
  plt.xlim([-0.005, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('1-Specificity (False Positive Rate)', fontsize=13)
  plt.ylabel('Sensitivity (True Positive Rate)', fontsize=13)
  plt.title('Receiver Operating Characteristic', fontweight = 'bold', fontsize=15)
  plt.legend(prop={'size':10}, loc="lower right")
  plt.show()

  # for Precision-Recall curves

  plt.figure()

  for model in model_list:
    if type(model).__name__!="TabNetClassifier":
      y_pred=model.predict(X_valid) # predict the validation data

      # Compute Precision, and Recall
      precision, recall, thresholds = precision_recall_curve(y_valid, model.predict_proba(X_valid)[:,1])

      # Calculate Average precision
      avg_pr = average_precision_score(y_valid, y_pred)

      # Now, plot the computed values
      plt.plot(recall, precision, label='%s Precision-Recall curve (average precision = %0.2f)' % (type(model).__name__, avg_pr))

    else:
      y_pred=model.predict(X_valid.values) # predict the validation data

      # Compute Precision, and Recall
      precision, recall, thresholds = precision_recall_curve(y_valid.values, model.predict_proba(X_valid.values)[:,1])

      # Calculate Average precision
      avg_pr = average_precision_score(y_valid.values, y_pred)

      # Now, plot the computed values
      plt.plot(recall, precision, label='%s Precision-Recall curve (average precision = %0.2f)' % (type(model).__name__, avg_pr))

  # Custom settings for the plot
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.01])
  plt.xlabel('Recall', fontsize=13)
  plt.ylabel('Precision', fontsize=13)
  plt.title('Precision Recall Curve', fontweight = 'bold', fontsize=15)
  plt.legend(prop={'size':10}, loc="lower right")
  plt.show()
  return