
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from bayes_opt import BayesianOptimization
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
from pytorch_tabnet.tab_model import TabNetClassifier


def run_GradientBoostingClassifier(X_train, y_train):

  def GradBoost_classifier(max_depth, max_features, learning_rate, n_estimators, subsample):
      params_gb = {}
      params_gb['max_depth'] = round(max_depth)
      params_gb['max_features'] = max_features
      params_gb['learning_rate'] = learning_rate
      params_gb['n_estimators'] = round(n_estimators)
      params_gb['subsample'] = subsample

      scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gb),
                              X_train, y_train, scoring='roc_auc', cv=5).mean()
      score = scores.mean()
      return score

  # Run Bayesian Optimization
  params_gb ={
      'max_depth':(3, 10),
      'max_features':(0.8, 1),
      'learning_rate':(0.01, 1),
      'n_estimators':(80, 150),
      'subsample': (0.8, 1)
  }

  gb_bo = BayesianOptimization(GradBoost_classifier, params_gb, random_state=111)
  gb_bo.maximize(init_points=4, n_iter=20)

  # Get the best parameters from the bayesian optimization process

  params = gb_bo.max['params']
  best_learning_rate = params['learning_rate']
  best_n_estimators = params['n_estimators']
  best_max_depth = params['max_depth']
  best_subsample = params['subsample']
  best_max_features = params['max_features']

  # Train the model with the best parameters
  best_params = {
      'learning_rate': best_learning_rate,
      'n_estimators': int(best_n_estimators),
      'max_depth': int(best_max_depth),
      'subsample': best_subsample,
      'max_features': best_max_features,
      'random_state': 111
      }

  gb_model = GradientBoostingClassifier(**best_params)
  gb_model.fit(X_train, y_train)

  return gb_model


def run_RandomForestClassifier(X_train, y_train):

  def RF_classifier(max_depth, max_features, n_estimators):
      params_rf = {}
      params_rf['max_depth'] = round(max_depth)
      params_rf['max_features'] = max_features
      params_rf['n_estimators'] = round(n_estimators)

      scores = cross_val_score(RandomForestClassifier(random_state=123, **params_rf),
                              X_train, y_train, scoring='roc_auc', cv=5).mean()
      score = scores.mean()
      return score

  # Run Bayesian Optimization
  params_rf ={
      'max_depth':(3, 10),
      'max_features':(0.8, 1),
      'n_estimators':(80, 150)
  }

  rf_bo = BayesianOptimization(RF_classifier, params_rf, random_state=111)
  rf_bo.maximize(init_points=4, n_iter=20)

  # Get the best parameters from the bayesian optimization process

  params = rf_bo.max['params']
  best_n_estimators = params['n_estimators']
  best_max_depth = params['max_depth']
  best_max_features = params['max_features']

  # Train the model with the best parameters
  best_params = {
      'n_estimators': int(best_n_estimators),
      'max_depth': int(best_max_depth),
      'max_features': best_max_features,
      'random_state': 111
      }

  rf_model = RandomForestClassifier(**best_params)
  rf_model.fit(X_train, y_train)

  return rf_model

def run_TabNetClassifier(X_train, y_train):

  def TabNet_classifier(learning_rate, n_d, n_a, n_steps, gamma, momentum, step_size, max_epochs, batch_size):

      aug = ClassificationSMOTE(p=0.2)

      params_tn = {}
      params_tn['n_d'] = int(n_d)
      params_tn['n_a'] = int(n_a)
      params_tn['n_steps'] = int(n_steps)
      params_tn['gamma'] = gamma
      params_tn['momentum'] = momentum
      params_tn['optimizer_fn'] = torch.optim.Adam
      params_tn['optimizer_params'] = dict(lr=learning_rate)
      params_tn['scheduler_params'] = {"step_size": int(step_size), "gamma":gamma}
      params_tn['scheduler_fn'] = torch.optim.lr_scheduler.StepLR
      params_tn['mask_type'] ='entmax'

      fit_params= {}
      fit_params['max_epochs'] = int(max_epochs)
      fit_params['batch_size'] = int(batch_size)
      fit_params['virtual_batch_size'] = int(batch_size//4)
      fit_params['eval_set'] = [(X_train.values, y_train)]
      fit_params['eval_name'] = ['train']
      fit_params['eval_metric'] = ['auc']
      fit_params['num_workers'] = 0
      fit_params['weights'] = 1

      scores = cross_val_score(TabNetClassifier(**params_tn),
                              X_train.values, y_train, fit_params=fit_params ,scoring='roc_auc', cv=5).mean()
      score = scores.mean()
      return score

  # Run Bayesian Optimization

  params_tn = {'learning_rate': (0.01,1),
              "n_d": (8, 64),
              'n_a': (8, 64),
              'n_steps': (3, 10),
              "gamma": (1.0, 2.0),
              'momentum': (0.01, 0.4),
              'step_size': (10, 100),
              'max_epochs': (50, 100),
              'batch_size': (256, 1024)
              }

  tn_bo = BayesianOptimization(TabNet_classifier, params_tn, random_state=111)
  tn_bo.maximize(init_points=4, n_iter=20)

  # Get the best parameters from the bayesian optimization process

  params = tn_bo.max['params']
  best_learning_rate = params['learning_rate']
  best_n_d = params['n_d']
  best_n_a = params['n_a']
  best_n_steps = params['n_steps']
  best_gamma = params['gamma']
  best_momentum = params['momentum']
  best_step_size = params['step_size']
  best_max_epochs = params['max_epochs']
  best_batch_size = params['batch_size']

  # Train the final model with the best parameters
  tabnet_best_params = {
                  "n_d": int(best_n_d),
                  "n_a": int(best_n_a),
                  "n_steps": int(best_n_steps),
                  "gamma": best_gamma,
                  "momentum": best_momentum,
                  "optimizer_fn":torch.optim.Adam,
                  "optimizer_params":dict(lr=best_learning_rate),
                  "scheduler_params":{"step_size":int(best_step_size), # how to use learning rate scheduler
                                  "gamma":best_gamma},
                  "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                  "mask_type":'entmax', # "sparsemax"
                  }

  tn_model = TabNetClassifier(**tabnet_best_params)

  aug = ClassificationSMOTE(p=0.2)

  tn_model.fit(
          X_train=X_train.values, y_train=y_train,
          eval_set=[(X_train.values, y_train)],
          eval_name=['train'],
          eval_metric=['auc'],
          max_epochs=int(best_max_epochs) , patience=20,
          batch_size=int(best_batch_size), virtual_batch_size=int(best_batch_size//4),
          num_workers=0,
          weights=1,
          drop_last=False,
          augmentations=aug, #aug, None
      )

  return tn_model

def train_model(classifier = "GradientBoostingClassifier", X_train=None, y_train=None):
  if classifier=="GradientBoostingClassifier":
    return run_GradientBoostingClassifier(X_train, y_train)
  elif classifier=="RandomForestClassifier":
    return run_RandomForestClassifier(X_train, y_train)
  elif classifier=="TabNetClassifier":
    return run_TabNetClassifier(X_train, y_train)



