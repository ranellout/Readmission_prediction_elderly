
import matplotlib.pyplot as plt
from yellowbrick.model_selection import FeatureImportances

def exp_models(model_list, X, y):
  for model in model_list:
    viz = FeatureImportances(model, topn=20, relative=False, absolute=True)
    viz.fit(X, y)
    plt.savefig(type(model).__name__+".pdf", format="pdf", bbox_inches="tight")
    viz.show()