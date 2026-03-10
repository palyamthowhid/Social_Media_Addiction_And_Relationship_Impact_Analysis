import joblib, os, sklearn
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
print('sklearn version', sklearn.__version__)
model = joblib.load(os.path.join(MODELS_DIR, "model_addiction.joblib"))
print('model type', type(model))
print('has multi_class?', hasattr(model, 'multi_class'))
print('attributes snippet', [a for a in dir(model) if 'multi' in a])
