import pickle
def save_model(model,path):
  with open(path, 'wb') as f:
    pickle.dump(model, f)
  print(f"Model saved: {path}")

def load_model(path):
  with open(path, 'rb') as f:
    model=pickle.load(f)
  print("Model Loaded!")
  return model