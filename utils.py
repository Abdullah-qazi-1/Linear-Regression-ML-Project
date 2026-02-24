import pickle
def save_model(model,filename):
  with open(filename, 'wb') as f:
    pickle.dump(model, f)
  print(f"Model saved: {filename}")

def load_model(filename):
  with open(filename, 'rb') as f:
    model=pickle.load(f)
  print("Model Loaded!")
  return model