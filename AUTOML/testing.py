import pickle
from pycaret.regression import setup, pull, compare_models, save_model,load_model,predict_model
pipeline=load_model('templates/best_model')

print(pipeline)
#print(predict_model(pipeline,data=data1.csv))

