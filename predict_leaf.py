import pickle
from PIL import Image
import math
import sys
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Neuron():
    def __init__(self,num_inputs):
        self.weights=[]
        self.bias=0.0
    def forward(self,inputs):
        z=0
        for i in range(len(inputs)):
            z+=inputs[i]*self.weights[i]
        z+=self.bias
        return sigmoid(z)

def prepare_image(img_path,target_size=(32,32)):
    img = Image.open(img_path)
    img_resized=img.resize(target_size)
    img_array=np.array(img_resized)
    img_normalized=img_array/255.0
    img_flaten=img_normalized.flatten()
    return img_flaten

def load_model(filename="leaf_model_rgb_v2.pkl"):
    with open(filename,'rb') as f:
        model_data=pickle.load(f)
    hidden_layer=[]
    for weights, bias in zip(model_data['hidden_weights'],model_data['hidden_biases']):
        neuron=Neuron(len(weights))
        neuron.weights=weights
        neuron.bias=bias
        hidden_layer.append(neuron)
    weights=model_data['output_weights']
    bias=model_data['output_bias']
    output_neuron=Neuron(len(weights))
    output_neuron.weights=weights
    output_neuron.bias=bias

    return hidden_layer,output_neuron

def predict(img_path,hidden_layer,output_neuron):
    img_flaten=prepare_image(img_path,target_size=(32,32))
    hidden_output=[]
    for hidden_neuron in hidden_layer:
        hidden_output.append(hidden_neuron.forward(img_flaten))
    final_output=output_neuron.forward(hidden_output)
    predicted=1 if final_output>0.5 else 0
    return final_output,predicted

if __name__ == '__main__':
    if len(sys.argv)>1:
        img_path=sys.argv[1]
    else:
        print("\n enter the path yo your photo:")
        img_path=input("> ").strip()
    try:
        print("loading trained model...")
        hidden_layer,output_neuron=load_model('leaf_model_rgb_v2.pkl')
        confidence,prediction=predict(img_path,hidden_layer,output_neuron)
        if prediction==1:
            print("✔️ this is a parijat leaf.")
            print(f"confidence: {confidence*100:.1f}%")
        else:
            print("❌ this is not a parijat leaf!")
            print(f"confidence: {(1-confidence)*100:.1f}%")
    except FileNotFoundError:
        print(f"Error could not found fil_path: '{img_path}'")
    except Exception as e:
        print("An error occured!")
