from prepare_data import load_data
import math
import random
import pickle
import numpy as np 

print("loading data...")
images,labels=load_data("data/healthy")
images_flatend = images.reshape(images.shape[0],-1)

indices = np.random.permutation(len(images_flatend))
images_flatend=images_flatend[indices]
labels=labels[indices]

split = int(0.8*len(images_flatend))
training_images=images_flatend[:split]
training_labels=labels[:split]

test_images=images_flatend[split:]
test_labels=labels[split:]

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Neuron():
    def __init__(self,num_inputs):
        self.weights=[random.uniform(-0.5,0.5) for _ in range(num_inputs)]
        self.bias=0.0
    def forward(self,inputs):
        z=0
        for i in range(len(inputs)):
            z+=inputs[i]*self.weights[i]
        z+=self.bias
        return sigmoid(z)


def save_trained_model(hidden_layer,output_neuron,filename="leaf_model.pkl"):
    model_data={
        'hidden_weights': [neuron.weights for neuron in hidden_layer],
        'hidden_biases': [neuron.bias for neuron in hidden_layer],
        'output_weights': output_neuron.weights,
        'output_bias': output_neuron.bias
        }
    with open(filename,"wb") as f:
        pickle.dump(model_data,f)

learning_rate=0.05
epochs=2000

input_size=32*32*3
hidden_size=50
output_size=1

hidden_layer=[Neuron(input_size) for _ in range(hidden_size)]
output_neuron=Neuron(hidden_size)
print("loaded ✔️")
print("Starting training..")
for epoch in range(epochs):
    total_error=0
    for img,label in zip(training_images,training_labels):
        hidden_output=[]
        for hidden_neuron in hidden_layer:
            hidden_output.append(hidden_neuron.forward(img))
        final_output=output_neuron.forward(hidden_output)


        error=label-final_output
        total_error+=error**2
        '''
        we do sqr of eero and then add up
        to total becasue we dont want to get total error cut down when erro is -vee, a -ve error
        is also an error, and psoitive also, an error is an errro it should be add up not canceld out!
        -ve is jsut the direction
        '''

        for i in range(len(output_neuron.weights)):
            grad=error*final_output*(1-final_output)*hidden_output[i]
            output_neuron.weights[i]+=learning_rate*grad
        grad_bias=error*final_output*(1-final_output)*1
        output_neuron.bias+=learning_rate*grad_bias

        for h_indx, hidden_neuron in enumerate(hidden_layer):
            hidden_error=error*output_neuron.weights[h_indx]
            hidden_out = hidden_output[h_indx]
            for i in range(len(hidden_neuron.weights)):
                grad = hidden_error*hidden_out*(1-hidden_out)*img[i]
                hidden_neuron.weights[i]+=learning_rate*grad
            grad_bias=hidden_error*hidden_out*(1-hidden_out)*1
            hidden_neuron.bias+=learning_rate*grad_bias
    if epoch%10==0:
        print(f"epoch: {epoch} error: {total_error}")

print("testing: ")
correct=0
for img, label in zip(test_images,test_labels):
    hidden_output=[]
    for hidden_neuron in hidden_layer:
        hidden_output.append(hidden_neuron.forward(img))
    final_output=output_neuron.forward(hidden_output)
    predicted = 1 if final_output > 0.5 else 0
    if predicted==label:
        correct+=1
print(f"\nAccuracy: {correct}/{len(test_images)} = {correct/len(test_images)*100:.1f}%")

print("show some sample predictions")
for img, label in zip(test_images[:10],test_labels[:10]):
    hidden_output=[]
    for hidden_neuron in hidden_layer:
        hidden_output.append(hidden_neuron.forward(img))
    final_output=output_neuron.forward(hidden_output)
    predicted = 1 if final_output > 0.5 else 0
    label_type= "parijat" if label else "other"
    prediction_type="parijat" if predicted else "other"
    mark = '✔️' if label==predicted else "❌"
    print(f" Label_type : {label_type}; prediction_type: {prediction_type}; so its {mark}")


        
save_trained_model(hidden_layer,output_neuron,"leaf_model_rgb_v2.pkl")
         







