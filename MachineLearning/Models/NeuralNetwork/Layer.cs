using System;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Layer
{
    public int Size => Neurons.Length;
    
    public Neuron[] Neurons;

    public Layer()
    {
    }

    public Layer(int size) => Neurons = new Neuron[size];

    public void InitNeurons()
    {
        for (int i = 0; i < Neurons.Length; i++)
            Neurons[i] = new();
    }

    public void InitWeights(Layer previousLayer, Random random)
    {
        foreach (Neuron neuron in Neurons)
            neuron.InitWeights(previousLayer.Size, random);
    }

    public void CopyWeights(Neuron[] neurons)
    {
        for (int i = 0; i < neurons.Length; i++)
            Neurons[i].CopyWeights(neurons[i].Weights);
    }

    public void FeedForward(Neuron[] previousNeurons)
    {
        foreach (Neuron neuron in Neurons)
        {
            float value = 0.25f;

            for (int k = 0; k < previousNeurons.Length; k++)
                value += neuron.Weights[k] * previousNeurons[k].Value;

            neuron.Value = MathF.Tanh(value);
        }
    }

    public void Mutate(Random random)
    {
        foreach (Neuron neuron in Neurons)
            neuron.Mutate(random);
    }
}
