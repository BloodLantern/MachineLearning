using System;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Layer
{
    public Neuron[] Neurons;

    public Layer(int size) => Neurons = new Neuron[size];

    public void InitNeurons()
    {
        for (int i = 0; i < Neurons.Length; i++)
            Neurons[i] = new();
    }

    public void InitWeights(Layer previousLayer)
    {
        foreach (Neuron neuron in Neurons)
            neuron.InitWeights(previousLayer);
    }

    public void CopyWeights(Neuron[] neurons)
    {
        for (int i = 0; i < neurons.Length; i++)
            Neurons[i].CopyWeights(neurons[i].Weights);
    }

    public void FeedForward(Neuron[] previousNeurons)
    {
        for (int j = 0; j < Neurons.Length; j++)
        {
            float value = 0.25f;

            for (int k = 0; k < previousNeurons.Length; k++)
                value += previousNeurons[j].Weights[k] * previousNeurons[k];

            Neurons[j].Value = MathF.Tanh(value);
        }
    }

    public void Mutate()
    {
        foreach (Neuron neuron in Neurons)
            neuron.Mutate();
    }

    public static implicit operator int(Layer layer) => layer.Neurons.Length;
}
