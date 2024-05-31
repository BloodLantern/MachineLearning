using System;
using System.Collections.Generic;

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

    public void InitLinks(Layer previousLayer, Random random)
    {
        foreach (Neuron neuron in Neurons)
            neuron.InitLinks(previousLayer.Neurons, random);
    }

    public void CopyLinks(IReadOnlyList<Neuron> neurons)
    {
        for (int i = 0; i < neurons.Count; i++)
            Neurons[i].CopyLinks(neurons[i].Links);
    }

    public void MergeLinks(IReadOnlyList<Neuron> goodNeurons, IReadOnlyList<Neuron> badNeurons)
    {
        if (goodNeurons.Count != badNeurons.Count)
            throw new ArgumentException("Cannot merge neuron arrays of different sizes");

        for (int i = 0; i < goodNeurons.Count; i++)
            Neurons[i].MergeLinks(goodNeurons[i].Links, badNeurons[i].Links);
    }

    public void FeedForward(IReadOnlyList<Neuron> previousNeurons)
    {
        foreach (Neuron neuron in Neurons)
        {
            double value = 0.25;

            for (int k = 0; k < previousNeurons.Count; k++)
                value += neuron.Links[k].Weight * previousNeurons[k].Value;

            neuron.Value = Math.Tanh(value);
        }
    }

    public void Mutate(Random random)
    {
        foreach (Neuron neuron in Neurons)
            neuron.Mutate(random);
    }
}
