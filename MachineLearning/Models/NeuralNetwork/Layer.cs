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

    public void FeedForward()
    {
        foreach (Neuron neuron in Neurons)
            neuron.FeedForward();
    }

    public void Mutate(Random random)
    {
        foreach (Neuron neuron in Neurons)
            neuron.Mutate(random);
    }

    public void Learn(NeuralNetwork network, double originalFitness)
    {
        foreach (Neuron neuron in Neurons)
            neuron.Learn(network, originalFitness);
    }

    public void ApplyGradients(double learnRate)
    {
        foreach (Neuron neuron in Neurons)
            neuron.ApplyGradients(learnRate);
    }
}
