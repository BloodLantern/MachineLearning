using System;
using System.Collections;
using System.Linq;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Layer
{
    public int NeuronCount => Neurons.Length;

    public double[] Outputs => Neurons.Select(n => n.Output).ToArray();

    public Neuron[] Neurons;

    public int PreviousLayerNeuronCount => Neurons.First().InputCount;

    public Layer()
    {
    }

    public Layer(int size)
    {
        if (size < 1)
            throw new ArgumentException("All NeuralNetwork Layers must have at least 1 Neuron");

        Neurons = new Neuron[size];
    }

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

    public void CopyLinks(Neuron[] neurons)
    {
        for (int i = 0; i < neurons.Length; i++)
            Neurons[i].CopyLinks(neurons[i].Links);
    }

    public void MergeLinks(Neuron[] goodNeurons, Neuron[] badNeurons)
    {
        if (goodNeurons.Length != badNeurons.Length)
            throw new ArgumentException("Cannot merge neuron arrays of different sizes");

        for (int i = 0; i < goodNeurons.Length; i++)
            Neurons[i].MergeLinks(goodNeurons[i].Links, badNeurons[i].Links);
    }

    public void FeedForward(ActivationFunction activationFunction)
    {
        foreach (Neuron neuron in Neurons)
            neuron.FeedForward(activationFunction);
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

    public void ApplyGradients(double gain)
    {
        foreach (Neuron neuron in Neurons)
            neuron.ApplyGradients(gain);
    }

    public Neuron this[int index] => Neurons[index];
}
