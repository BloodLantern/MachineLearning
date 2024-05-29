using System;
using System.Collections.Generic;
using System.IO;
using MachineLearning.Utils;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class NeuralNetwork : IComparable<NeuralNetwork>
{
    public Layer[] Layers;
    
    [NonSerialized]
    public float Fitness = 0;

    [NonSerialized]
    public int Rank;

    private readonly Random random = new();

    public NeuralNetwork()
    {
    }

    public NeuralNetwork(int[] layers)
    {
        if (layers.Length < 3)
            throw new ArgumentException("A NeuralNetwork must have at least 1 input layer, 1 hidden layer, and 1 output layer");
        
        Layers = new Layer[layers.Length];

        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i] < 1)
                throw new ArgumentException("A NeuralNetwork Layer must have at least 1 neuron");
            
            Layers[i] = new(layers[i]);
        }

        InitNeurons();
        InitWeights();
    }

    public NeuralNetwork(NeuralNetwork copy)
    {
        Layers = new Layer[copy.Layers.Length];

        for (int i = 0; i < copy.Layers.Length; i++)
            Layers[i] = copy.Layers[i];

        InitNeurons();
        InitWeights();
        CopyWeights(copy.Layers);
    }

    private void InitNeurons()
    {
        foreach (Layer layer in Layers)
            layer.InitNeurons();
    }

    private void InitWeights()
    {
        Layers[0].InitWeights(Layers[1]);
        
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].InitWeights(Layers[i - 1]);
    }

    private void CopyWeights(IReadOnlyList<Layer> layers)
    {
        for (int i = 0; i < layers.Count; i++)
            Layers[i].CopyWeights(layers[i].Neurons);
    }

    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < inputs.Length; i++)
            Layers[0].Neurons[i].Value = inputs[i];

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].FeedForward(Layers[i - 1].Neurons);

        Neuron[] lastLayerNeurons = Layers[^1].Neurons;
        float[] result = new float[lastLayerNeurons.Length];

        for (int i = 0; i < result.Length; i++)
            result[i] = lastLayerNeurons[i];

        return result;
    }

    public void Mutate()
    {
        foreach (Layer layer in Layers)
            layer.Mutate();
    }

    public void ResetNeurons() => InitNeurons();

    public void Save(string path) => File.WriteAllText(path, this.GetXml(true));

    public static NeuralNetwork Load(string path) => File.ReadAllText(path).LoadFromXml<NeuralNetwork>();

    public int CompareTo(NeuralNetwork other)
    {
        if (other == null)
            return -1;

        if (Fitness > other.Fitness)
            return -1;
            
        return Fitness < other.Fitness ? 1 : 0;
    }
}
