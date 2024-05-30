using System;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;
using MachineLearning.Utils;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class NeuralNetwork : IComparable<NeuralNetwork>
{
    public Layer[] Layers;
    
    [XmlIgnore]
    public double Fitness = 0;

    [XmlIgnore]
    public int Rank;

    [XmlIgnore]
    private readonly Random random;

    public NeuralNetwork() => random = new(DateTime.Now.Millisecond);

    public NeuralNetwork(int[] layers)
    {
        if (layers.Length < 3)
            throw new ArgumentException("A NeuralNetwork must have at least 1 input layer, 1 hidden layer, and 1 output layer");
        
        Layers = new Layer[layers.Length];

        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i] < 1)
                throw new ArgumentException("All NeuralNetwork Layers must have at least 1 Neuron");
            
            Layers[i] = new(layers[i]);
        }

        random = new(DateTime.Now.Millisecond);

        InitNeurons();
        InitWeights();
    }

    public NeuralNetwork(NeuralNetwork copy)
    {
        Layers = new Layer[copy.Layers.Length];

        for (int i = 0; i < copy.Layers.Length; i++)
            Layers[i] = copy.Layers[i];

        random = new(DateTime.Now.Millisecond);

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
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].InitWeights(Layers[i - 1], random);
    }

    private void CopyWeights(IReadOnlyList<Layer> layers)
    {
        for (int i = 1; i < layers.Count; i++)
            Layers[i].CopyWeights(layers[i].Neurons);
    }

    public double[] FeedForward(double[] inputs)
    {
        if (inputs.Length != Layers[0].Size)
            throw new ArgumentException("Inputs array has the wrong size");
        
        for (int i = 0; i < inputs.Length; i++)
            Layers[0].Neurons[i].Value = inputs[i];

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].FeedForward(Layers[i - 1].Neurons);

        Neuron[] lastLayerNeurons = Layers[^1].Neurons;
        double[] result = new double[lastLayerNeurons.Length];

        for (int i = 0; i < result.Length; i++)
            result[i] = lastLayerNeurons[i].Value;

        return result;
    }

    public void Mutate()
    {
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Mutate(random);
    }

    public void ResetNeurons()
    {
        InitNeurons();
        InitWeights();
        
        Mutate();
    }

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
