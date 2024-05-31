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

    public NeuralNetwork(int[] layers, Random random)
    {
        this.random = random;

        InitLayers(layers);
        InitNeurons();
        InitLinks();
    }

    /// <summary>
    /// Constructs a new NeuralNetwork which is a deep copy of the given network.
    /// </summary>
    /// <param name="copy">The network to create a deep copy of.</param>
    public NeuralNetwork(NeuralNetwork copy)
    {
        random = copy.random;

        InitLayers(GetLayerSizeArray(copy.Layers));
        InitNeurons();
        InitLinks();
        CopyLinks(copy.Layers);
    }

    /// <summary>
    /// Constructs a new NeuralNetwork which is the result of a merge between a good and a bad network.
    /// </summary>
    /// <param name="goodNetwork">A good network to merge with the bad one.</param>
    /// <param name="badNetwork">A bad network to merge with the good one.</param>
    public NeuralNetwork(NeuralNetwork goodNetwork, NeuralNetwork badNetwork)
    {
        if (goodNetwork.Layers.Length != badNetwork.Layers.Length)
            throw new ArgumentException("Cannot merge networks with different amounts of layers");
        
        random = badNetwork.random;

        InitLayers(GetLayerSizeArray(goodNetwork.Layers));
        InitNeurons();
        InitLinks();
        MergeLinks(goodNetwork.Layers, badNetwork.Layers);
    }

    private void InitLayers(int[] layers)
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
    }

    private void InitNeurons()
    {
        foreach (Layer layer in Layers)
            layer.InitNeurons();
    }

    private void InitLinks()
    {
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].InitLinks(Layers[i - 1], random);
    }

    private void CopyLinks(IReadOnlyList<Layer> layers)
    {
        for (int i = 1; i < layers.Count; i++)
            Layers[i].CopyLinks(layers[i].Neurons);
    }
    
    private void MergeLinks(IReadOnlyList<Layer> goodLayers, IReadOnlyList<Layer> badLayers)
    {
        for (int i = 1; i < goodLayers.Count; i++)
            Layers[i].MergeLinks(goodLayers[i].Neurons, badLayers[i].Neurons);
    }

    public double[] FeedForward(IReadOnlyList<double> inputs)
    {
        if (inputs.Count != Layers[0].Size)
            throw new ArgumentException("Inputs array has the wrong size");
        
        for (int i = 0; i < inputs.Count; i++)
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
        InitLinks();
        
        Mutate();
    }

    private static int[] GetLayerSizeArray(Layer[] layers)
    {
        int[] result = new int[layers.Length];

        for (int i = 0; i < layers.Length; i++)
            result[i] = layers[i].Size;

        return result;
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
