using System;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class NeuralNetwork : IComparable<NeuralNetwork>
{
    public delegate double FitnessComputation(NeuralNetwork network);

    [XmlIgnore]
    public Layer[] Layers;

    [XmlElement("Layers", Order = 1)]
    public Layer[] SerializedLayers
    {
        get => Layers[1..];
        set
        {
            Layers = new Layer[LayerSizes.Length];

            Layers[0] = new(LayerSizes[0]);

            for (int i = 0; i < value.Length; i++)
                Layers[i + 1] = value[i];
        }
    }

    [XmlElement("LayerSizes", Order = 0)]
    public int[] LayerSizes;

    [XmlIgnore]
    public double Fitness;

    [XmlIgnore]
    public FitnessComputation FitnessFunction;

    [XmlIgnore]
    public int Rank;

    [XmlIgnore]
    private readonly Random random;

    public NeuralNetwork() => random = new(DateTime.Now.Millisecond);

    public NeuralNetwork(Random random, params int[] layerSizes)
        : this(random, null, layerSizes)
    {
    }

    public NeuralNetwork(Random random, FitnessComputation fitnessFunction, params int[] layerSizes)
    {
        this.random = random;
        FitnessFunction = fitnessFunction;

        InitLayers(layerSizes);
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
        FitnessFunction = copy.FitnessFunction;

        InitLayers(copy.LayerSizes);
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
        FitnessFunction = badNetwork.FitnessFunction;

        InitLayers(goodNetwork.LayerSizes);
        InitNeurons();
        InitLinks();
        MergeLinks(goodNetwork.Layers, badNetwork.Layers);
    }

    private void InitLayers(params int[] layerSizes)
    {
        if (layerSizes.Length < 3)
            throw new ArgumentException("A NeuralNetwork must have at least 1 input layer, 1 hidden layer, and 1 output layer");
        
        LayerSizes = layerSizes;
        Layers = new Layer[layerSizes.Length];

        for (int i = 0; i < layerSizes.Length; i++)
        {
            if (layerSizes[i] < 1)
                throw new ArgumentException("All NeuralNetwork Layers must have at least 1 Neuron");
            
            Layers[i] = new(layerSizes[i]);
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

    public double[] ComputeOutputs(IReadOnlyList<double> inputs)
    {
        if (inputs.Count != Layers[0].Size)
            throw new ArgumentException("Inputs array has the wrong size");
        
        for (int i = 0; i < inputs.Count; i++)
            Layers[0].Neurons[i].Value = inputs[i];

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].FeedForward();

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

    public void Learn(double learnRate) => Learn(FitnessFunction, learnRate);

    public void Learn(FitnessComputation fitnessFunction, double learnRate)
    {
        double originalFitness = fitnessFunction(this);
        
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Learn(this, originalFitness);
        
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].ApplyGradients(learnRate);
    }

    public void ResetNeurons()
    {
        InitNeurons();
        InitLinks();
        
        Mutate();
    }

    public double ComputeFitness()
    {
        if (FitnessFunction == null)
            throw new ArgumentNullException(nameof(FitnessFunction), "Cannot call ComputeFitness() on a NeuralNetwork with a null FitnessFunction");
        
        return ComputeFitness(FitnessFunction);
    }

    public double ComputeFitness(FitnessComputation fitnessFunction) => fitnessFunction(this);

    internal double ComputeFitnessDifference(double originalFitness, ref double value)
    {
        const double Offset = 1e-5;

        value += Offset;
        double fitnessDiff = ComputeFitness() - originalFitness;
        value -= Offset;
        
        return fitnessDiff / Offset;
    }

    public void UpdateFitness() => Fitness = ComputeFitness();

    public void Save(string path) => File.WriteAllText(path, Utils.GetXml(this, true));

    public static NeuralNetwork Load(string path) => Utils.LoadFromXml<NeuralNetwork>(File.ReadAllText(path));

    public int CompareTo(NeuralNetwork other)
    {
        if (other == null)
            return -1;

        if (Fitness > other.Fitness)
            return -1;
            
        return Fitness < other.Fitness ? 1 : 0;
    }
}
