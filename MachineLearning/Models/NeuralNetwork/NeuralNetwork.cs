using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class NeuralNetwork : IComparable<NeuralNetwork>
{
    public delegate double FitnessComputation(NeuralNetwork network);

    [XmlIgnore]
    public Layer[] Layers;

    public int LayerCount => Layers.Length;

    public ActivationFunction ActivationFunction = ActivationFunctions.Sigmoid;

    [XmlElement("Layers", Order = 1)]
    public Layer[] SerializedLayers
    {
        get => Layers[1..];
        set
        {
            Layers = new Layer[Layers.Length];

            Layers[0] = new(Layers[0].NeuronCount);

            for (int i = 0; i < value.Length; i++)
                Layers[i + 1] = value[i];
        }
    }

    public Layer InputLayer => Layers.First();

    public Layer[] HiddenLayers => Layers[1..^1];

    public Layer OutputLayer => Layers.Last();

    public double[] Outputs => OutputLayer.Outputs;

    [XmlIgnore]
    public double Fitness;

    [XmlIgnore]
    public FitnessComputation FitnessFunction;

    [XmlIgnore]
    public int Rank;

    [XmlIgnore]
    private readonly Random random;

    public NeuralNetwork() => random = new();

    public NeuralNetwork(Random random, int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(random, null, inputCount, outputCount, hiddenLayerSizes)
    {
    }

    public NeuralNetwork(Random random, FitnessComputation fitnessFunction, int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        this.random = random;
        FitnessFunction = fitnessFunction;

        InitLayers(inputCount, outputCount, hiddenLayerSizes);
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

        InitLayers(copy.InputLayer.NeuronCount, copy.OutputLayer.NeuronCount, copy.HiddenLayers.Select(l => l.NeuronCount).ToArray());
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

        InitLayers(goodNetwork.InputLayer.NeuronCount, goodNetwork.OutputLayer.NeuronCount,
            goodNetwork.HiddenLayers.Select(l => l.NeuronCount).ToArray());
        InitNeurons();
        InitLinks();
        MergeLinks(goodNetwork.Layers, badNetwork.Layers);
    }

    private void InitLayers(int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        Layers = new Layer[2 + hiddenLayerSizes.Length];

        Layers[0] = new(inputCount);
        for (int i = 0; i < hiddenLayerSizes.Length; i++)
            Layers[i + 1] = new(hiddenLayerSizes[i]);
        Layers[^1] = new(outputCount);
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

    public double[] ComputeOutputs(params double[] inputs)
    {
        if (inputs.Length != Layers[0].NeuronCount)
            throw new ArgumentException("Inputs array has the wrong size");

        for (int i = 0; i < inputs.Length; i++)
            Layers[0].Neurons[i].Output = inputs[i];

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].FeedForward(ActivationFunction);

        Neuron[] lastLayerNeurons = Layers[^1].Neurons;
        double[] result = new double[lastLayerNeurons.Length];

        for (int i = 0; i < result.Length; i++)
            result[i] = lastLayerNeurons[i].Output;

        return result;
    }

    public void Mutate()
    {
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Mutate(random);
    }

    public void LearnByFitness(double gain) => LearnByFitness(FitnessFunction, gain);

    public void LearnByFitness(FitnessComputation fitnessFunction, double gain)
    {
        double originalFitness = fitnessFunction(this);

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Learn(this, originalFitness);

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].ApplyGradients(gain);
    }

    public void LearnByBackpropagation(double gain, double[] inputs, double[] expectedOutputs)
    {
        ComputeOutputs(inputs);

        for (int i = 0; i < OutputLayer.NeuronCount; i++)
        {
            Neuron neuron = OutputLayer[i];
            double errorFactor = neuron.Output * (1.0 - neuron.Output) * (expectedOutputs[i] - neuron.Output);
            neuron.LearnByBackpropagation(errorFactor, gain, Layers[^2].Outputs);
        }

        for (int i = 1; i < Layers.Length - 1; i++)
        {
            Layer previousLayer = Layers[i - 1];
            Layer layer = Layers[i];
            Layer nextLayer = Layers[i + 1];

            for (int j = 0; j < layer.NeuronCount; j++)
            {
                Neuron neuron = layer[j];

                double sum = 0f;
                foreach (Neuron nextNeuron in nextLayer)
                    sum += nextNeuron.Weights[j] * nextNeuron.LastErrorFactor;

                double errorFactor = neuron.Output * (1.0 - neuron.Output) * sum;
                neuron.LearnByBackpropagation(errorFactor, gain, previousLayer.Outputs);
            }
        }
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
