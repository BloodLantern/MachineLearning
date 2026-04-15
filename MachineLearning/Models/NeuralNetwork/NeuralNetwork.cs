using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

// TODO - Add binary serialization

[Serializable]
public class NeuralNetwork
{
    public delegate double RewardComputation(NeuralNetwork network);

    public static NeuralNetwork Load(string path) => Utils.LoadFromXml<NeuralNetwork>(File.ReadAllText(path));

    [XmlIgnore]
    public RewardComputation RewardFunction;

    [XmlIgnore]
    public Layer[] Layers;

    public int LayerCount => Layers.Length;

    [XmlElement("Layers")]
    public Layer[] SerializedLayers
    {
        get => Layers[1..];
        set
        {
            if (value.Length < 1)
                throw new ArgumentException("Invalid layer count");

            Layers = new Layer[value.Length + 1];

            Layers[0] = new(value.First().PreviousLayerNeuronCount);

            for (int i = 0; i < value.Length; i++)
                Layers[i + 1] = value[i];
        }
    }

    public Layer InputLayer => Layers.First();

    public Layer[] HiddenLayers => Layers[1..^1];

    public Layer OutputLayer => Layers.Last();

    public double[] Outputs => OutputLayer.Outputs;

    public NeuralNetwork() { }

    public NeuralNetwork(Random random, int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(random, null, inputCount, outputCount, hiddenLayerSizes) { }

    public NeuralNetwork(Random random, RewardComputation rewardFunction, int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        RewardFunction = rewardFunction;

        InitLayers(inputCount, outputCount, hiddenLayerSizes);
        InitNeurons();
        InitLinks(random);
    }

    /// <summary>
    /// Constructs a new NeuralNetwork, which is a deep copy of the given network.
    /// </summary>
    /// <param name="copy">The network to create a deep copy of.</param>
    public NeuralNetwork(NeuralNetwork copy)
    {
        RewardFunction = copy.RewardFunction;

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

        RewardFunction = badNetwork.RewardFunction;

        InitLayers(
            goodNetwork.InputLayer.NeuronCount,
            goodNetwork.OutputLayer.NeuronCount,
            goodNetwork.HiddenLayers.Select(l => l.NeuronCount).ToArray()
        );
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
            Layers[i].InitLinks(Layers[i - 1]);
    }

    private void InitLinks(Random random)
    {
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].InitLinks(Layers[i - 1], random);
    }

    private void CopyLinks(Layer[] layers)
    {
        for (int i = 1; i < layers.Length; i++)
            Layers[i].CopyLinks(layers[i].Neurons);
    }

    private void MergeLinks(Layer[] goodLayers, Layer[] badLayers)
    {
        for (int i = 1; i < goodLayers.Length; i++)
            Layers[i].MergeLinks(goodLayers[i].Neurons, badLayers[i].Neurons);
    }

    public double[] ComputeOutputs(
        double[] inputs, ActivationFunction hiddenLayersActivationFunction, ActivationFunction outputLayerActivationFunction
    )
    {
        if (inputs.Length != Layers[0].NeuronCount)
            throw new ArgumentException("Inputs array has the wrong size");

        for (int i = 0; i < inputs.Length; i++)
            Layers[0].Neurons[i].Output = inputs[i];

        for (int i = 1; i < Layers.Length - 1; i++)
            Layers[i].FeedForward(hiddenLayersActivationFunction);

        OutputLayer.FeedForward(outputLayerActivationFunction);

        Neuron[] lastLayerNeurons = OutputLayer.Neurons;
        double[] result = new double[lastLayerNeurons.Length];

        for (int i = 0; i < result.Length; i++)
            result[i] = lastLayerNeurons[i].Output;

        return result;
    }

    public void Mutate(Random random)
    {
        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Mutate(random);
    }

    public void LearnByGradientDescent(double gain) => LearnByGradientDescent(gain, RewardFunction);

    public void LearnByGradientDescent(double gain, RewardComputation rewardFunction)
    {
        Debug.Assert(double.IsFinite(gain));

        double originalReward = ComputeRewardGain(rewardFunction);

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].Learn(this, originalReward, rewardFunction);

        for (int i = 1; i < Layers.Length; i++)
            Layers[i].ApplyGradients(gain);
    }

    public void LearnByBackpropagation(
        double gain, double[] inputs, double[] expectedOutputs, ActivationFunction hiddenLayersActivationFunction,
        ActivationFunction outputLayerActivationFunction
    )
    {
        Debug.Assert(double.IsFinite(gain));

        ComputeOutputs(inputs, hiddenLayersActivationFunction, outputLayerActivationFunction);

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
                foreach (Neuron nextNeuron in nextLayer.Neurons)
                    sum += nextNeuron.Weights[j] * nextNeuron.LastErrorFactor;

                double errorFactor = neuron.Output * (1.0 - neuron.Output) * sum;
                neuron.LearnByBackpropagation(errorFactor, gain, previousLayer.Outputs);
            }
        }
    }

    public double ComputeRewardGain()
    {
        if (RewardFunction == null)
        {
            throw new ArgumentNullException(
                nameof(RewardFunction),
                $"Cannot call {nameof(ComputeRewardGain)} on a {nameof(NeuralNetwork)} with a null {nameof(RewardFunction)}"
            );
        }

        return ComputeRewardGain(RewardFunction);
    }

    public double ComputeRewardGain(RewardComputation rewardFunction) => rewardFunction(this);

    internal double ComputeRewardDifference(double originalRewardGain, ref double value)
        => ComputeRewardDifference(originalRewardGain, ref value, RewardFunction);

    internal double ComputeRewardDifference(double originalRewardGain, ref double value, RewardComputation rewardFunction)
    {
        const double Offset = 1e-5;

        value += Offset;
        double rewardDiff = ComputeRewardGain(rewardFunction) - originalRewardGain;
        value -= Offset;

        return rewardDiff / Offset;
    }

    public void Save(string path) => File.WriteAllText(path, Utils.GetXml(this, true), Encoding.Unicode);
}
