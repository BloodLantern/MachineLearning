using System;
using System.Linq;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    [XmlAttribute]
    public double Bias = 0.1;

    [XmlIgnore]
    internal double BiasGradient;

    /// <summary>
    /// Links with the previous layer's neurons. <c>.Length</c> should always be equal to <c>PreviousLayer.NeuronCount</c>.
    /// </summary>
    public Link[] Links;

    [XmlIgnore]
    public double Output;

    public int InputCount => Links.Length;

    public double[] Weights => Links.Select(l => l.Weight).ToArray();

    [XmlIgnore]
    public double LastErrorFactor { get; private set; }

    public Neuron() { }

    public Neuron(double output) => Output = output;

    public void InitLinks(Neuron[] previousNeurons)
    {
        Links = new Link[previousNeurons.Length];

        // Set the weights randomly between -BaseWeightRange and BaseWeightRange
        for (int i = 0; i < previousNeurons.Length; i++)
            Links[i] = new(0.0, previousNeurons[i], this);
    }

    public void InitLinks(Neuron[] previousNeurons, Random random)
    {
        const double BaseWeightRange = 1.0;

        Links = new Link[previousNeurons.Length];

        // Set the weights randomly between -BaseWeightRange and BaseWeightRange
        for (int i = 0; i < previousNeurons.Length; i++)
            Links[i] = new(random.NextDouble() * BaseWeightRange * 2.0 - BaseWeightRange, previousNeurons[i], this);
    }

    public void CopyLinks(Link[] links)
    {
        for (int i = 0; i < InputCount; i++)
            Links[i].CopyWeight(links[i]);
    }

    public void MergeLinks(Link[] goodLinks, Link[] badLinks)
    {
        if (goodLinks.Length != badLinks.Length)
            throw new ArgumentException("Cannot merge weight arrays of different sizes");

        for (int i = 0; i < goodLinks.Length; i++)
            Links[i].MergeWeights(goodLinks[i], badLinks[i]);
    }

    public void FeedForward(ActivationFunction activationFunction)
    {
        double value = Bias;

        foreach (Link link in Links)
            value += link.Weight * link.Origin.Output;

        Output = activationFunction(value);
    }

    public void Mutate(Random random)
    {
        foreach (Link link in Links)
            link.Mutate(random);

        Utils.MutateValue(random, ref Bias);
    }

    public void Learn(NeuralNetwork network, double originalReward, NeuralNetwork.RewardComputation rewardFunction)
    {
        foreach (Link link in Links)
            link.Learn(network, originalReward, rewardFunction);

        BiasGradient = network.ComputeRewardDifference(originalReward, ref Bias, rewardFunction);
    }

    public void ApplyGradients(double gain)
    {
        foreach (Link link in Links)
            link.ApplyGradients(gain);

        Bias -= BiasGradient * gain;
    }

    public void LearnByBackpropagation(double errorFactor, double gain, double[] previousLayerOutputs)
    {
        for (int i = 0; i < InputCount; i++)
            Links[i].Weight += gain * errorFactor * previousLayerOutputs[i];

        LastErrorFactor = errorFactor;
    }
}
