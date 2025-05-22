using System;
using System.Linq;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    [XmlIgnore]
    public double Output;

    public double Bias = 0.1;

    [XmlIgnore]
    internal double BiasGradient;

    public Link[] Links;

    public int InputCount => Links.Length;

    public double[] Weights => Links.Select(l => l.Weight).ToArray();

    [XmlIgnore]
    public double LastErrorFactor { get; private set; }

    public Neuron()
    {
    }

    public Neuron(double output) => Output = output;

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

    public void Learn(NeuralNetwork network, double originalFitness)
    {
        foreach (Link link in Links)
            link.Learn(network, originalFitness);

        BiasGradient = network.ComputeFitnessDifference(originalFitness, ref Bias);
    }

    public void ApplyGradients(double gain)
    {
        foreach (Link link in Links)
            link.ApplyGradients(gain);

        if (BiasGradient > 0.0)
            Bias += BiasGradient * gain;
    }

    public void LearnByBackpropagation(double errorFactor, double gain, double[] previousLayerOutputs)
    {
        for (int i = 0; i < InputCount; i++)
            Links[i].Weight += gain * errorFactor * previousLayerOutputs[i];

        LastErrorFactor = errorFactor;
    }
}
