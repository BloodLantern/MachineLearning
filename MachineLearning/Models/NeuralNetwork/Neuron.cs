using System;
using System.Collections.Generic;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    [XmlIgnore]
    public double Value;
    
    public double Bias;

    [XmlIgnore]
    internal double BiasGradient;
    
    public Link[] Links;
    
    public Neuron()
    {
    }

    public Neuron(double value) => Value = value;

    public void InitLinks(IReadOnlyList<Neuron> previousNeurons, Random random)
    {
        Links = new Link[previousNeurons.Count];
        
        // Set the weights randomly between -1 and +1
        for (int i = 0; i < previousNeurons.Count; i++)
            Links[i] = new((random.NextDouble() * 2.0 - 1.0) / Math.Sqrt(previousNeurons.Count), previousNeurons[i], this);
    }

    public void CopyLinks(IReadOnlyList<Link> links)
    {
        for (int i = 0; i < links.Count; i++)
            Links[i].CopyWeight(links[i]);
    }

    public void MergeLinks(IReadOnlyList<Link> goodLinks, IReadOnlyList<Link> badLinks)
    {
        if (goodLinks.Count != badLinks.Count)
            throw new ArgumentException("Cannot merge weight arrays of different sizes");

        for (int i = 0; i < goodLinks.Count; i++)
            Links[i].MergeWeights(goodLinks[i], badLinks[i]);
    }

    public void FeedForward()
    {
        double value = Bias;

        foreach (Link link in Links)
            value += link.Weight * link.Origin.Value;

        Value = Utils.Sigmoid(value);
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

    public void ApplyGradients(double learnRate)
    {
        foreach (Link link in Links)
            link.ApplyGradients(learnRate);

        if (BiasGradient > 0.0)
            Bias += BiasGradient * learnRate;
    }
}
