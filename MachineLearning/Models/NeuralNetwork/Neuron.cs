using System;
using System.Collections.Generic;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    public double Value;
    
    public Link[] Links;
    
    public Neuron()
    {
    }

    public Neuron(double value) => Value = value;

    public void InitLinks(IReadOnlyList<Neuron> previousNeurons, Random random)
    {
        Links = new Link[previousNeurons.Count];
        
        // Set the weights randomly between -0.5 and 0.5
        for (int i = 0; i < previousNeurons.Count; i++)
            Links[i] = new(random.NextDouble() - 0.5, previousNeurons[i], this);
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

    public void Mutate(Random random)
    {
        foreach (Link link in Links)
            link.Mutate(random);
    }
}
