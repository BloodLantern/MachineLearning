using System;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Link
{
    [XmlAttribute]
    public double Weight;
    
    [XmlIgnore]
    public Neuron Origin;
    [XmlIgnore]
    public Neuron Destination;

    [XmlIgnore]
    public bool Mutated;

    public Link()
    {
    }

    public Link(double weight, Neuron origin, Neuron destination)
    {
        Weight = weight;
        Origin = origin;
        Destination = destination;
    }

    public void CopyWeight(Link other) => Weight = other.Weight;

    public void MergeWeights(Link good, Link bad) => Weight = good.Weight * 0.8 + bad.Weight * 0.2;

    public void Mutate(Random random)
    {
        double oldWeight = Weight;
        
        switch (random.NextDouble() * 1000.0)
        {
            case <= 2.0:
                Weight *= -1.0;
                break;
            case <= 4.0:
                Weight = random.NextDouble() - 0.5;
                break;
            case <= 6.0:
                Weight *= random.NextDouble() + 1.0;
                break;
            case <= 8.0:
                Weight *= random.NextDouble();
                break;
        }

        Mutated = Math.Abs(oldWeight - Weight) != 0.0;
    }
}
