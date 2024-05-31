using System;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Link
{
    public double Weight;
    
    public Neuron Origin { get; private set; }
    public Neuron Destination { get; private set; }

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

    public void MergeWeights(Link good, Link bad) => Weight = good.Weight * 0.6 + bad.Weight * 0.4;

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
