using System;
using System.Xml.Serialization;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Link
{
    [XmlIgnore]
    public Neuron Destination;

    [XmlIgnore]
    public bool Mutated;

    [XmlIgnore]
    public Neuron Origin;

    [XmlAttribute]
    public double Weight;

    [XmlIgnore]
    internal double WeightGradient;

    public Link() { }

    public Link(double weight, Neuron origin, Neuron destination)
    {
        Weight = weight;
        Origin = origin;
        Destination = destination;
    }

    public void CopyWeight(Link other) => Weight = other.Weight;

    public void MergeWeights(Link good, Link bad) => Weight = good.Weight * 0.8 + bad.Weight * 0.2;

    public void Mutate(Random random) => Mutated = Utils.MutateValue(random, ref Weight);

    public void Learn(NeuralNetwork network, double originalReward, NeuralNetwork.RewardComputation rewardFunction)
        => WeightGradient = network.ComputeRewardDifference(originalReward, ref Weight, rewardFunction);

    public void ApplyGradients(double learnRate) => Weight -= WeightGradient * learnRate;
}
