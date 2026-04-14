using System;
using System.Linq;

namespace MachineLearning.Models.NeuralNetwork;

public class QLearner
{
    private readonly NeuralNetwork network;

#if NET6_0_OR_GREATER
    public QLearner(int inputCount, params int[] hiddenLayerSizes) : this(Random.Shared, inputCount, hiddenLayerSizes) { }
#endif

    public QLearner(Random random, int inputCount, params int[] hiddenLayerSizes)
        => network = new(random, inputCount, 1, hiddenLayerSizes);

    public double EstimateQuality(double[] state)
        => network.ComputeOutputs(state, ActivationFunctions.RectifiedLinearUnit, ActivationFunctions.Sigmoid).Single();

    public void LearnByGradientDescent(double[] state, double actualReward, double gain)
        => network.LearnByGradientDescent(gain, _ => ComputeFitness(state, actualReward));

    private double ComputeFitness(double[] state, double actualReward)
    {
        double error = actualReward - EstimateQuality(state);
        return error * error;
    }
}
