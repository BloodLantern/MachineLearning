using System;
using System.Linq;
using JetBrains.Annotations;

namespace MachineLearning.Models.NeuralNetwork;

public class QLearner
{
    public NeuralNetwork Network { get; }

#if NET6_0_OR_GREATER
    public QLearner(int inputCount, params int[] hiddenLayerSizes) : this(Random.Shared, inputCount, hiddenLayerSizes) { }
#endif

    public QLearner(Random random, int inputCount, params int[] hiddenLayerSizes)
        => Network = new(random, inputCount, 1, hiddenLayerSizes);

    public double EstimateQuality(double[] state)
        => Network.ComputeOutputs(state, ActivationFunctions.RectifiedLinearUnit, ActivationFunctions.Sigmoid).Single();

    public void LearnByGradientDescent(double[] state, double actualReward, double gain)
        => Network.LearnByGradientDescent(gain, _ => -ComputeLoss(state, actualReward));

    private double ComputeLoss(double[] state, double actualReward)
    {
        double error = actualReward - EstimateQuality(state);
        return error * error;
    }
}
