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

    public double EstimateReward(double[] state)
        => Network.ComputeOutputs(state, ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid).Single();

    public void LearnByGradientDescent(double[] state, double actualReward, double gain)
        => Network.LearnByGradientDescent(gain, _ => -ComputeLoss(state, actualReward));

    private double ComputeLoss(double[] state, double actualReward)
    {
        double error = EstimateReward(state) - actualReward;
        return error * error;
    }

    private double ComputeLossDerivative(double[] state, double actualReward) => 2.0 * (EstimateReward(state) - actualReward);

    private double SigmoidDerivative(double value)
    {
        double activation = ActivationFunctions.Sigmoid(value);
        return activation * (1.0 - activation);
    }
}
