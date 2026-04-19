using System;
using System.Linq;

namespace MachineLearning.NeuralNetwork;

public class QLearner
{
    public NeuralNetwork Network { get; }

#if NET6_0_OR_GREATER
    public QLearner(int inputCount, params int[] hiddenLayerSizes) : this(Random.Shared, inputCount, hiddenLayerSizes) { }
#endif

    public QLearner(Random random, int inputCount, params int[] hiddenLayerSizes)
        => Network = new(random, inputCount, 1, hiddenLayerSizes);

    public double EstimateReward(double[] state) => Network.ComputeOutputs(state).Single() * 2.0 - 1.0;

    public void Learn(NeuralNetwork.TrainingData[] trainingData, double gain) => Network.Learn(trainingData, gain);
}
