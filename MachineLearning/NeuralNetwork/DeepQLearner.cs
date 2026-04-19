using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning.NeuralNetwork;

public class DeepQLearner
{
    public NeuralNetwork Network { get; }

#if NET6_0_OR_GREATER
    public DeepQLearner(int inputCount, params int[] hiddenLayerSizes) : this(Random.Shared, inputCount, hiddenLayerSizes) { }
#endif

    public DeepQLearner(Random random, int inputCount, params int[] hiddenLayerSizes)
    {
        Network = new(random, inputCount, 1, hiddenLayerSizes);
        Network.SetOutputLayerActivationFunction(ActivationFunctionType.HyperbolicTangent);
    }

    public double EstimateReward(double[] state) => Network.ComputeOutputs(state).Single();

    public double ComputeQuality(IList<TimeStep> previousTimeSteps, double[] currentState) { }

    public void Learn(NeuralNetwork.TrainingData[] trainingData, double gain) => Network.Learn(trainingData, gain);

    public class TimeStep
    {
        public double[] State;
        public double[] Actions;
    }
}
