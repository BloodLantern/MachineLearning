using System;
using System.Diagnostics;
using System.Linq;
using System.Xml.Serialization;
using JetBrains.Annotations;

namespace MachineLearning.NeuralNetwork;

[Serializable]
[PublicAPI]
public class QNetwork
{
    /// <summary>
    /// The current neural network used to compute the action qualities.
    /// </summary>
    [XmlElement]
    public NeuralNetwork Online { get; set; }

    /// <summary>
    /// A previous version of the <see cref="Online"/> network used to compute the cost.
    /// </summary>
    [XmlIgnore]
    public NeuralNetwork Target { get; private set; }

    public int InputCount => Online.InputCount;

    public int HiddenLayerCount => Online.HiddenLayerCount;

    public int OutputNeuronCount => Online.OutputCount;

    [XmlAttribute]
    public double DiscountFactor = 0.99;

    [XmlAttribute]
    public double ExplorationProbability = 0.95;

    public QNetwork() { }

    public QNetwork(Random random, int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        Online = new(random, CostFunctionType.Huber, inputCount, outputCount, hiddenLayerSizes);
        Online.SetOutputLayerActivationFunction(ActivationFunctionType.Linear);
        UpdateTargetNetwork();
    }

#if NET6_0_OR_GREATER
    public QNetwork(int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(Random.Shared, inputCount, outputCount, hiddenLayerSizes) { }
#endif

    public bool ShouldExplore(Random random) => random.NextDouble() <= ExplorationProbability;

#if NET6_0_OR_GREATER
    public bool ShouldExplore() => ShouldExplore(Random.Shared);
#endif

    public double[] ComputeActionQualities(double[] state) => Online.ComputeOutputs(state);

    public double[] ComputeExplorationQualities(Random random)
    {
        double[] result = new double[Online.OutputCount];
        for (int i = 0; i < result.Length; i++)
            result[i] = random.NextDouble() - 0.5;
        return result;
    }

#if NET6_0_OR_GREATER
    public double[] ComputeExplorationQualities() => ComputeExplorationQualities(Random.Shared);
#endif

    public bool[] ChooseActions(double[] state, Random random)
        => (ShouldExplore(random) ? ComputeExplorationQualities(random) : ComputeActionQualities(state)).Select(IsActionChosen).ToArray();

#if NET6_0_OR_GREATER
    public bool[] ChooseActions(double[] state) => ChooseActions(state, Random.Shared);
#endif

    public static bool IsActionChosen(double quality) => quality > 0.0;

    public void UpdateTargetNetwork() => Target = (NeuralNetwork) Online.Clone();

    public void Learn(NeuralNetwork.TrainingData[] trainingData, double gain)
    {
        ICost[] costs = new ICost[trainingData.Length];
        for (int i = 0; i < costs.Length; i++)
        {
            Cost cost = new(this);
            cost.UpdateTargetResults(trainingData[i]);
            costs[i] = cost;
        }

        Online.Learn(trainingData, gain, costs);
    }

    private class Cost : ICost
    {
        private readonly QNetwork qNetwork;

        private readonly HuberCost cost = new();

        private double[] targetResults;

        public Cost(QNetwork qNetwork) => this.qNetwork = qNetwork;

        public void UpdateTargetResults(NeuralNetwork.TrainingData trainingData)
            => targetResults = ComputeTargetResults(qNetwork, trainingData);

        public static double[] ComputeTargetResults(QNetwork qNetwork, NeuralNetwork.TrainingData trainingData)
        {
            double[] onlineActions = qNetwork.Online.ComputeOutputs(trainingData.Inputs);
            double[] targetActions = qNetwork.Target.ComputeOutputs(trainingData.NextInputs);
            double[] results = new double[targetActions.Length];

            for (int i = 0; i < targetActions.Length; i++)
                results[i] = IsActionChosen(trainingData.ExpectedOutputs[i]) ? trainingData.Reward + qNetwork.DiscountFactor * targetActions[i] : onlineActions[i];

            return results;
        }

        CostFunctionType ICost.CostFunctionType => CostFunctionType.Huber;

        // Here predictedOutputs are the outputs of the target network
        public double ComputeCost(double[] predictedOutputs, double[] expectedOutputs)
        {
            Debug.Assert(targetResults != null);
            return cost.ComputeCost(predictedOutputs, targetResults);
        }

        public double[] ComputeCostDerivative(double[] predictedOutputs, double[] expectedOutputs)
        {
            Debug.Assert(targetResults != null);
            return cost.ComputeCostDerivative(predictedOutputs, targetResults);
        }
    }
}
