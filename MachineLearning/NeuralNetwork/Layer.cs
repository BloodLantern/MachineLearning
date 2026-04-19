using System;
using System.Xml.Serialization;

namespace MachineLearning.NeuralNetwork;

[Serializable]
public class Layer : ICloneable
{
    public int PreviousLayerNeuronCount;
    public int NeuronCount;

    public double[] Weights;
    public double[] Biases;

    // Cost gradient with respect to weights and with respect to biases
    public double[] WeightCostGradients;
    public double[] BiasCostGradients;

    // Used for adding momentum to gradient descent
    public double[] WeightVelocities;
    public double[] BiasVelocities;

    [XmlIgnore]
    public IActivation ActivationFunction = IActivation.FromType(ActivationFunctionType.Sigmoid);

    [XmlAttribute(nameof(ActivationFunction))]
    public ActivationFunctionType ActivationFunctionType
    {
        get => ActivationFunction.ActivationFunctionType;
        set => ActivationFunction = IActivation.FromType(value);
    }

    public Layer() { }

    public Layer(int previousLayerSize, int size, Random random)
    {
        if (size < 1)
            throw new ArgumentException($"A {nameof(NeuralNetwork)} {nameof(Layer)} must have at least 1 neuron");

        PreviousLayerNeuronCount = previousLayerSize;
        NeuronCount = size;

        Weights = new double[PreviousLayerNeuronCount * NeuronCount];
        WeightCostGradients = new double[Weights.Length];
        Biases = new double[NeuronCount];
        BiasCostGradients = new double[Biases.Length];

        WeightVelocities = new double[Weights.Length];
        BiasVelocities = new double[Biases.Length];

        InitializeRandomWeights(random);
    }

    private void InitializeRandomWeights(Random random)
    {
        double sqrt = Math.Sqrt(PreviousLayerNeuronCount);

        for (int i = 0; i < Weights.Length; i++)
            Weights[i] = RandomInNormalDistribution(0.0, 1.0) / sqrt;

        return;

        double RandomInNormalDistribution(double mean, double standardDeviation)
        {
            double x1 = 1.0 - random.NextDouble();
            double x2 = 1.0 - random.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }

    public double[] ComputeOutputs(double[] inputs)
    {
        double[] weightedInputs = new double[NeuronCount];

        for (int neuronIndex = 0; neuronIndex < NeuronCount; neuronIndex++)
        {
            double weightedInput = Biases[neuronIndex];

            for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < PreviousLayerNeuronCount; previousLayerNeuronIndex++)
                weightedInput += inputs[previousLayerNeuronIndex] * GetWeight(previousLayerNeuronIndex, neuronIndex);

            weightedInputs[neuronIndex] = weightedInput;
        }

        double[] activations = new double[NeuronCount];

        for (int outputNode = 0; outputNode < NeuronCount; outputNode++)
            activations[outputNode] = ActivationFunction.ComputeActivation(weightedInputs, outputNode);

        return activations;
    }

    internal double[] ComputeOutputs(double[] inputs, LearnData learnData)
    {
        learnData.Inputs = inputs;

        for (int neuronIndex = 0; neuronIndex < NeuronCount; neuronIndex++)
        {
            double weightedInput = Biases[neuronIndex];

            for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < PreviousLayerNeuronCount; previousLayerNeuronIndex++)
                weightedInput += inputs[previousLayerNeuronIndex] * GetWeight(previousLayerNeuronIndex, neuronIndex);

            learnData.WeightedInputs[neuronIndex] = weightedInput;
        }

        for (int outputNode = 0; outputNode < NeuronCount; outputNode++)
            learnData.Activations[outputNode] = ActivationFunction.ComputeActivation(learnData.WeightedInputs, outputNode);

        return learnData.Activations;
    }

    // Update weights and biases based on previously calculated gradients.
    // Also resets the gradients to zero.
    internal void ApplyGradients(double gain, double regularization, double momentum)
    {
        double weightDecay = 1.0 - regularization * gain;

        for (int i = 0; i < Weights.Length; i++)
        {
            double weight = Weights[i];
            double velocity = WeightVelocities[i] * momentum - WeightCostGradients[i] * gain;
            WeightVelocities[i] = velocity;
            Weights[i] = weight * weightDecay + velocity;
            WeightCostGradients[i] = 0.0;
        }

        for (int i = 0; i < Biases.Length; i++)
        {
            double velocity = BiasVelocities[i] * momentum - BiasCostGradients[i] * gain;
            BiasVelocities[i] = velocity;
            Biases[i] += velocity;
            BiasCostGradients[i] = 0.0;
        }
    }

    // Calculate the "node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    internal void ComputeOutputLayerNeuronValues(LearnData learnData, double[] expectedOutputs, ICost costFunction)
    {
        for (int i = 0; i < learnData.NeuronValues.Length; i++)
        {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            double costDerivative = costFunction.ComputeCostDerivative(learnData.Activations[i], expectedOutputs[i]);
            double activationDerivative = ActivationFunction.ComputeActivationDerivative(learnData.WeightedInputs, i);
            learnData.NeuronValues[i] = costDerivative * activationDerivative;
        }
    }

    // Calculate the "node values" for a hidden layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    internal void ComputeHiddenLayerNeuronValues(LearnData learnData, Layer oldLayer, double[] oldNeuronValues)
    {
        for (int newNeuronIndex = 0; newNeuronIndex < NeuronCount; newNeuronIndex++)
        {
            double newNeuronValue = 0.0;

            for (int oldNeuronIndex = 0; oldNeuronIndex < oldNeuronValues.Length; oldNeuronIndex++)
            {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldLayer.GetWeight(newNeuronIndex, oldNeuronIndex);
                newNeuronValue += weightedInputDerivative * oldNeuronValues[oldNeuronIndex];
            }

            newNeuronValue *= ActivationFunction.ComputeActivationDerivative(learnData.WeightedInputs, newNeuronIndex);
            learnData.NeuronValues[newNeuronIndex] = newNeuronValue;
        }
    }

    internal void UpdateGradients(LearnData learnData)
    {
        lock (WeightCostGradients)
        {
            for (int neuronIndex = 0; neuronIndex < NeuronCount; neuronIndex++)
            {
                double neuronValue = learnData.NeuronValues[neuronIndex];

                for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < PreviousLayerNeuronCount; previousLayerNeuronIndex++)
                {
                    // Evaluate the partial derivative: cost / weight of current connection
                    double derivativeWeightCost = learnData.Inputs[previousLayerNeuronIndex] * neuronValue;
                    // The WeightCostGradients array stores these partial derivatives for each weight.
                    // Note: the derivative is being added to the array here because ultimately we want
                    // to calculate the average gradient across all the data in the training batch
                    WeightCostGradients[GetWeightIndex(previousLayerNeuronIndex, neuronIndex)] += derivativeWeightCost;
                }
            }
        }

        lock (BiasCostGradients)
        {
            for (int neuronIndex = 0; neuronIndex < NeuronCount; neuronIndex++)
            {
                // Evaluate partial derivative: cost / bias
                double derivativeBiasCost = learnData.NeuronValues[neuronIndex];
                BiasCostGradients[neuronIndex] += derivativeBiasCost;
            }
        }
    }

    public double GetWeight(int previousLayerNeuronIndex, int thisLayerNeuronIndex)
        => Weights[GetWeightIndex(previousLayerNeuronIndex, thisLayerNeuronIndex)];

    public int GetWeightIndex(int previousLayerNeuronIndex, int thisLayerNeuronIndex)
        => thisLayerNeuronIndex * PreviousLayerNeuronCount + previousLayerNeuronIndex;

    public object Clone() => new Layer
    {
        PreviousLayerNeuronCount = PreviousLayerNeuronCount,
        NeuronCount = NeuronCount,
        Weights = (double[]) Weights.Clone(),
        Biases = (double[]) Biases.Clone(),
        WeightCostGradients = (double[]) WeightCostGradients.Clone(),
        BiasCostGradients = (double[]) BiasCostGradients.Clone(),
        WeightVelocities = (double[]) WeightVelocities.Clone(),
        BiasVelocities = (double[]) BiasVelocities.Clone(),
        ActivationFunction = ActivationFunction
    };

    protected bool Equals(Layer other) => Equals(Weights, other.Weights) && Equals(Biases, other.Biases);

    public override bool Equals(object obj)
    {
        if (obj is null)
            return false;
        if (ReferenceEquals(this, obj))
            return true;
        if (obj.GetType() != GetType())
            return false;
        return Equals((Layer) obj);
    }

    public override int GetHashCode() => HashCode.Combine(Weights, Biases);

    public static bool operator==(Layer left, Layer right) => Equals(left, right);

    public static bool operator!=(Layer left, Layer right) => !Equals(left, right);

    internal class LearnData
    {
        public double[] Inputs;
        public readonly double[] WeightedInputs;
        public readonly double[] Activations;
        public readonly double[] NeuronValues;

        public LearnData(Layer layer)
        {
            Inputs = new double[layer.PreviousLayerNeuronCount];
            WeightedInputs = new double[layer.NeuronCount];
            Activations = new double[layer.NeuronCount];
            NeuronValues = new double[layer.NeuronCount];
        }
    }
}
