using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using JetBrains.Annotations;

namespace MachineLearning.NeuralNetwork;

// TODO - Add binary serialization

[Serializable]
[PublicAPI]
public class NeuralNetwork : ICloneable
{
    private const ActivationFunctionType DefaultHiddenLayerActivationFunctionType = ActivationFunctionType.RectifiedLinearUnit;
    private const ActivationFunctionType DefaultOutputLayerActivationFunctionType = ActivationFunctionType.Sigmoid;

    public static NeuralNetwork Load(string path) => Utils.LoadFromXml<NeuralNetwork>(File.ReadAllText(path));

    [XmlElement]
    public Layer[] Layers;

    public int InputCount => Layers.First().PreviousLayerNeuronCount;

    public int HiddenLayerCount => Layers.Length - 1;

    public int OutputCount => OutputLayer.NeuronCount;

    [XmlIgnore]
    public ICost CostFunction;

    [XmlAttribute]
    public CostFunctionType CostFunctionType
    {
        get => CostFunction.CostFunctionType;
        set => CostFunction = ICost.FromType(value);
    }

    public Layer[] HiddenLayers => Layers[..^1];

    public Layer OutputLayer => Layers.Last();

    private LearnData[] batchLearnData;

    public NeuralNetwork() { }

#if NET6_0_OR_GREATER
    public NeuralNetwork(int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(Random.Shared, CostFunctionType.MeanSquaredError, inputCount, outputCount, hiddenLayerSizes) { }

    public NeuralNetwork(CostFunctionType costFunctionType, int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(Random.Shared, costFunctionType, inputCount, outputCount, hiddenLayerSizes) { }
#endif

    public NeuralNetwork(Random random, int inputCount, int outputCount, params int[] hiddenLayerSizes)
        : this(random, CostFunctionType.MeanSquaredError, inputCount, outputCount, hiddenLayerSizes) { }

    public NeuralNetwork(Random random, CostFunctionType costFunctionType, int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        CostFunctionType = costFunctionType;

        InitLayers(random, inputCount, outputCount, hiddenLayerSizes);
    }

    private void InitLayers(Random random, int inputCount, int outputCount, params int[] hiddenLayerSizes)
    {
        Layers = new Layer[hiddenLayerSizes.Length + 1];

        Layers[0] = new(inputCount, hiddenLayerSizes[0], random) { ActivationFunctionType = DefaultHiddenLayerActivationFunctionType };

        for (int i = 1; i < Layers.Length - 1; i++)
            Layers[i] = new(hiddenLayerSizes[i - 1], hiddenLayerSizes[i], random) { ActivationFunctionType = DefaultHiddenLayerActivationFunctionType };

        Layers[^1] = new(hiddenLayerSizes[^1], outputCount, random) { ActivationFunctionType = DefaultOutputLayerActivationFunctionType };
    }

    public void SetHiddenLayersActivationFunction(ActivationFunctionType activationFunctionType)
    {
        foreach (Layer layer in HiddenLayers)
            layer.ActivationFunction = IActivation.FromType(activationFunctionType);
    }

    public void SetOutputLayerActivationFunction(ActivationFunctionType activationFunctionType)
        => OutputLayer.ActivationFunction = IActivation.FromType(activationFunctionType);

    public double[] ComputeOutputs(double[] inputs)
    {
        if (inputs.Length != InputCount)
            throw new ArgumentException("Inputs array has the wrong size", nameof(inputs));

        foreach (Layer layer in Layers)
            inputs = layer.ComputeOutputs(inputs);

        return inputs;
    }

    private double[] ComputeOutputs(double[] inputs, LearnData learnData)
    {
        if (inputs.Length != InputCount)
            throw new ArgumentException("Inputs array has the wrong size", nameof(inputs));

        for (int i = 0; i < Layers.Length; i++)
            inputs = Layers[i].ComputeOutputs(inputs, learnData.Layers[i]);

        return inputs;
    }

    public void Learn(TrainingData[] trainingData, double gain, ICost[] cost = null, double regularization = 0.1, double momentum = 0.9)
    {
        if (batchLearnData == null || batchLearnData.Length != trainingData.Length)
        {
            batchLearnData = new LearnData[trainingData.Length];
            for (int i = 0; i < batchLearnData.Length; i++)
                batchLearnData[i] = new(Layers);
        }

        Parallel.For(0, trainingData.Length, i => UpdateGradients(trainingData[i], batchLearnData[i], cost != null ? cost[i] : CostFunction));

        foreach (Layer layer in Layers)
            layer.ApplyGradients(gain / trainingData.Length, regularization, momentum);
    }

    public void Learn(TrainingData trainingData, double gain, ICost cost = null, double regularization = 0.1, double momentum = 0.9)
    {
        UpdateGradients(trainingData, new(Layers), cost ?? CostFunction);

        foreach (Layer layer in Layers)
            layer.ApplyGradients(gain, regularization, momentum);
    }

    private void UpdateGradients(TrainingData trainingData, LearnData learnData, ICost cost)
    {
        _ = ComputeOutputs(trainingData.Inputs, learnData);

        Layer.LearnData outputLayerLearnData = learnData.Layers.Last();
        OutputLayer.ComputeOutputLayerNeuronValues(outputLayerLearnData, trainingData.ExpectedOutputs, cost);
        OutputLayer.UpdateGradients(outputLayerLearnData);

        for (int i = HiddenLayerCount - 1; i >= 0; i--)
        {
            Layer.LearnData hiddenLayerLearnData = learnData.Layers[i];
            Layer hiddenLayer = HiddenLayers[i];

            hiddenLayer.ComputeHiddenLayerNeuronValues(hiddenLayerLearnData, Layers[i + 1], learnData.Layers[i + 1].NeuronValues);
            hiddenLayer.UpdateGradients(hiddenLayerLearnData);
        }
    }

    public void Save(string path) => File.WriteAllText(path, Utils.GetXml(this, true), Encoding.Unicode);

    public object Clone() => new NeuralNetwork
    {
        Layers = Layers.Select(l => (Layer) l.Clone()).ToArray(),
        CostFunction = CostFunction
    };

    protected bool Equals(NeuralNetwork other) => Equals(Layers, other.Layers);

    public override bool Equals(object obj)
    {
        if (obj is null)
            return false;
        if (ReferenceEquals(this, obj))
            return true;
        return obj.GetType() == GetType() && Equals((NeuralNetwork) obj);
    }

    public override int GetHashCode() => Layers != null ? Layers.GetHashCode() : 0;

    public static bool operator==(NeuralNetwork left, NeuralNetwork right) => Equals(left, right);

    public static bool operator!=(NeuralNetwork left, NeuralNetwork right) => !Equals(left, right);

    public class TrainingData
    {
        public readonly double[] Inputs;
        public readonly double[] ExpectedOutputs;
        public readonly double Reward;
        public readonly double[] NextInputs;

        public TrainingData(double[] inputs, double[] expectedOutputs, double reward, double[] nextInputs)
        {
            Inputs = inputs;
            ExpectedOutputs = expectedOutputs;
            Reward = reward;
            NextInputs = nextInputs;
        }
    }

    private class LearnData
    {
        public readonly Layer.LearnData[] Layers;

        public LearnData(Layer[] layers)
        {
            Layers = new Layer.LearnData[layers.Length];
            for (int i = 0; i < Layers.Length; i++)
                Layers[i] = new(layers[i]);
        }
    }
}
