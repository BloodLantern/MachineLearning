using System;

namespace MachineLearning;

// Most of the code in this file is from https://github.com/SebLague/Neural-Network-Experiments/blob/main/Assets/Scripts/Neural%20Network/Activation/Activation.cs

public enum ActivationFunctionType
{
    Linear,
    /// <summary>
    /// Output range: [0, 1]
    /// </summary>
    Sigmoid,
    /// <summary>
    /// Output range: [-1, 1]
    /// </summary>
    HyperbolicTangent,
    /// <summary>
    /// Output range: [0, 1]
    /// </summary>
    RectifiedLinearUnit,
    SigmoidLinearUnit,
    Softmax
}

public interface IActivation
{
    ActivationFunctionType ActivationFunctionType { get; }

    double ComputeActivation(double[] inputs, int index);

    double ComputeActivationDerivative(double[] inputs, int index);

    public static IActivation FromType(ActivationFunctionType type) => type switch
    {
        ActivationFunctionType.Linear => new LinearActivation(),
        ActivationFunctionType.Sigmoid => new SigmoidActivation(),
        ActivationFunctionType.HyperbolicTangent => new HyperbolicTangentActivation(),
        ActivationFunctionType.RectifiedLinearUnit => new RectifiedLinearUnitActivation(),
        ActivationFunctionType.SigmoidLinearUnit => new SigmoidLinearUnitActivation(),
        ActivationFunctionType.Softmax => new SoftmaxActivation(),
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
    };
}

public class LinearActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.Linear;

    public double ComputeActivation(double[] inputs, int index) => inputs[index];

    public double ComputeActivationDerivative(double[] inputs, int index) => 1.0;
}

public class SigmoidActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.Sigmoid;

    public double ComputeActivation(double[] inputs, int index) => 1.0 / (1.0 + Math.Exp(-inputs[index]));

    public double ComputeActivationDerivative(double[] inputs, int index)
    {
        double a = ComputeActivation(inputs, index);
        return a * (1.0 - a);
    }
}

public class HyperbolicTangentActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.HyperbolicTangent;

    public double ComputeActivation(double[] inputs, int index)
    {
        double e2 = Math.Exp(2.0 * inputs[index]);
        return (e2 - 1.0) / (e2 + 1.0);
    }

    public double ComputeActivationDerivative(double[] inputs, int index)
    {
        double e2 = Math.Exp(2.0 * inputs[index]);
        double t = (e2 - 1.0) / (e2 + 1.0);
        return 1.0 - t * t;
    }
}


public class RectifiedLinearUnitActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.RectifiedLinearUnit;

    public double ComputeActivation(double[] inputs, int index)
    {
        return Math.Max(0.0, inputs[index]);
    }

    public double ComputeActivationDerivative(double[] inputs, int index)
    {
        return inputs[index] > 0.0 ? 1.0 : 0.0;
    }
}

public class SigmoidLinearUnitActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.SigmoidLinearUnit;

    public double ComputeActivation(double[] inputs, int index)
    {
        return inputs[index] / (1.0 + Math.Exp(-inputs[index]));
    }

    public double ComputeActivationDerivative(double[] inputs, int index)
    {
        double sig = 1.0 / (1.0 + Math.Exp(-inputs[index]));
        return inputs[index] * sig * (1.0 - sig) + sig;
    }
}


public class SoftmaxActivation : IActivation
{
    public ActivationFunctionType ActivationFunctionType => ActivationFunctionType.Softmax;

    public double ComputeActivation(double[] inputs, int index)
    {
        double expSum = 0.0;
        foreach (double input in inputs)
            expSum += Math.Exp(input);

        double res = Math.Exp(inputs[index]) / expSum;

        return res;
    }

    public double ComputeActivationDerivative(double[] inputs, int index)
    {
        double expSum = 0.0;
        foreach (double input in inputs)
            expSum += Math.Exp(input);

        double ex = Math.Exp(inputs[index]);

        return (ex * expSum - ex * ex) / (expSum * expSum);
    }
}
