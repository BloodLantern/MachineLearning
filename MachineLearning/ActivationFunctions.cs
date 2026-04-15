using System;
using System.Collections.Generic;

namespace MachineLearning;

[Serializable]
public delegate double ActivationFunction(double value);

public static class ActivationFunctions
{
    private static readonly Random randomInstance = new();
    public static ActivationFunction GetRandom() => GetRandom(randomInstance);

    public static ActivationFunction GetRandom(Random random) => Functions[random.Next(Functions.Count)];

    public static ActivationFunction GetFromType(ActivationFunctionType type) => type switch
    {
        ActivationFunctionType.Step => Step,
        ActivationFunctionType.Sigmoid => Sigmoid,
        ActivationFunctionType.HyperbolicTangent => HyperbolicTangent,
        ActivationFunctionType.RectifiedLinearUnit => RectifiedLinearUnit,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
    };

    public static ActivationFunctionType GetTypeFrom(ActivationFunction activationFunction)
    {
        if (activationFunction == Step)
            return ActivationFunctionType.Step;
        if (activationFunction == Sigmoid)
            return ActivationFunctionType.Sigmoid;
        if (activationFunction == HyperbolicTangent)
            return ActivationFunctionType.HyperbolicTangent;
        if (activationFunction == RectifiedLinearUnit)
            return ActivationFunctionType.RectifiedLinearUnit;
        throw new ArgumentException(null, nameof(activationFunction));
    }

    /// <summary>
    /// Output range: [0, 1]
    /// </summary>
    public static readonly ActivationFunction Step = value => value < 0.0 ? 0.0 : 1.0;

    /// <summary>
    /// Output range: [0, 1]
    /// </summary>
    public static readonly ActivationFunction Sigmoid = value => 1.0 / (1.0 + Math.Exp(-value));

    /// <summary>
    /// Output range: [-1, 1]
    /// </summary>
    public static readonly ActivationFunction HyperbolicTangent = Math.Tanh;

    /// <summary>
    /// Output range: [0, 1]
    /// </summary>
    public static readonly ActivationFunction RectifiedLinearUnit = value => Math.Max(0, value);

    private static readonly ActivationFunction[] functions = [Step, Sigmoid, HyperbolicTangent, RectifiedLinearUnit];

    public static IReadOnlyList<ActivationFunction> Functions => functions;
}
