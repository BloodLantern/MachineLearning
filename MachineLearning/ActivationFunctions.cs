using System;
using MonoGame.Utils.Extensions;

namespace MachineLearning;

[Serializable]
public delegate double ActivationFunction(double value);

public static class ActivationFunctions
{
    public static ActivationFunction GetRandom() => GetRandom(Random.Shared);

    public static ActivationFunction GetRandom(Random random) => random.Choose(Step, Sigmoid, HyperbolicTangent, RectifiedLinearUnit);

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

    public static ActivationFunction Step => value => value < 0.0 ? 0.0 : 1.0;

    public static ActivationFunction Sigmoid => value => 1.0 / (1.0 + Math.Exp(-value));

    public static ActivationFunction HyperbolicTangent => Math.Tanh;

    public static ActivationFunction RectifiedLinearUnit => value => Math.Max(0, value);
}
