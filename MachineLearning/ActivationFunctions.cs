using System;
using MonoGame.Utils.Extensions;

namespace MachineLearning;

[Serializable]
public delegate double ActivationFunction(double value);

public static class ActivationFunctions
{
    public enum Type
    {
        Step,
        Sigmoid,
        HyperbolicTangent,
        RectifiedLinearUnit
    }

    public static ActivationFunction Step => value => value < 0.0 ? 0.0 : 1.0;

    public static ActivationFunction Sigmoid => value => 1.0 / (1.0 + Math.Exp(-value));

    public static ActivationFunction HyperbolicTangent => Math.Tanh;

    public static ActivationFunction RectifiedLinearUnit => value => Math.Max(0, value);

    public static ActivationFunction GetRandom() => GetRandom(Random.Shared);

    public static ActivationFunction GetRandom(Random random) => random.Choose(Step, Sigmoid, HyperbolicTangent, RectifiedLinearUnit);

    public static ActivationFunction GetFromType(Type type)
    {
        return type switch
        {
            Type.Step => Step,
            Type.Sigmoid => Sigmoid,
            Type.HyperbolicTangent => HyperbolicTangent,
            Type.RectifiedLinearUnit => RectifiedLinearUnit,
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
        };
    }

    public static Type GetTypeFrom(ActivationFunction activationFunction)
    {
        if (activationFunction == Step)
            return Type.Step;
        if (activationFunction == Sigmoid)
            return Type.Sigmoid;
        if (activationFunction == HyperbolicTangent)
            return Type.HyperbolicTangent;
        if (activationFunction == RectifiedLinearUnit)
            return Type.RectifiedLinearUnit;
        throw new ArgumentException(null, nameof(activationFunction));
    }
}
