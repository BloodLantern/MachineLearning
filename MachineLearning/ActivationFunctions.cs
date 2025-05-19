using System;
using MonoGame.Utils.Extensions;

namespace MachineLearning;

[Serializable]
public delegate double ActivationFunction(double value);

public static class ActivationFunctions
{
    public static ActivationFunction Step => value => value < 0.0 ? 0.0 : 1.0;
    
    public static ActivationFunction Sigmoid => value => 1.0 / (1.0 + Math.Exp(-value));

    public static ActivationFunction HyperbolicTangent => Math.Tanh;

    public static ActivationFunction RectifiedLinearUnit => value => Math.Max(0, value);

    public static ActivationFunction GetRandom() => GetRandom(Random.Shared);
    
    public static ActivationFunction GetRandom(Random random) => random.Choose(Step, Sigmoid, HyperbolicTangent, RectifiedLinearUnit);
}
