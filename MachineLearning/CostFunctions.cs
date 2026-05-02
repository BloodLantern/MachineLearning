using System;

namespace MachineLearning;

// Most of the code in this file is from https://github.com/SebLague/Neural-Network-Experiments/blob/main/Assets/Scripts/Neural%20Network/Cost/Cost.cs

public enum CostFunctionType
{
    MeanSquaredError,
    CrossEntropy
}

public interface ICost
{
    CostFunctionType CostFunctionType { get; }

    double ComputeCost(double[] predictedOutputs, double[] expectedOutputs);

    double[] ComputeCostDerivative(double[] predictedOutputs, double[] expectedOutputs);

    public static ICost FromType(CostFunctionType type) => type switch
    {
        CostFunctionType.MeanSquaredError => new MeanSquaredErrorCost(),
        CostFunctionType.CrossEntropy => new CrossEntropyCost(),
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
    };
}

public class MeanSquaredErrorCost : ICost
{
    public CostFunctionType CostFunctionType => CostFunctionType.MeanSquaredError;

    public double ComputeCost(double[] predictedOutputs, double[] expectedOutputs)
    {
        // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
        double cost = 0.0;

        for (int i = 0; i < predictedOutputs.Length; i++)
        {
            double error = predictedOutputs[i] - expectedOutputs[i];
            cost += error * error;
        }

        return 0.5 * cost;
    }

    public double[] ComputeCostDerivative(double[] predictedOutputs, double[] expectedOutputs)
    {
        double[] results = new double[predictedOutputs.Length];
        for (int i = 0; i < results.Length; i++)
            results[i] = ComputeCostDerivative(predictedOutputs[i], expectedOutputs[i]);
        return results;
    }

    public double ComputeCostDerivative(double predictedOutput, double expectedOutput) => predictedOutput - expectedOutput;
}

public class CrossEntropyCost : ICost
{
    public CostFunctionType CostFunctionType => CostFunctionType.CrossEntropy;

    public double ComputeCost(double[] predictedOutputs, double[] expectedOutputs)
    {
        double cost = 0.0;

        for (int i = 0; i < predictedOutputs.Length; i++)
        {
            double x = predictedOutputs[i];
            double y = expectedOutputs[i];
            double v = Utils.Approximately(y, 1.0) ? -Math.Log(x) : -Math.Log(1.0 - x);
            cost += double.IsNaN(v) ? 0.0 : v;
        }

        return cost;
    }

    public double[] ComputeCostDerivative(double[] predictedOutputs, double[] expectedOutputs)
    {
        double[] results = new double[predictedOutputs.Length];
        for (int i = 0; i < results.Length; i++)
            results[i] = ComputeCostDerivative(predictedOutputs[i], expectedOutputs[i]);
        return results;
    }

    public double ComputeCostDerivative(double predictedOutput, double expectedOutput)
    {
        if (predictedOutput == 0.0 || Utils.Approximately(predictedOutput, 1.0))
            return 0.0;

        return (-predictedOutput + expectedOutput) / (predictedOutput * (predictedOutput - 1.0));
    }
}
