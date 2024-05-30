using System;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    public double Value;
    
    public double[] Weights;

    public Neuron()
    {
    }

    public Neuron(double value) => Value = value;

    public void InitWeights(int neuronsInPreviousLayer, Random random)
    {
        Weights = new double[neuronsInPreviousLayer];
        
        // Set the weights randomly between -1 and 1
        for (int k = 0; k < neuronsInPreviousLayer; k++)
            Weights[k] = random.NextSingle() - 0.5;
    }

    public void CopyWeights(double[] weights)
    {
        for (int i = 0; i < weights.Length; i++)
            Weights[i] = weights[i];
    }

    public void Mutate(Random random)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            double weight = Weights[i];

            double randomNumber = random.NextDouble() * 1000.0;

            switch (randomNumber)
            {
                case <= 2.0:
                    weight *= -1.0;
                    break;
                case <= 4.0:
                    weight = random.NextDouble() - 0.5;
                    break;
                case <= 6.0:
                    weight *= random.NextDouble() + 1.0;
                    break;
                case <= 8.0:
                    weight *= random.NextDouble();
                    break;
            }

            Weights[i] = weight;
        }
    }
}
