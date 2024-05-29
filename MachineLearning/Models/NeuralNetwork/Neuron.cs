using System;

namespace MachineLearning.Models.NeuralNetwork;

[Serializable]
public class Neuron
{
    public float Value;
    public float[] Weights;

    private readonly Random random = new();

    public Neuron()
    {
    }

    public Neuron(float value) => Value = value;

    public void InitWeights(int neuronsInPreviousLayer)
    {
        Weights = new float[neuronsInPreviousLayer];
        
        // Set the weights randomly between -1 and 1
        for (int k = 0; k < neuronsInPreviousLayer; k++)
            Weights[k] = random.NextSingle() - 0.5f;
    }

    public void CopyWeights(float[] weights)
    {
        for (int i = 0; i < weights.Length; i++)
            Weights[i] = weights[i];
    }

    public void Mutate()
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            float weight = Weights[i];

            float randomNumber = random.NextSingle() * 1000f;

            switch (randomNumber)
            {
                case <= 2f:
                    weight *= -1f;
                    break;
                case <= 4f:
                    weight = random.NextSingle() - 0.5f;
                    break;
                case <= 6f:
                    weight *= random.NextSingle() + 1f;
                    break;
                case <= 8f:
                    weight *= random.NextSingle();
                    break;
            }

            Weights[i] = weight;
        }
    }

    public static implicit operator float(Neuron neuron) => neuron.Value;
}
