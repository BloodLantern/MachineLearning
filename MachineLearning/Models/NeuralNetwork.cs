using System;
using System.Collections.Generic;
using System.IO;

namespace MachineLearning;

[Serializable]
public class NeuralNetwork : IComparable<NeuralNetwork>
{
    public int[] Layers { get; set; }
    public float[][] Neurons { get; set; }
    public float[][][] Weights { get; set; }
    
    [NonSerialized]
    public float Fitness = 0;

    [NonSerialized]
    public int Rank;

    private readonly Random random = new();

    public NeuralNetwork()
    {
    }

    public NeuralNetwork(int[] layers)
    {
        Layers = new int[layers.Length];

        for (int i = 0; i < layers.Length; i++)
            Layers[i] = layers[i];

        InitNeurons();
        InitWeights();
    }

    public NeuralNetwork(NeuralNetwork copy)
    {
        Layers = new int[copy.Layers.Length];

        for (int i = 0; i < copy.Layers.Length; i++)
            Layers[i] = copy.Layers[i];

        InitNeurons();
        InitWeights();
        CopyWeights(copy.Weights);
    }

    private void InitNeurons()
    {
        List<float[]> neuronsList = [];

        foreach (int layer in Layers)
            neuronsList.Add(new float[layer]);

        Neurons = neuronsList.ToArray();
    }

    private void InitWeights()
    {
        List<float[][]> weightsList = [];

        for (int i = 1; i < Layers.Length; i++)
        {
            List<float[]> layerWeightList = [];

            int neuronsInPreviousLayer = Layers[i - 1];

            for (int j = 0; j < Neurons[i].Length; j++)
            {
                float[] neuronWeights = new float[neuronsInPreviousLayer];

                // Set the weights randomly between -1 and 1
                for (int k = 0; k < neuronsInPreviousLayer; k++)
                    neuronWeights[k] = random.NextSingle() - 0.5f;

                layerWeightList.Add(neuronWeights);
            }

            weightsList.Add(layerWeightList.ToArray());
        }

        Weights = weightsList.ToArray();
    }

    private void CopyWeights(float[][][] copyWeights)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            for (int j = 0; j < Weights[i].Length; j++)
            {
                for (int k = 0; k < Weights[i][j].Length; k++)
                    Weights[i][j][k] = copyWeights[i][j][k];
            }
        }
    }

    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < inputs.Length; i++)
            Neurons[0][i] = inputs[i];

        for (int i = 1; i < Layers.Length; i++)
        {
            for (int j = 0; j < Neurons[i].Length; j++)
            {
                float value = 0.25f;

                for (int k = 0; k < Neurons[i - 1].Length; k++)
                    value += Weights[i - 1][j][k] * Neurons[i - 1][k];

                Neurons[i][j] = MathF.Tanh(value);
            }
        }

        return Neurons[^1];
    }

    public void Mutate()
    {
        foreach (float[][] weightMatrix in Weights)
        {
            foreach (float[] weightRow in weightMatrix)
            {
                for (int i = 0; i < weightRow.Length; i++)
                {
                    float weight = weightRow[i];

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

                    weightRow[i] = weight;
                }
            }
        }
    }

    public void ResetNeurons() => InitNeurons();

    public void Save(string path) => File.WriteAllText(path, this.GetXml(true));

    public static NeuralNetwork Load(string path) => File.ReadAllText(path).LoadFromXml<NeuralNetwork>();

    public int CompareTo(NeuralNetwork other)
    {
        if (other == null)
            return -1;

        if (Fitness > other.Fitness)
            return -1;
            
        return Fitness < other.Fitness ? 1 : 0;
    }
}
