using System;
using System.Collections.Generic;
using System.IO;

namespace MachineLearning;

public class NeuralNetwork : IComparable<NeuralNetwork>
{
    private readonly int[] layers;
    private float[][] neurons;
    private float[][][] weights;
    [NonSerialized]
    public float Fitness = 0;

    [NonSerialized]
    public int Rank;

    [NonSerialized]
    private readonly Random random = new();

    public NeuralNetwork(int[] layers)
    {
        this.layers = new int[layers.Length];

        for (int i = 0; i < layers.Length; i++)
            this.layers[i] = layers[i];

        InitNeurons();
        InitWeights();
    }

    public NeuralNetwork(NeuralNetwork copy)
    {
        layers = new int[copy.layers.Length];

        for (int i = 0; i < copy.layers.Length; i++)
            layers[i] = copy.layers[i];

        InitNeurons();
        InitWeights();
        CopyWeights(copy.weights);
    }

    private void InitNeurons()
    {
        List<float[]> neuronsList = new();

        foreach (int layer in layers)
            neuronsList.Add(new float[layer]);

        neurons = neuronsList.ToArray();
    }

    private void InitWeights()
    {
        List<float[][]> weightsList = new();

        for (int i = 1; i < layers.Length; i++)
        {
            List<float[]> layerWeightList = new();

            int neuronsInPreviousLayer = layers[i - 1];

            for (int j = 0; j < neurons[i].Length; j++)
            {
                float[] neuronWeights = new float[neuronsInPreviousLayer];

                // Set the weights randomly between -1 and 1
                for (int k = 0; k < neuronsInPreviousLayer; k++)
                    neuronWeights[k] = random.NextSingle() - 0.5f;

                layerWeightList.Add(neuronWeights);
            }

            weightsList.Add(layerWeightList.ToArray());
        }

        weights = weightsList.ToArray();
    }

    private void CopyWeights(float[][][] copyWeights)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].Length; j++)
            {
                for (int k = 0; k < weights[i][j].Length; k++)
                    weights[i][j][k] = copyWeights[i][j][k];
            }
        }
    }

    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < inputs.Length; i++)
            neurons[0][i] = inputs[i];

        for (int i = 1; i < layers.Length; i++)
        {
            for (int j = 0; j < neurons[i].Length; j++)
            {
                float value = 0.25f;

                for (int k = 0; k < neurons[i - 1].Length; k++)
                    value += weights[i - 1][j][k] * neurons[i - 1][k];

                neurons[i][j] = MathF.Tanh(value);
            }
        }

        return neurons[^1];
    }

    public void Mutate()
    {
        foreach (float[][] weightMatrix in weights)
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
