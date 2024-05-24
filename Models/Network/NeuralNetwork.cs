using System;
using System.Collections.Generic;

namespace Network
{
    public class NeuralNetwork : IComparable<NeuralNetwork>
    {
        private int[] layers;
        private float[][] neurons;
        private float[][][] weights;
        public float Fitness = 0;

        private Random random = new();

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

            for (int i = 0; i < layers.Length; i++)
                neuronsList.Add(new float[layers[i]]);

            neurons = neuronsList.ToArray();
        }

        private void InitWeights()
        {
            List<float[][]> weightsList = new();

            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> layerWeigthList = new();

                int neuronsInPreviousLayer = layers[i - 1];

                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];

                    // Set the weights randomly between -1 and 1
                    for (int k = 0; k < neuronsInPreviousLayer; k++)
                    {
                        neuronWeights[k] = random.NextSingle() - 0.5f;
                    }

                    layerWeigthList.Add(neuronWeights);
                }

                weightsList.Add(layerWeigthList.ToArray());
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
            {
                neurons[0][i] = inputs[i];
            }

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

            return neurons[neurons.Length - 1];
        }

        public void Mutate()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        float weight = weights[i][j][k];

                        float randomNumber = random.NextSingle() * 1000f;

                        if (randomNumber <= 2f)
                            weight *= -1f;
                        else if (randomNumber <= 4f)
                            weight = random.NextSingle() - 0.5f;
                        else if (randomNumber <= 6f)
                            weight *= random.NextSingle() + 1f;
                        else if (randomNumber <= 8f)
                            weight *= random.NextSingle();

                        weights[i][j][k] = weight;
                    }
                }
            }
        }

        public void ResetNeurons()
        {
            InitNeurons();
        }

        public int CompareTo(NeuralNetwork other)
        {
            if (other == null)
                return -1;

            if (Fitness > other.Fitness)
                return -1;
            else if (Fitness < other.Fitness)
                return 1;
            return 0;
        }
    }
}
