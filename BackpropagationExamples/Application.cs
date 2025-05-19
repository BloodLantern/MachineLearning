using System;
using System.Linq;
using ImGuiNET;
using MachineLearning;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.ImGuiNet;

namespace Xor;

public class Application : Game
{
    private enum SimulationType
    {
        Xor,
        And,
        Decisions
    }

    private readonly GraphicsDeviceManager graphics;
    private SpriteBatch spriteBatch;
    private ImGuiRenderer imGuiRenderer;

    private NeuralNetwork neuralNetwork;

    private readonly Random random = new();

    private bool[] inputs;
    private double output;

    private double gain = 0.3;

    private double[][] trainingSet;
    private double[] refOutputs;
    private int trainIterations = 1000;
    private bool showNetwork;

    private bool networkChanged = true;

    private SimulationType simulationType = SimulationType.Xor;

    private int hiddenLayerSize = 5;

    public Application()
    {
        graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
        Window.AllowUserResizing = true;
    }

    protected override void Initialize()
    {
        graphics.PreferredBackBufferWidth = 1280;
        graphics.PreferredBackBufferHeight = 720;
        graphics.ApplyChanges();

        imGuiRenderer = new(this);

        SetupXor();

        base.Initialize();
    }

    protected override void LoadContent()
    {
        spriteBatch = new(GraphicsDevice);

        imGuiRenderer.RebuildFontAtlas();
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.CornflowerBlue);

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui();
        imGuiRenderer.EndLayout();
    }

    protected virtual void DrawImGui()
    {
        ImGui.Begin("Simulation");

        if (ImGuiUtils.ComboEnum("Type", ref simulationType))
        {
            networkChanged = true;

            // ReSharper disable once SwitchStatementHandlesSomeKnownEnumValuesWithDefault
            switch (simulationType)
            {
                case SimulationType.Xor: SetupXor(); break;
                case SimulationType.And: SetupAnd(); break;
                case SimulationType.Decisions: SetupDecisions(); break;
            }
        }

        ImGui.SeparatorText("Controls");

        bool inputsModified = networkChanged;
        if (simulationType == SimulationType.Decisions)
        {
            inputsModified |= ImGui.Checkbox("Input Life", ref inputs[0]);
            ImGui.SameLine();
            inputsModified |= ImGui.Checkbox("Input Ammo", ref inputs[1]);
            ImGui.SameLine();
            inputsModified |= ImGui.Checkbox("Input E.Strength", ref inputs[2]);
        }
        else
        {
            inputsModified |= ImGui.Checkbox("Input 1", ref inputs[0]);
            for (int i = 1; i < inputs.Length; i++)
            {
                ImGui.SameLine();
                inputsModified |= ImGui.Checkbox($"Input {i + 1}", ref inputs[i]);
            }
        }

        if (inputsModified)
            output = neuralNetwork.ComputeOutputs(inputs.Select(Convert.ToDouble).ToArray()).Single();
        ImGui.Text($"Output: {output}");

        ImGui.Separator();

        if (ImGui.Button("Train network"))
            TrainNetwork(trainIterations);
        ImGui.SameLine();
        ImGui.InputInt("Iterations", ref trainIterations);

        ImGui.SeparatorText("Settings");

        if (ImGui.InputInt("Hidden layer size", ref hiddenLayerSize) && hiddenLayerSize > 0)
            neuralNetwork = new(random, neuralNetwork.InputLayer.NeuronCount, neuralNetwork.OutputLayer.NeuronCount, hiddenLayerSize);

        float gainFloat = (float) gain;
        ImGui.SliderFloat("Gain", ref gainFloat, 0f, 1f);
        gain = gainFloat;

        ImGui.Checkbox("Show network", ref showNetwork);
        if (showNetwork)
        {
            ImGui.Begin("Neural Network");
            ImGuiUtils.DisplayNeuralNetwork(neuralNetwork);
            ImGui.End();
        }

        ImGui.End();
    }

    private void SetupXor()
    {
        const int InputCount = 2;

        // XOR operator training set
        neuralNetwork = new(random, null, InputCount, 1, hiddenLayerSize);
        inputs = new bool[InputCount];
        // set input values
        trainingSet =
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        // set expected output values
        refOutputs =
        [
            0.0,
            1.0,
            1.0,
            0.0,
        ];
    }

    private void SetupAnd()
    {
        const int InputCount = 2;

        // AND operator training set
        neuralNetwork = new(random, null, InputCount, 1, hiddenLayerSize);
        inputs = new bool[InputCount];
        // set input values
        trainingSet =
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        // set expected output values
        refOutputs =
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ];
    }

    private void SetupDecisions()
    {
        const int InputCount = 3;

        neuralNetwork = new(random, null, InputCount, 1, hiddenLayerSize);
        inputs = new bool[InputCount];
        // set input values
        trainingSet =
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        // set expected output values
        refOutputs =
        [
            0.0,
            0.0,
            0.5,
            0.0,
            0.5,
            1.0,
            1.0,
        ];
    }

    private void TrainNetwork(int iterations)
    {
        networkChanged = true;

        if (iterations <= 0)
            iterations = 1000;

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int i = 0; i < refOutputs.Length; i++)
            {
                double[] inputList = new double[neuralNetwork.InputLayer.NeuronCount];
                double[] outputList = [refOutputs[i]];

                for (int j = 0; j < neuralNetwork.InputLayer.NeuronCount; j++)
                    inputList[j] = trainingSet[i][j];

                neuralNetwork.LearnByBackpropagation(gain, inputList, outputList);
            }
        }
    }
}
