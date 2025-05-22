using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ImGuiNET;
using MachineLearning;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.ImGuiNet;
using MonoGame.Utils;
using MonoGame.Utils.Extensions;

namespace Arrows;

public class Application : Game
{
    private enum SimulationType
    {
        Xor,
        Arrows
    }

    public static Application Instance;

    private readonly GraphicsDeviceManager graphics;
    private SpriteBatch spriteBatch;
    private ImGuiRenderer imGuiRenderer;

    public int WindowWidth { get => graphics.PreferredBackBufferWidth; init => graphics.PreferredBackBufferWidth = value; }
    public int WindowHeight { get => graphics.PreferredBackBufferHeight; init => graphics.PreferredBackBufferHeight = value; }
    public Point WindowSize => new(WindowWidth, WindowHeight);

    private readonly Random random;

    private int networkCount = 100;
    private Arrow[] arrows;
    private NeuralNetwork[] networks;

    private const int ArrowNetworkInputCount = 2;
    private readonly int[] networkHiddenLayersCount = [5, 3];
    private const int ArrowNetworkOutputCount = 1;
    private const double NetworkGain = 0.5;

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private bool running;
    private bool runningForOneFrame;

    private Vector2 targetPosition;

    private float newTimeBetweenResets = 0.01f;
    public float TimeBetweenResets { get; private set; }
    public float TimeLeftBeforeReset { get; private set; }
    private int currentIteration = 1;

    private bool bestArrowSelected;
    private bool showFitnessGraphs;
    private Arrow selectedArrow;
    private bool showNeuralNetwork;

    private float simulationFrameRate = 60f;
    private bool simulationSpeedUncapped;

    private const int MaxFitnessGraphSize = 1000;
    private List<float> fitnessAverages = [];
    private List<float> fitnessMedians = [];

    public ActivationFunctions.Type HiddenLayersActivationFunction = ActivationFunctions.Type.Sigmoid;
    public ActivationFunctions.Type OutputLayerActivationFunction = ActivationFunctions.Type.Sigmoid;

    private SimulationType simulationType = SimulationType.Xor;

    public Application()
    {
        Instance = this;
        graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        Window.AllowUserResizing = true;
        WindowWidth = 1600;
        WindowHeight = 900;

        IsFixedTimeStep = false;
        InactiveSleepTime = TimeSpan.Zero;
        graphics.SynchronizeWithVerticalRetrace = true;

        random = new(DateTime.Now.Millisecond);

        arrowSpawnBounds = new(WindowSize.ToVector2() * 0.1f, WindowSize.ToVector2() * 0.9f);
    }

    protected override void Initialize()
    {
        imGuiRenderer = new(this);

        InitializeSimulation();

        base.Initialize();
    }

    private void InitializeNetworks(int count, int inputCount, int outputCount)
    {
        networks = new NeuralNetwork[count];

        for (int i = 0; i < count; i++)
        {
            networks[i] = new(random, ComputeFitness, inputCount, outputCount, networkHiddenLayersCount)
            {
                Rank = i
            };
        }
    }

    private void InitializeArrows()
    {
        Vector2 startingArrowPosition = GetRandomArrowSpawn();

        arrows = new Arrow[networkCount];

        for (int i = 0; i < networks.Length; i++)
        {
            arrows[i] = new(startingArrowPosition, networks[i], -1)
            {
                Angle = random.NextSingle() * MathHelper.TwoPi
            };
        }
    }

    protected override void LoadContent()
    {
        spriteBatch = new(GraphicsDevice);

        imGuiRenderer.RebuildFontAtlas();

        Arrow.Texture = Content.Load<Texture2D>("arrow");
    }

    protected override void Update(GameTime gameTime)
    {
        MouseStateExtended mouse = MouseExtended.GetState();
        Vector2 mousePosition = mouse.Position.ToVector2();

        float deltaTime = simulationSpeedUncapped ? 1f / simulationFrameRate : gameTime.GetElapsedSeconds();

        if (
            simulationType == SimulationType.Arrows &&
            mouse.WasButtonJustDown(MouseButton.Left) &&
            !ImGui.GetIO().WantCaptureMouse &&
            new Rectangle(Point.Zero, WindowSize).Contains(mouse.Position)
        )
        {
            foreach (Arrow arrow in arrows)
            {
                if (mousePosition.X > arrow.Position.X - Arrow.Size.X * 0.5f &&
                    mousePosition.Y > arrow.Position.Y - Arrow.Size.Y * 0.5f &&
                    mousePosition.X < arrow.Position.X + Arrow.Size.X * 0.5f &&
                    mousePosition.Y < arrow.Position.Y + Arrow.Size.Y * 0.5f)
                {
                    selectedArrow = arrow;
                    break;
                }

                selectedArrow = null;
            }
        }

        if (running || runningForOneFrame)
        {
            switch (simulationType)
            {
                case SimulationType.Xor:
                    foreach (NeuralNetwork network in networks)
                        network.UpdateFitness();
                    break;

                case SimulationType.Arrows:
                    foreach (Arrow arrow in arrows)
                        arrow.Update(deltaTime, targetPosition);
                    break;

                default: throw new ArgumentOutOfRangeException();
            }

            fitnessAverages.Add((float) networks.Average(n => n.Fitness));
            fitnessMedians.Add((float) networks[networkCount / 2].Fitness);

            if (fitnessAverages.Count > MaxFitnessGraphSize)
                fitnessAverages = fitnessAverages[^MaxFitnessGraphSize..];

            if (fitnessMedians.Count > MaxFitnessGraphSize)
                fitnessMedians = fitnessMedians[^MaxFitnessGraphSize..];

            if (TimeLeftBeforeReset <= 0f)
            {
                ResetSimulation(true);
            }
            else
            {
                Array.Sort(networks);

                for (int i = 0; i < networkCount; i++)
                    networks[i].Rank = i;

                if (simulationType == SimulationType.Arrows)
                    Array.Sort(arrows);
            }

            TimeLeftBeforeReset -= deltaTime;

            if (runningForOneFrame)
                runningForOneFrame = false;
        }

        base.Update(gameTime);
    }

    private double ComputeNetworkXor(NeuralNetwork network, bool lhs, bool rhs)
    {
        return network.ComputeOutputs(
            [
                Convert.ToDouble(lhs),
                Convert.ToDouble(rhs)
            ],
            ActivationFunctions.GetFromType(HiddenLayersActivationFunction),
            ActivationFunctions.GetFromType(OutputLayerActivationFunction)
        ).Single();
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.SlateGray);

        spriteBatch.Begin();

        switch (simulationType)
        {
            case SimulationType.Xor: break;

            case SimulationType.Arrows:
                spriteBatch.DrawCircle(targetPosition, 10f, 20, Color.Red, 10f);

                foreach (Arrow arrow in arrows)
                {
                    Color color = Color.Blue;
                    if (selectedArrow != null && selectedArrow != arrow)
                        color *= 0.05f;

                    arrow.Render(spriteBatch, color);
                }

                selectedArrow?.Render(spriteBatch, Color.Lime);
                break;

            default: throw new ArgumentOutOfRangeException();
        }

        spriteBatch.End();

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui(gameTime);
        imGuiRenderer.EndLayout();
    }

    private bool xorInput1, xorInput2;

    protected virtual void DrawImGui(GameTime gameTime)
    {
        ImGui.Begin("Simulation");

        if (ImGuiUtils.ComboEnum("Type", ref simulationType))
            InitializeSimulation();

        ImGui.SeparatorText("Settings");

        ImGui.InputInt("Network count", ref networkCount);

        ImGuiUtils.ComboEnum("Network hidden layers activation function", ref HiddenLayersActivationFunction);
        ImGuiUtils.ComboEnum("Network output layers activation function", ref OutputLayerActivationFunction);

        if (ImGui.DragFloat("Time between resets", ref newTimeBetweenResets, 0.1f, 1f))
            TimeLeftBeforeReset = MathF.Min(TimeLeftBeforeReset, newTimeBetweenResets);
        if (ImGui.Checkbox("Uncap FPS", ref simulationSpeedUncapped))
        {
            graphics.SynchronizeWithVerticalRetrace = !simulationSpeedUncapped;
            graphics.ApplyChanges();
        }

        if (!simulationSpeedUncapped)
            ImGui.BeginDisabled();

        ImGui.SliderFloat("Simulation FPS", ref simulationFrameRate, 5f, 300f);

        if (!simulationSpeedUncapped)
            ImGui.EndDisabled();

        ImGui.SeparatorText("Readonly data");

        double fps = 1.0 / gameTime.ElapsedGameTime.TotalSeconds;
        ImGui.Text($"FPS: {fps}");
        ImGui.Text($"Total time: {gameTime.TotalGameTime}");
        ImGui.Text($"Current iteration: {currentIteration}");
        ImGui.Text($"Average fitness: {(fitnessAverages.Count > 0 ? fitnessAverages[^1] : 0f)}");
        ImGui.Text($"Median fitness: {(fitnessMedians.Count > 0 ? fitnessMedians[^1] : 0f)}");
        double simulationSpeed = fps / simulationFrameRate;
        ImGui.Text($"Running at {simulationSpeed.ToString("F2", CultureInfo.CurrentCulture)}x speed");
        ImGui.Text($"{(simulationSpeed / TimeBetweenResets).ToString("F3", CultureInfo.CurrentCulture)} iterations per second");

        ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {TimeLeftBeforeReset}s");
        ImGui.Text($"Target position {targetPosition}");

        ImGui.SeparatorText("Actions");

        ImGui.Checkbox("Running", ref running);

        if (ImGui.Button("Next generation"))
            ResetSimulation(true);

        if (running)
            ImGui.BeginDisabled();

        if (ImGui.Button("Next frame"))
            runningForOneFrame = true;

        if (running)
            ImGui.EndDisabled();

        if (ImGui.Button("Save best"))
            SaveBestNetwork();

        if (ImGui.Button("Load save"))
            LoadSavedNetwork();

        ImGui.Checkbox("Show fitness graphs", ref showFitnessGraphs);

        switch (simulationType)
        {
            case SimulationType.Xor:
                ImGui.SeparatorText("Best network");
                ImGui.Checkbox("Input 1", ref xorInput1);
                ImGui.SameLine();
                ImGui.Checkbox("Input 2", ref xorInput2);
                NeuralNetwork bestNetwork = networks.First();
                ImGui.Text($"Output: {ComputeNetworkXor(bestNetwork, xorInput1, xorInput2)}");

                ImGui.Separator();

                ImGui.Text($"Fitness {bestNetwork.Fitness}");

                ImGui.Checkbox("Show neural network", ref showNeuralNetwork);

                if (showNeuralNetwork)
                {
                    ImGui.Begin("Neural Network");
                    ImGuiUtils.DisplayNeuralNetwork(bestNetwork);
                    ImGui.End();
                }
                break;

            case SimulationType.Arrows:
                if (ImGui.Button("Select best arrow"))
                    selectedArrow = arrows[0];

                if (ImGui.Checkbox("Keep best arrow selected", ref bestArrowSelected) && !bestArrowSelected)
                    selectedArrow = null;

                if (bestArrowSelected)
                    selectedArrow = arrows[0];

                if (selectedArrow != null)
                {
                    ImGui.SeparatorText("Selected arrow data");

                    ImGui.Text($"Position {selectedArrow.Position}");
                    ImGui.Text($"Angle {MathHelper.ToDegrees(selectedArrow.Angle)}deg");
                    Vector2 direction = new(MathF.Cos(selectedArrow.Angle), -MathF.Sin(selectedArrow.Angle));
                    ImGuiUtils.DirectionVector("Direction", ref direction, selectedArrow.TargetDirection);

                    if (ImGui.Button("+"))
                        selectedArrow = arrows[(selectedArrow.Rank - 1 + arrows.Length) % arrows.Length];
                    ImGui.SameLine();
                    if (ImGui.Button("-"))
                        selectedArrow = arrows[(selectedArrow.Rank + 1) % arrows.Length];
                    ImGui.SameLine();
                    ImGui.Text($"Current rank {selectedArrow.Rank + 1}");
                    ImGui.Text($"Last rank {selectedArrow.LastRank + 1}");
                    ImGui.Text($"Fitness {selectedArrow.Network.Fitness}");

                    ImGui.Checkbox("Show neural network", ref showNeuralNetwork);

                    if (showNeuralNetwork)
                    {
                        ImGui.Begin("Neural Network");
                        ImGuiUtils.DisplayNeuralNetwork(selectedArrow.Network);
                        ImGui.End();
                    }
                }
                break;

            default: throw new ArgumentOutOfRangeException();
        }

        ImGui.End();

        if (showFitnessGraphs)
        {
            ImGui.Begin("Fitness graphs");
            ImGui.BeginTabBar("fitnessGraphsTabBar");

            unsafe
            {
                ref float pointer = ref Unsafe.AsRef<float>((void*) nint.Zero);
                if (ImGui.BeginTabItem("Average"))
                {
                    if (fitnessAverages.Count > 0)
                        pointer = ref CollectionsMarshal.AsSpan(fitnessAverages)[0];

                    ImGui.PlotLines(
                        "##graph",
                        ref pointer,
                        fitnessAverages.Count,
                        0,
                        string.Empty,
                        float.MaxValue,
                        float.MaxValue,
                        ImGui.GetContentRegionAvail()
                    );
                    ImGui.EndTabItem();
                }

                if (ImGui.BeginTabItem("Median"))
                {
                    if (fitnessMedians.Count > 0)
                        pointer = ref CollectionsMarshal.AsSpan(fitnessMedians)[0];

                    ImGui.PlotLines(
                        "##graph",
                        ref pointer,
                        fitnessMedians.Count,
                        0,
                        string.Empty,
                        float.MaxValue,
                        float.MaxValue,
                        ImGui.GetContentRegionAvail()
                    );
                    ImGui.EndTabItem();
                }
            }

            ImGui.EndTabBar();
            ImGui.End();
        }
    }

    private void ResetSimulation(bool evolve)
    {
        Array.Sort(networks);

        if (evolve)
            EvolveSimulation();

        for (int i = 0; i < networkCount; i++)
        {
            networks[i].Rank = i;
            networks[i].UpdateFitness();
        }

        if (simulationType == SimulationType.Arrows)
        {
            int selectedArrowIndex = selectedArrow?.Rank ?? -1;

            Vector2 startingArrowPosition = GetRandomArrowSpawn();

            for (int i = 0; i < networkCount; i++)
            {
                ref Arrow arrow = ref arrows[i];

                arrow = new(startingArrowPosition, networks[i], arrow.Rank);

                if (arrow.LastRank == selectedArrowIndex)
                    selectedArrow = arrow;
            }

            Array.Sort(arrows);
        }

        TimeBetweenResets = newTimeBetweenResets;
        TimeLeftBeforeReset = newTimeBetweenResets;

        currentIteration++;
    }

    private void EvolveSimulation()
    {
        foreach (NeuralNetwork network in networks)
            network.LearnByFitness(NetworkGain);

        // Mutate the worst 50%
        const float MutatedNetworkRatio = 0.5f;
        int mutatedNetworkCount = (int) (networkCount * MutatedNetworkRatio);
        int nonMutatedNetworkCount = networkCount - mutatedNetworkCount;
        for (int i = 0; i < mutatedNetworkCount; i++)
        {
            ref NeuralNetwork badNetwork = ref networks[nonMutatedNetworkCount + i];

            badNetwork = new(networks[i], badNetwork);
            badNetwork.Mutate();
        }

        // Mutate again every network that has a negative fitness (only the case for the arrow simulation)
        foreach (NeuralNetwork network in networks)
        {
            if (network.Fitness < 0.0)
                network.Mutate();
        }
    }

    private void InitializeSimulation()
    {
        switch (simulationType)
        {
            case SimulationType.Xor:
                InitializeNetworks(networkCount, 2, 1);
                break;

            case SimulationType.Arrows:
                InitializeNetworks(networkCount, ArrowNetworkInputCount, ArrowNetworkOutputCount);
                targetPosition = new(
                    random.NextSingle() * 0.5f * WindowWidth + WindowWidth * 0.25f,
                    random.NextSingle() * 0.5f * WindowHeight + WindowHeight * 0.25f
                );
                InitializeArrows();
                break;

            default: throw new ArgumentOutOfRangeException();
        }
    }

    private void SaveBestNetwork() => arrows[0].Network.Save(SavePath);

    private void LoadSavedNetwork()
    {
        NeuralNetwork saved = NeuralNetwork.Load(SavePath);

        for (int i = 0; i < networkCount; i++)
            networks[i] = new(saved);

        ResetSimulation(false);
    }

    private Vector2 GetRandomArrowSpawn() => random.NextVector2() * ((Point2) arrowSpawnBounds.Size - arrowSpawnBounds.Position) + arrowSpawnBounds.Position;

    private readonly bool[][] possibleXorInputs =
    [
        [false, false],
        [true, false],
        [false, true],
        [true, true]
    ];

    private double ComputeFitness(NeuralNetwork network)
    {
        switch (simulationType)
        {
            case SimulationType.Xor:
                double sum = 0.0;

                foreach (bool[] possibleInput in possibleXorInputs)
                {
                    double output = ComputeNetworkXor(network, possibleInput[0], possibleInput[1]);
                    double expectedOutput = Convert.ToDouble(XorGate(possibleInput[0], possibleInput[1]));

                    double difference = Math.Abs(output - expectedOutput);
                    double factor = (0.5 - difference) * 10.0; // between -5 and 5

                    sum += factor * factor * factor;
                }

                return sum;

            case SimulationType.Arrows:
                Arrow arrow = arrows[network.Rank];

                // First, compute the network outputs with the current inputs and apply them to the angle

                float oldAngle = arrow.Angle;
                arrow.Angle += arrow.ComputeNextAngleDelta(targetPosition);

                // Then compute the actual fitness

                const float MaxDistance = 100f;
                const float MaxDistanceSquared = MaxDistance * MaxDistance;

                // The fitness is computed in two different ways:
                // - If the arrow is outside the MaxDistance range of the target,
                //   we only care about the current arrow angle
                // - If it is inside that range, we check how close it is
                //   to the target, and we make sure that the arrow isn't facing away
                //   from the target

                const float AngleFitnessValueFar = 5f;
                const float MaxAngleValueFar = MathHelper.PiOver4;

                const float AngleFitnessValueClose = 4f;
                const float MaxAngleValueClose = MathHelper.PiOver2;
                const float PositionXFitnessValue = 0.5f;
                const float PositionYFitnessValue = 0.5f;

                float result = 0f;
                Vector2 angleDirection = Calc.DirectionFromAngle(arrow.Angle);
                float dot = Vector2.Dot(angleDirection, arrow.TargetDirection);

                if ((targetPosition - arrow.Position).LengthSquared() > MaxDistanceSquared)
                {
                    result += Calc.ComputeDifference(dot, 1f, 1f - MaxAngleValueFar / MathHelper.Pi, AngleFitnessValueFar) - AngleFitnessValueFar;
                }
                else
                {
                    result += Calc.ComputeDifference(dot, 1f, 1f - MaxAngleValueClose / MathHelper.Pi, AngleFitnessValueClose);
                    result += Calc.ComputeDifference(arrow.Position.X, targetPosition.X, MaxDistance, PositionXFitnessValue);
                    result += Calc.ComputeDifference(arrow.Position.Y, targetPosition.Y, MaxDistance, PositionYFitnessValue);
                }

                // Eventually undo the changes we've done to the angle

                arrow.Angle = oldAngle;

                return result;

            default: throw new ArgumentOutOfRangeException();
        }
    }

    private static bool XorGate(bool lhs, bool rhs) => (lhs && !rhs) || (!lhs && rhs);
}
