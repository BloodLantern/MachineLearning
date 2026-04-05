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
    private const int NetworkInputCount = 2;
    private const int NetworkOutputCount = 1;

    private const string SavePath = "network_save.xml";

    private const int MaxFitnessGraphSize = 1000;

    public static Application Instance;

    private readonly RectangleF arrowSpawnBounds;

    private readonly GraphicsDeviceManager graphics;
    private readonly int[] networkHiddenNeuronsCount = [5, 3];

    private readonly Random random;
    private Arrow[] arrows;

    private bool bestArrowSelected;
    private int currentIteration = 1;
    private List<float> fitnessAverages = [];
    private List<float> fitnessMedians = [];

    public ActivationFunctionType HiddenLayersActivationFunction = ActivationFunctionType.HyperbolicTangent;
    private ImGuiRenderer imGuiRenderer;

    private int networkCount = 100;
    private float networkGain = 0.5f;
    private NeuralNetwork[] networks;

    private float newTimeBetweenResets = 10f;
    public ActivationFunctionType OutputLayerActivationFunction = ActivationFunctionType.HyperbolicTangent;

    private bool running;
    private bool runningForOneFrame;
    private Arrow selectedArrow;
    private bool showFitnessGraphs;
    private bool showNeuralNetwork;

    private float simulationFrameRate = 60f;
    private bool simulationSpeedUncapped;

    private SpriteBatch spriteBatch;

    private Vector2 targetPosition;

    public int WindowWidth
    {
        get => graphics.PreferredBackBufferWidth;
        init => graphics.PreferredBackBufferWidth = value;
    }

    public int WindowHeight
    {
        get => graphics.PreferredBackBufferHeight;
        init => graphics.PreferredBackBufferHeight = value;
    }

    public Vector2 WindowSize
    {
        get => new(WindowWidth, WindowHeight);
        init
        {
            WindowWidth = (int) value.X;
            WindowHeight = (int) value.Y;
        }
    }

    public float TimeBetweenResets { get; private set; }
    public float TimeLeftBeforeReset { get; private set; }

    public Application()
    {
        Instance = this;
        graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        Window.AllowUserResizing = true;
        WindowSize = new(1600, 900);

        IsFixedTimeStep = false;
        InactiveSleepTime = TimeSpan.Zero;
        graphics.SynchronizeWithVerticalRetrace = true;

        random = new(DateTime.Now.Millisecond);

        arrowSpawnBounds = new(WindowSize * 0.1f, WindowSize * 0.9f);

        TimeLeftBeforeReset = newTimeBetweenResets;
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
            networks[i] = new(random, ComputeFitness, inputCount, outputCount, networkHiddenNeuronsCount)
            {
                Rank = i
            };
        }
    }

    public const float ArrowAngleEpsilon = 1e-5f;

    private void InitializeArrows()
    {
        Vector2 startingArrowPosition = GetRandomArrowSpawn();

        arrows = new Arrow[networkCount];

        float randomAngle = GetRandomArrowAngle();
        for (int i = 0; i < networks.Length; i++)
        {
            arrows[i] = new(startingArrowPosition, networks[i], -1)
            {
                Angle = randomAngle + ArrowAngleEpsilon * i
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

        // If mouse is inside the game window
        if (!ImGui.GetIO().WantCaptureMouse &&
            new Rectangle(Point.Zero, WindowSize.ToPoint()).Contains(mouse.Position))
        {
            if (mouse.IsButtonDown(MouseButton.Right))
                targetPosition = mousePosition;

            if (mouse.WasButtonJustDown(MouseButton.Left))
            {
                foreach (Arrow arrow in arrows)
                {
                    if ((mousePosition - arrow.Position).LengthSquared() < Arrow.Size.LengthSquared())
                    {
                        selectedArrow = arrow;
                        break;
                    }

                    selectedArrow = null;
                }
            }
        }

        if (running || runningForOneFrame)
        {
            foreach (Arrow arrow in arrows)
                arrow.Update(deltaTime, targetPosition);

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
                {
                    networks[i].LearnByFitness(networkGain / TimeBetweenResets);
                    networks[i].Rank = i;
                }

                Array.Sort(arrows);
            }

            TimeLeftBeforeReset -= deltaTime;

            if (runningForOneFrame)
                runningForOneFrame = false;
        }

        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.SlateGray);

        spriteBatch.Begin();

        spriteBatch.DrawCircle(targetPosition, 10f, 20, Color.Red, 10f);

        foreach (Arrow arrow in arrows)
        {
            Color color = Color.White;
            if (selectedArrow != null && selectedArrow != arrow)
                color *= 0.05f;

            arrow.Render(spriteBatch, color);
        }

        selectedArrow?.Render(spriteBatch, Color.Lime);

        spriteBatch.DrawCircle(targetPosition, FitnessBonusMaxDistance, 30, Color.Red);

        spriteBatch.End();

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui(gameTime);
        imGuiRenderer.EndLayout();
    }

    protected virtual void DrawImGui(GameTime gameTime)
    {
        ImGui.Begin("Simulation");

        ImGui.SeparatorText("Settings");

        ImGui.InputInt("Network count", ref networkCount);

        ImGuiUtils.ComboEnum("Network hidden layers activation function", ref HiddenLayersActivationFunction);
        ImGuiUtils.ComboEnum("Network output layers activation function", ref OutputLayerActivationFunction);

        ImGui.SliderFloat("Network gain", ref networkGain, 0f, 1f);

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
                DrawNeuralNetworkWindow();
        }

        ImGui.End();

        if (showFitnessGraphs)
            DrawFitnessGraphWindow();
    }

    private void DrawNeuralNetworkWindow()
    {
        ImGui.Begin("Neural Network");
        ImGuiUtils.DisplayNeuralNetwork(selectedArrow.Network);
        ImGui.End();
    }

    private void DrawFitnessGraphWindow()
    {
        ImGui.Begin("Fitness graphs");

        ref float pointer = ref Unsafe.NullRef<float>();
        if (fitnessAverages.Count > 0)
            pointer = ref CollectionsMarshal.AsSpan(fitnessAverages)[0];

        ImGui.PlotLines(
            "Average",
            ref pointer,
            fitnessAverages.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            ImGui.GetContentRegionAvail()
        );

        if (fitnessMedians.Count > 0)
            pointer = ref CollectionsMarshal.AsSpan(fitnessMedians)[0];

        ImGui.PlotLines(
            "Median",
            ref pointer,
            fitnessMedians.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            ImGui.GetContentRegionAvail()
        );

        ImGui.End();
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

        int selectedArrowIndex = selectedArrow?.Rank ?? -1;

        Vector2 startingArrowPosition = GetRandomArrowSpawn();
        float startingArrowAngle = GetRandomArrowAngle();

        for (int i = 0; i < networkCount; i++)
        {
            arrows[i] = new(startingArrowPosition, networks[i], arrows[i].Rank)
            {
                Angle = startingArrowAngle + ArrowAngleEpsilon * i
            };

            if (arrows[i].LastRank == selectedArrowIndex)
                selectedArrow = arrows[i];
        }

        Array.Sort(arrows);

        TimeBetweenResets = newTimeBetweenResets;
        TimeLeftBeforeReset = newTimeBetweenResets;

        currentIteration++;
    }

    private void EvolveSimulation()
    {
        const float BestNetworksFraction = 0.5f;
        const float MutateFraction = 0.05f;

        // Assume sorted networks array
        NeuralNetwork[] bestNetworks = networks[..(int) (networkCount / (1f / BestNetworksFraction))];

        // Keep the 50% best and make them learn from their mistakes
        foreach (NeuralNetwork network in bestNetworks)
            network.LearnByFitness(networkGain);

        // Then create copies of the best networks and mutate 5% of them
        for (int i = 0; i < networkCount; i++)
        {
            networks[i] = new(bestNetworks[i % bestNetworks.Length]);

            if (i < networkCount / (1f / MutateFraction))
                networks[i].Mutate();
        }
    }

    private void InitializeSimulation()
    {
        InitializeNetworks(networkCount, NetworkInputCount, NetworkOutputCount);
        targetPosition = random.NextVector2() * 0.5f * WindowSize + WindowSize * 0.25f;
        InitializeArrows();
    }

    private void SaveBestNetwork() => arrows[0].Network.Save(SavePath);

    private void LoadSavedNetwork()
    {
        NeuralNetwork saved = NeuralNetwork.Load(SavePath);

        for (int i = 0; i < networkCount; i++)
            networks[i] = new(saved);

        ResetSimulation(false);
    }

    private Vector2 GetRandomArrowSpawn()
        => random.NextVector2() * ((Point2) arrowSpawnBounds.Size - arrowSpawnBounds.Position) + arrowSpawnBounds.Position;

    private float GetRandomArrowAngle() => random.NextSingle() * MathHelper.TwoPi;

    public const float FitnessBonusMaxDistance = 100f;
    public const float FitnessBonusMaxDistanceSquared = FitnessBonusMaxDistance * FitnessBonusMaxDistance;

    private double ComputeFitness(NeuralNetwork network)
    {
        Arrow arrow = arrows[network.Rank];

        // The fitness is computed in two different ways:
        // - If the arrow is outside the FitnessBonusMaxDistance range of the target,
        //   we only care about the current arrow angle
        // - If it is inside that range, we check how close it is
        //   to the target, and we make sure that the arrow isn't facing away
        //   from the target

        const float AngleFitnessValueFar = 5f;
        const float MaxAngleValueFar = MathHelper.PiOver4 * 0.5f;

        const float AngleFitnessValueClose = 10f;
        const float MaxAngleValueClose = MathHelper.PiOver4 * 1.5f;
        const float PositionXFitnessValue = 2.5f;
        const float PositionYFitnessValue = 2.5f;

        float result = 0f;
        Vector2 angleDirection = Vector2.FromAngle(arrow.Angle);
        float dot = Vector2.Dot(angleDirection, arrow.TargetDirection);

        if ((targetPosition - arrow.Position).LengthSquared() > FitnessBonusMaxDistanceSquared)
        {
            result += Calc.ComputeDifference(dot, 1f, MaxAngleValueFar / MathHelper.Pi, AngleFitnessValueFar);
        }
        else
        {
            result += Calc.ComputeDifference(dot, 1f, MaxAngleValueClose / MathHelper.Pi, AngleFitnessValueClose);
            result += Calc.ComputeDifference(arrow.Position.X, targetPosition.X, FitnessBonusMaxDistance, PositionXFitnessValue);
            result += Calc.ComputeDifference(arrow.Position.Y, targetPosition.Y, FitnessBonusMaxDistance, PositionYFitnessValue);
        }

        return result;
    }
}
