using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Arrows.Utils;
using ImGuiNET;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.ImGuiNet;

namespace Arrows;

public class Application : Game
{
    public static Application Instance;

    private readonly GraphicsDeviceManager graphics;
    private SpriteBatch spriteBatch;
    private ImGuiRenderer imGuiRenderer;

    public int WindowWidth { get => graphics.PreferredBackBufferWidth; init => graphics.PreferredBackBufferWidth = value; }
    public int WindowHeight { get => graphics.PreferredBackBufferHeight; init => graphics.PreferredBackBufferHeight = value; }
    public Point WindowSize => new(WindowWidth, WindowHeight);

    private readonly Random random;

    private const int ArrowCount = 200;
    private readonly Arrow[] arrows = new Arrow[ArrowCount];
    private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];
        
    private const int NetworkInputCount = 6;
    private readonly int[] networkHiddenLayersCount = [5];
    private const int NetworkOutputCount = 1;
    private const double NetworkLearnRate = 0.15;

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private bool running;
    private bool runningForOneFrame;
        
    private Vector2 targetPosition;

    private float newTimeBetweenResets = 7f;
    public float TimeBetweenResets { get; private set; }
    public float TimeLeftBeforeReset { get; private set; }
    private int currentIteration;
    
    private bool bestArrowSelected;
    private bool showFitnessGraphs;
    private Arrow selectedArrow;
    private bool showNeuralNetwork;

    private float simulationFrameRate = 60f;
    private bool simulationSpeedUncapped;

    private List<float> fitnessAverages = [];
    private List<float> fitnessMedians = [];

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

        Vector2 startingArrowPosition = GetRandomArrowSpawn();
            
        for (int i = 0; i < ArrowCount; i++)
        {
            int[] networkSize = new int[2 + networkHiddenLayersCount.Length];
            networkSize[0] = NetworkInputCount;
            for (int j = 0; j < networkHiddenLayersCount.Length; j++)
                networkSize[j + 1] = networkHiddenLayersCount[j];
            networkSize[^1] = NetworkOutputCount;
                
            networks[i] = new(random, ComputeFitness, networkSize)
            {
                Rank = i
            };
            arrows[i] = new(startingArrowPosition, networks[i], -1)
            {
                Angle = random.NextSingle() * MathHelper.TwoPi
            };
        }

        base.Initialize();
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

        if (mouse.WasButtonJustUp(MouseButton.Left) && !ImGui.GetIO().WantCaptureMouse && new Rectangle(Point.Zero, WindowSize).Contains(mouse.Position))
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
            foreach (Arrow arrow in arrows)
                arrow.Update(deltaTime, targetPosition);
            
            fitnessAverages.Add((float) networks.Average(n => n.Fitness));
            fitnessMedians.Add((float) networks[ArrowCount / 2].Fitness);

            if (TimeLeftBeforeReset <= 0f)
            {
                ResetSimulation(true);
            }
            else
            {
                Array.Sort(networks);

                for (int i = 0; i < ArrowCount; i++)
                    networks[i].Rank = i;
                
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
            Color color = Color.Blue;
            if (selectedArrow != null && selectedArrow != arrow)
                color *= 0.05f;
                
            arrow.Render(spriteBatch, color);
        }

        selectedArrow?.Render(spriteBatch, Color.Lime);

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
        
        ImGui.DragFloat("Time between resets", ref newTimeBetweenResets, 0.1f, 1f);
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
        ImGui.Text($"Running at {simulationSpeed.ToString("F2", CultureInfo.InvariantCulture)}x speed");
        ImGui.Text($"{(simulationSpeed / TimeBetweenResets).ToString("F3", CultureInfo.InvariantCulture)} iterations per second");
        
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

        if (ImGui.Button("Select best arrow"))
            selectedArrow = arrows[0];

        if (ImGui.Checkbox("Keep best arrow selected", ref bestArrowSelected) && !bestArrowSelected)
            selectedArrow = null;

        ImGui.Checkbox("Show fitness graphs", ref showFitnessGraphs);

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
                    
                    ImGui.PlotLines("##graph", ref pointer, fitnessAverages.Count, 0, string.Empty, float.MaxValue, float.MaxValue, ImGui.GetContentRegionAvail());
                    ImGui.EndTabItem();
                }

                if (ImGui.BeginTabItem("Median"))
                {
                    if (fitnessMedians.Count > 0)
                        pointer = ref CollectionsMarshal.AsSpan(fitnessMedians)[0];
                    
                    ImGui.PlotLines("##graph", ref pointer, fitnessMedians.Count, 0, string.Empty, float.MaxValue, float.MaxValue, ImGui.GetContentRegionAvail());
                    ImGui.EndTabItem();
                }
            }
            
            ImGui.EndTabBar();
            ImGui.End();
        }
    }

    private void ResetSimulation(bool evolve)
    {
        if (evolve)
            EvolveSimulation();

        for (int i = 0; i < ArrowCount; i++)
        {
            networks[i].Rank = i;
            networks[i].Fitness = 0.0;
        }

        int selectedArrowIndex = selectedArrow?.Rank ?? -1;

        Vector2 startingArrowPosition = GetRandomArrowSpawn();
        
        for (int i = 0; i < ArrowCount; i++)
        {
            ref Arrow arrow = ref arrows[i];
            
            arrow = new(startingArrowPosition, networks[i], arrow.Rank);

            if (arrow.LastRank == selectedArrowIndex)
                selectedArrow = arrow;
        }
        
        Array.Sort(arrows);
        
        fitnessAverages.Clear();
        fitnessMedians.Clear();

        InitializeSimulation();
    }

    private void EvolveSimulation()
    {
        Array.Sort(networks);

        foreach (NeuralNetwork network in networks)
            network.Learn(NetworkLearnRate);

        // Mutate the worst 50%
        const float MutatedArrowRatio = 0.5f;
        const int MutatedArrowCount = (int) (ArrowCount * MutatedArrowRatio);
        const int NonMutatedArrowCount = ArrowCount - MutatedArrowCount;
        for (int i = 0; i < MutatedArrowCount; i++)
        {
            ref NeuralNetwork badNetwork = ref networks[NonMutatedArrowCount + i];
            
            badNetwork = new(networks[i], badNetwork);
            badNetwork.Mutate();
        }

        // Mutate again every network that has a negative fitness
        foreach (NeuralNetwork network in networks)
        {
            if (network.Fitness < 0.0)
                network.Mutate();
        }
    }

    private void InitializeSimulation()
    {
        TimeBetweenResets = newTimeBetweenResets;
        TimeLeftBeforeReset = newTimeBetweenResets;

        targetPosition = new(random.NextSingle() * 0.5f * WindowWidth + WindowWidth * 0.25f, random.NextSingle() * 0.5f * WindowHeight + WindowHeight * 0.25f);

        currentIteration++;
    }

    private void SaveBestNetwork() => arrows[0].Network.Save(SavePath);

    private void LoadSavedNetwork()
    {
        NeuralNetwork saved = NeuralNetwork.Load(SavePath);
        
        for (int i = 0; i < ArrowCount; i++)
            networks[i] = new(saved);
        
        ResetSimulation(false);
    }

    private Vector2 GetRandomArrowSpawn() => random.NextVector2() * ((Point2) arrowSpawnBounds.Size - arrowSpawnBounds.Position) + arrowSpawnBounds.Position;

    private double ComputeFitness(NeuralNetwork network)
    {
        Arrow arrow = arrows[network.Rank];
        
        // First compute the network outputs with the current inputs and apply them to the angle

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
    }
}
