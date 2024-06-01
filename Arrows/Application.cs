using System;
using System.Globalization;
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
        
    private const int NetworkInputCount = 5;
    private readonly int[] networkHiddenLayersCount = [5];
    private const int NetworkOutputCount = 1;

    private const string SavePath = "network_save.xml";

    private readonly Vector2 startingArrowPosition;
    private readonly Vector2 startingTargetPosition;
    private readonly float startingTargetOffsetY;

    private bool running;
    private bool runningForOneFrame;
        
    private Vector2 targetPosition;

    private float timeBetweenResets = 7f;
    private float timeLeftBeforeReset;
    private int currentIteration;
    
    private bool bestArrowSelected;
    private Arrow selectedArrow;
    private bool showNeuralNetwork;

    private float simulationFrameRate = 60f;
    private bool simulationSpeedUncapped;

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

        startingArrowPosition = new(WindowWidth * 0.3f, WindowHeight * 0.5f);
        startingTargetPosition = new(WindowWidth * 0.6f, WindowHeight * 0.5f);
        startingTargetOffsetY = WindowHeight * 0.4f;

        random = new(DateTime.Now.Millisecond);
    }

    protected override void Initialize()
    {
        imGuiRenderer = new(this);

        InitializeSimulation();
            
        for (int i = 0; i < ArrowCount; i++)
        {
            int[] networkSize = new int[2 + networkHiddenLayersCount.Length];
            networkSize[0] = NetworkInputCount;
            for (int j = 0; j < networkHiddenLayersCount.Length; j++)
                networkSize[j + 1] = networkHiddenLayersCount[j];
            networkSize[^1] = NetworkOutputCount;
                
            networks[i] = new(networkSize, random)
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

            if (timeLeftBeforeReset <= 0f)
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

            timeLeftBeforeReset -= deltaTime;

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
            
        ImGui.DragFloat("Time between resets", ref timeBetweenResets, 0.1f, 1f);
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
        double simulationSpeed = fps / simulationFrameRate;
        ImGui.Text($"Running at {simulationSpeed.ToString("F2", CultureInfo.InvariantCulture)}x speed");
        ImGui.Text($"{(simulationSpeed / timeBetweenResets).ToString("F3", CultureInfo.InvariantCulture)} iterations per second");
        
        ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {timeLeftBeforeReset}s");
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
            ImGui.Text($"Fitness {selectedArrow.NeuralNetwork.Fitness}");

            ImGui.Checkbox("Show neural network", ref showNeuralNetwork);
            
            if (showNeuralNetwork)
            {
                ImGui.Begin("Neural Network");
                ImGuiUtils.DisplayNeuralNetwork(selectedArrow.NeuralNetwork);
                ImGui.End();
            }
        }
                
        ImGui.End();
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

        Vector2 randomPosition = new(random.NextSingle() * 0.5f * WindowWidth + WindowWidth * 0.25f,
            random.NextSingle() * 0.5f * WindowHeight + WindowHeight * 0.25f);
        for (int i = 0; i < ArrowCount; i++)
        {
            ref Arrow arrow = ref arrows[i];
            
            arrow = new(randomPosition, networks[i], arrow.Rank);

            if (arrow.LastRank == selectedArrowIndex)
                selectedArrow = arrow;
        }
        
        Array.Sort(arrows);

        InitializeSimulation();
    }

    private void EvolveSimulation()
    {
        Array.Sort(networks);

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
        timeLeftBeforeReset = timeBetweenResets;

        targetPosition = new(random.NextSingle() * 0.5f * WindowWidth + WindowWidth * 0.25f, random.NextSingle() * 0.5f * WindowHeight + WindowHeight * 0.25f);

        currentIteration++;
    }

    private void SaveBestNetwork() => arrows[0].NeuralNetwork.Save(SavePath);

    private void LoadSavedNetwork()
    {
        NeuralNetwork saved = NeuralNetwork.Load(SavePath);
        
        for (int i = 0; i < ArrowCount; i++)
            networks[i] = new(saved);
        
        ResetSimulation(false);
    }
}
