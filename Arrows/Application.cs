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

    private readonly Random random = new();

    private const int ArrowCount = 200;
    private readonly Arrow[] arrows = new Arrow[ArrowCount];
    private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];
        
    private const int NetworkInputCount = 5;
    private readonly int[] networkHiddenLayersCount = [20, 20, 20, 20];
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
        WindowWidth = 1280;
        WindowHeight = 720;

        IsFixedTimeStep = false;
        graphics.SynchronizeWithVerticalRetrace = true;

        startingArrowPosition = new(WindowWidth * 0.1f, WindowHeight * 0.5f);
        startingTargetPosition = new(WindowWidth * 0.9f, WindowHeight * 0.5f);
        startingTargetOffsetY = WindowHeight * 0.4f;
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
                
            networks[i] = new(networkSize)
            {
                Rank = i
            };
            arrows[i] = new(startingArrowPosition, targetPosition, networks[i], -1);
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
                ResetSimulation();
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
            if (selectedArrow != null)
            {
                if (arrow == selectedArrow)
                    color = Color.Lime;
                else
                    color *= 0.05f;
            }
                
            arrow.Render(spriteBatch, color);
        }

        spriteBatch.End();

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui(gameTime);
        imGuiRenderer.EndLayout();
    }

    protected virtual void DrawImGui(GameTime gameTime)
    {
        ImGui.Begin("Simulation");

        ImGuiUtils.SeparatorText("Settings");
            
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
        
        ImGuiUtils.SeparatorText("Readonly data");

        double fps = 1.0 / gameTime.ElapsedGameTime.TotalSeconds;
        ImGui.Text($"FPS: {fps}");
        ImGui.Text($"Total time: {gameTime.TotalGameTime}");
        ImGui.Text($"Current iteration: {currentIteration}");
        double simulationSpeed = fps / simulationFrameRate;
        ImGui.Text($"Running at {fps.ToString("F2", CultureInfo.InvariantCulture)}x speed");
        ImGui.Text($"{(simulationSpeed / timeBetweenResets).ToString("F3", CultureInfo.InvariantCulture)} iterations per second");
        
        ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {timeLeftBeforeReset}s");
        ImGui.Text($"Target position {targetPosition}");
            
        ImGuiUtils.SeparatorText("Actions");

        ImGui.Checkbox("Running", ref running);

        if (ImGui.Button("Next generation"))
            ResetSimulation();

        if (ImGui.Button("Next frame"))
            runningForOneFrame = true;

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
            ImGuiUtils.SeparatorText("Selected arrow data");
                
            ImGui.Text($"Position {selectedArrow.Position}");
            ImGui.Text($"Angle {selectedArrow.Angle}rad = {MathHelper.ToDegrees(selectedArrow.Angle)}deg");
            Vector2 direction = new(MathF.Cos(selectedArrow.Angle), -MathF.Sin(selectedArrow.Angle));
            ImGuiUtils.DirectionVector("Direction", ref direction, selectedArrow.TargetDirection);
                    
            ImGui.Text($"Current rank {selectedArrow.Rank}");
            ImGui.Text($"Last rank {selectedArrow.LastRank}");

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

    private void ResetSimulation()
    {
        Array.Sort(networks);

        // Mutate the worst 50%
        const float MutatedArrowRatio = 0.5f;
        const int MutatedArrowCount = (int) (ArrowCount * MutatedArrowRatio);
        const int NonMutatedArrowCount = ArrowCount - MutatedArrowCount;
        for (int i = 0; i < MutatedArrowCount; i++)
        {
            NeuralNetwork goodNetwork = networks[i];
            
            networks[NonMutatedArrowCount + i] = new(goodNetwork);
            
            NeuralNetwork badNetwork = networks[NonMutatedArrowCount + i];
            
            badNetwork.Mutate();
        }

        for (int i = 0; i < ArrowCount; i++)
        {
            networks[i].Rank = i;
            networks[i].Fitness = 0.0;
        }

        int selectedArrowIndex = selectedArrow?.Rank ?? -1;

        for (int i = 0; i < ArrowCount; i++)
        {
            arrows[i] = new(startingArrowPosition, targetPosition, networks[i], arrows[i].Rank);

            Arrow arrow = arrows[i];
            if (arrow.LastRank == selectedArrowIndex)
                selectedArrow = arrow;
        }
        
        Array.Sort(arrows);

        InitializeSimulation();
    }

    private void InitializeSimulation()
    {
        timeLeftBeforeReset = timeBetweenResets;

        targetPosition = startingTargetPosition + Vector2.UnitY * startingTargetOffsetY;

        currentIteration++;
    }

    private void SaveBestNetwork() => arrows[0].NeuralNetwork.Save(SavePath);

    private void LoadSavedNetwork()
    {
        for (int i = 0; i < arrows.Length; i++)
            arrows[i] = new(startingArrowPosition, targetPosition, NeuralNetwork.Load(SavePath), -1);
        
        ResetSimulation();
    }
}
